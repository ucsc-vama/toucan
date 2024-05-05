#include "circt/Dialect/HW/HWTypes.h"
#include "circt/Support/LLVM.h"
#include "circt/Dialect/Comb/CombDialect.h"
#include "circt/Dialect/Comb/CombOps.h"
#include "circt/Dialect/Seq/SeqOps.h"

#include "mlir/IR/Value.h"
#include "mlir/Pass/AnalysisManager.h"
#include "mlir/Support/LLVM.h"
#include "toucan/ToucanAnalysis.h"
#include "toucan/ToucanAttributes.h"
#include "toucan/ToucanOps.h"
#include "toucan/ToucanTypes.h"
#include "toucan/ToucanUtils.h"

#include "llvm/ADT/STLExtras.h"
#include "llvm/ADT/SmallVector.h"
#include "llvm/Support/Casting.h"
#include "llvm/Support/Debug.h"
#include "llvm/Support/ErrorHandling.h"
#include <_types/_uint8_t.h>
#include <cstddef>
#include <cstdint>

#include <boost/graph/topological_sort.hpp>
#include <cstring>
#include <iterator>
#include <array>
#include <optional>
#include <tuple>
#include <vector>


using namespace toucan;

using namespace mlir;
using namespace llvm;
using namespace circt;

void SingleRegionScheduler::levelizePartitions(DesignGraph &graph) {
  std::vector<uint32_t> topo_order;
  topo_order.reserve(boost::num_vertices(graph.g));
  boost::topological_sort(graph.g, std::back_inserter(topo_order));
  std::reverse(topo_order.begin(), topo_order.end());

  // Initialize levels
  std::vector<uint32_t> levels(boost::num_vertices(graph.g), 0);
  std::vector<uint32_t> sinkVtxes;

  // Assign levels based on dependencies
  for (auto v : topo_order) {
    if (boost::out_degree(v, graph.g) == 0) {
      // a sink node
      sinkVtxes.push_back(v);
      levels[v] = UINT32_MAX;
    } else if (boost::in_degree(v, graph.g) != 0) {
      uint32_t max_pred_level = 0;
      for (auto ei = boost::in_edges(v, graph.g); ei.first != ei.second; ++ei.first) {
          auto u = boost::source(*ei.first, graph.g);
          max_pred_level = std::max(max_pred_level, levels[u]);
      }
      auto v_level = max_pred_level + 1;
      levels[v] = v_level;
      assert(v_level < UINT32_MAX);
    }
  }

  assert(!sinkVtxes.empty());

  // init
  for (size_t partId = 0; partId < partitions.size(); ++partId) {
    partLevels.emplace_back();
  }

  // save level in partition
  for (uint32_t vtx = 0; vtx < boost::num_vertices(graph.g); ++vtx) {
    auto vtxLevel = levels[vtx];
    if (vtxLevel == UINT32_MAX) continue;

    auto vtxPartition = vtxIdToPartId[vtx];
    // grow container if needed
    for (size_t levelId = partLevels[vtxPartition].size(); levelId <= vtxLevel; levelId++) {
      partLevels[vtxPartition].emplace_back();
    }
    partLevels[vtxPartition][vtxLevel].push_back(vtx);
  }

  // Condense, remove empty levels
  partLevels.erase(std::remove_if(partLevels.begin(), partLevels.end(), [](auto innerVec) {
    return innerVec.empty();
  }), partLevels.end());

  // push all sink vtx to last level
  for (auto &eachPart: partLevels) {
    eachPart.emplace_back();
  }

  assert(!sinkVtxes.empty());
  for (auto sinkVtx: sinkVtxes) {
    auto partId = vtxIdToPartId[sinkVtx];
    auto numParts = partLevels.size();
    assert(partId < numParts);
    partLevels[partId].back().push_back(sinkVtx);
  }
  
  // debug
  for (size_t partId = 0; partId < partLevels.size(); partId++) {
    auto &currentPart = partLevels[partId];
    llvm::dbgs() << "Partition " << partId << " has " << currentPart.size() << " levels\n";
    for (size_t levelId = 0; levelId < currentPart.size(); levelId++) {
      auto &currentLevel = currentPart[levelId];
      llvm::dbgs() << "  Level " << levelId << " has " << currentLevel.size() << " verticies\n";

      // if (currentLevel.size() == 1) {
      //   // level with only 1 node. 
      //   auto vtxId = currentLevel.front();
      //   auto op = graph.g[vtxId].op;
      //   op->print(llvm::dbgs());
      //   llvm::dbgs() << "\n";
      // }
    }
  }
  return;
}


void SingleRegionScheduler::generateRegMemLayout(DesignGraph &graph, uint32_t partitionRegPaddingSpace, uint32_t memPaddingSpace) {
  // collect all reg and memory, generate layout
  for (size_t partId = 0; partId < partitions.size(); partId++) {
    SmallVector<TypedValue<toucan::RegType>> regsInPartitionOrdered;
    mlir::DenseSet<TypedValue<toucan::RegType>> regsInPartition;

    auto &firstLevel = partLevels[partId][0];
    auto &lastLevel = partLevels[partId].back();

    // insert first level
    for (auto vtxId: firstLevel) {
      auto vtxOpName = graph.g[vtxId].toucanOpName;
      if (vtxOpName == CGToucanOPName::RegRead) {
        auto regReadOp = cast<toucan::RegReadOp>(graph.g[vtxId].op);
        auto regVal = regReadOp.getReg();
        assert(!regsInPartition.contains(regVal));
        regsInPartition.insert(regVal);
        regsInPartitionOrdered.push_back(regVal);
      }
    }
    // Check if there exists write-only registers.
    // They are generally external module output or Top module output
    for (auto vtxId: lastLevel) {
      auto vtxOpName = graph.g[vtxId].toucanOpName;
      if (vtxOpName == CGToucanOPName::RegWrite) {
        auto regWriteOp = cast<toucan::RegWriteOp>(graph.g[vtxId].op);
        auto regVal = regWriteOp.getReg();
        if (!regsInPartition.contains(regVal)) {
          regsInPartition.insert(regVal);
          regsInPartitionOrdered.push_back(regVal);
        }
      }
    }

    // Insert any register that not found in previous step
    for (auto regVal: graph.regs) {
      if (!regsInPartition.contains(regVal)) {
        regsInPartition.insert(regVal);
        regsInPartitionOrdered.push_back(regVal);
      }
    }

    for (auto regVal: regsInPartitionOrdered) {
      // for each vtx in current partition
      auto regDefiningOp = regVal.getDefiningOp();

      CGRegMetaInfo regMeta;

      regMeta.namehint = getSVNameHintAttr(regDefiningOp);
      auto fragmentIdAttr = getSignalFragmentIDAttr(regDefiningOp);
      if (fragmentIdAttr) {
        regMeta.fragment_id = fragmentIdAttr->getInt();
      } else {
        regMeta.fragment_id = UINT32_MAX;
      }
      regMeta.bitWidth = regVal.getType().getElementWidth();
      regMeta.isPadding = false;
      regMeta.isIO = hasIOSignalMarker(regDefiningOp);

      auto regId = codeGenInfo.regPool.size();
      codeGenInfo.regPool.push_back(regMeta);
      codeGenInfo.toucanRegToId[regVal] = regId;
    }

    // Add extra padding
    for (size_t i = 0; i < partitionRegPaddingSpace; i++) {
      CGRegMetaInfo paddingRegMeta;

      paddingRegMeta.isPadding = true;
      paddingRegMeta.bitWidth = 0;
      paddingRegMeta.fragment_id = 0;
      paddingRegMeta.isIO = false;

      // auto regId = codeGenInfo.regPool.size();
      codeGenInfo.regPool.push_back(paddingRegMeta);
    }
  }

  // collect all reg and memory, generate layout
  // Here we assume each memory has at least 1 writer
  uint64_t memBaseAddr = 0;
  for (size_t vtxId = 0; vtxId < boost::num_vertices(graph.g); vtxId++) {
    // for each mem write
    auto vtxOpName = graph.g[vtxId].toucanOpName;
    if (vtxOpName == CGToucanOPName::MemWrite) {
      // Note: For now, mems with multiple write ports are still merged, so at this time, each memory will only have 1 writer.
      auto memWriteOp = cast<toucan::MemWriteOp>(graph.g[vtxId].op);
      auto memVal = memWriteOp.getMem();
      auto memDefiningOp = memVal.getDefiningOp();

      CGMemMetaInfo memMeta;

      memMeta.namehint = getSVNameHintAttr(memDefiningOp);
      auto fragmentIdAttr = getSignalFragmentIDAttr(memDefiningOp);
      if (fragmentIdAttr) {
        memMeta.fragment_id = fragmentIdAttr->getInt();
      } else {
        memMeta.fragment_id = UINT32_MAX;
      }
      memMeta.bitWidth = memVal.getType().getElementWidth();
      memMeta.memDepth = memVal.getType().getDepth();
      memMeta.hasMultipleWriter = (graph.g[vtxId].weight > 1);

      // 
      assert(memMeta.bitWidth <= 4);
      uint64_t memCapacity = (memMeta.hasMultipleWriter) ? memMeta.memDepth * 4 : memMeta.memDepth;
      memMeta.memBase = memBaseAddr;
      memBaseAddr += (memCapacity + memPaddingSpace);

      auto memId = codeGenInfo.memPool.size();
      codeGenInfo.memPool.push_back(memMeta);
      codeGenInfo.toucanMemToId[memVal] = memId; 
    }
  }
  codeGenInfo.totalMemSize = memBaseAddr;
}

void SingleRegionScheduler::collectPrintString(DesignGraph &graph) {
  uint32_t stringId = 0;

  for (uint32_t vtxId = 0; vtxId < boost::num_vertices(graph.g); vtxId++) {
    auto vtxOpName = graph.g[vtxId].toucanOpName;
    if (vtxOpName == CGToucanOPName::Print) {
      auto printOp = cast<toucan::PrintOp>(graph.g[vtxId].op);

      auto printStr = printOp.getMsg();
      if (!codeGenInfo.printStrings.contains(printStr)) {
        // a new string
        codeGenInfo.printStrings[printStr] = stringId;
        stringId++;
      }
    }
  }
}

void SingleRegionScheduler::collectConstant(DesignGraph &graph, CGPartitionMetaInfo &partInfo, uint32_t partId) {
  // Collect all consts, populate value pool
  for (uint32_t vtxId = 0; vtxId < boost::num_vertices(graph.g); vtxId++) {
    if (vtxIdToPartId[vtxId] == partId) {
      // for each vtx in current partition
      auto vtxOpName = graph.g[vtxId].toucanOpName;
      if (vtxOpName == CGToucanOPName::ConstDecl) {
        auto op = graph.g[vtxId].op;
        
        // partInfo.opToResultValueId[op] = partInfo.valuePool.size();
        // auto constResultVal = op->getResult(0);
        // assert(!partInfo.valueToValId.contains(constResultVal));
        // partInfo.valueToValId[constResultVal] = partInfo.valuePool.size();

        if (auto constOp = dyn_cast<toucan::ConstantOp>(op)) {
          // regular const
          auto constVal = constOp.getValue();
          auto bitWidth = constVal.getBitWidth();
          assert(bitWidth <= 4);
          auto rawVal = static_cast<uint8_t>(constVal.getZExtValue());
          // save op result value
          auto valId = partInfo.valuePool.size();

          auto constResultVal = constOp.getResult();
          assert(!partInfo.valueToValId.contains(constResultVal));
          partInfo.valueToValId[constResultVal] = valId;

          partInfo.valuePool.push_back({true, false, rawVal, op, 0, 0, static_cast<uint8_t>(bitWidth), std::nullopt, 0});
        } else {
          // it must be a vector const
          auto defConstVecOp = cast<toucan::DefConstVectorOp>(op);
          // save op result value
          // Vec result map to first vec element
          auto vecHandle = defConstVecOp.getHandle();
          auto bitWidth = vecHandle.getType().getElementWidth();
          assert(bitWidth <= 4);

          auto valId = partInfo.valuePool.size();
          assert(!partInfo.valueToValId.contains(vecHandle));
          partInfo.valueToValId[vecHandle] = valId;

          // Why reverse? vector decl op elements are MSB first. Reorder to LSB first to make vecRead's life easier
          for (auto &vecValElem: llvm::reverse(defConstVecOp.getValues().getValue())) {
            auto elemVal = cast<mlir::IntegerAttr>(vecValElem).getValue();
            auto elemValWidth = elemVal.getBitWidth();
            assert(elemValWidth <= 4);

            auto elemValMask = static_cast<uint8_t>((1 << elemValWidth) - 1);
            uint8_t rawVal = elemValMask & static_cast<uint8_t>(elemVal.getZExtValue());
            partInfo.valuePool.push_back({true, false, rawVal, op, 0, 0, static_cast<uint8_t>(bitWidth), std::nullopt, 0});
          }
        }
      }
    }
  }
}

static void populateOpMetaDebugInfo(CGOpMetaInfo &opMeta, Operation *op) {
  opMeta.namehint = getSVNameHintAttr(op);

  auto fragmentIdAttr = getSignalFragmentIDAttr(op);
  if (fragmentIdAttr) {
    opMeta.fragment_id = fragmentIdAttr->getInt();
  } else {
    opMeta.fragment_id = UINT32_MAX;
  }
}

void SingleRegionScheduler::schedule(DesignGraph &graph, uint32_t partitionRegPaddingSpace, uint32_t memPaddingSpace) {
  // TODO: parallelize for each partition

  // Collect information for code gen
  // This function also determines register layout
  generateRegMemLayout(graph, partitionRegPaddingSpace, memPaddingSpace);

  // dedup strings
  collectPrintString(graph);


  // Temporary storage for operations inside current level
  mlir::SmallVector<CGOpMetaInfo> currentLevelOps;

  for (size_t partId = 0; partId < partitions.size(); partId++) {
    CGPartitionMetaInfo partInfo;
    std::memset(&partInfo.opStatistics, 0, sizeof(CGOpStatistics));

    // A const zero for all luts
    CGValueMetaInfo zeroConst = {true, false, 0, nullptr, 0, 0, 0, std::nullopt, 0};
    partInfo.valuePool.push_back(zeroConst);

    // Collect all constants
    collectConstant(graph, partInfo, partId);
    // Utill now, all elements inside valuePool are constant
    partInfo.numConstsInValuePool = partInfo.valuePool.size();

    // Note: first level should contains regread and memread
    // middle levels should contain lut or vecread
    // last level should contains print, stop, regwrite, memwrite
    auto &firstLevel = partLevels[partId][0];
    auto &lastLevel = partLevels[partId].back();

    {
      // First level. should only contains regread or const decl, the later will be pushed into const pool
      currentLevelOps.clear();
      currentLevelOps.reserve(firstLevel.size());
      for (auto vtxId: firstLevel) {
        auto tOpName = graph.g[vtxId].toucanOpName;

        auto op = graph.g[vtxId].op;
        CGOpMetaInfo opMeta;
        opMeta.opName = tOpName;
        opMeta.op = op;
        opMeta.vtxId = vtxId;
        populateOpMetaDebugInfo(opMeta, op);

        if (tOpName == CGToucanOPName::RegRead) {
          // a regread
          auto regReadOp = cast<toucan::RegReadOp>(op);
          auto regVal = regReadOp.getReg();
          assert(codeGenInfo.toucanRegToId.contains(regVal) && "A register that never seen was read!");
          auto regValId = codeGenInfo.toucanRegToId[regVal];

          opMeta.regRead.reg = regValId;
          currentLevelOps.push_back(opMeta);
        } else if (tOpName == CGToucanOPName::ConstDecl) {
          // A constant decl. Do nothing.
        } else {
          assert(false && "Should not reach here");
        }
      }
      // Allocate result storage
      for (size_t opId = 0; opId < currentLevelOps.size(); opId++) {
        auto &opMeta = currentLevelOps[opId];

        assert(opMeta.opName == CGToucanOPName::RegRead);
        auto valId = partInfo.valuePool.size();

        // new val
        CGValueMetaInfo valMeta;
        valMeta.isConst = false;
        valMeta.isPlaceholder = false;
        valMeta.value = 0;
        valMeta.definingOp = opMeta.op;
        valMeta.levelId = 0;
        valMeta.opId = opId;
        valMeta.bitWidth = static_cast<uint8_t>(hw::getBitWidth(opMeta.op->getResult(0).getType()));
        valMeta.namehint = opMeta.namehint;
        valMeta.fragment_id = opMeta.fragment_id;

        partInfo.valuePool.push_back(valMeta);
        auto resultVal = opMeta.op->getResult(0);
        assert(!partInfo.valueToValId.contains(resultVal));
        partInfo.valueToValId[resultVal] = valId;
        opMeta.setResult(valId);
      }

      // Save statistics
      CGLayerValueStatistics stats;
      std::memset(&stats, 0, sizeof(CGLayerValueStatistics));
      stats.numRegReads = currentLevelOps.size();
      partInfo.opStatisticsPerLevel.push_back(stats);

      partInfo.opStatistics.numRegReads = currentLevelOps.size();
      // Save ops
      partInfo.opPool.push_back(std::move(currentLevelOps));
    }



    // Codegen for middle layers
    // At least 3 layers: 1 for input, 1 for lut, the last 1 for output
    assert(partLevels[partId].size() > 2);

    // lut ops. May be reordered for performance reason
    mlir::SmallVector<CGOpMetaInfo> lutOps;
    // vec produced by nops. ** Don't reorder **
    // each defvector is converted to a list of LUT_Nop
    mlir::SmallVector<mlir::SmallVector<CGOpMetaInfo>> vecDecls;
    mlir::SmallVector<mlir::TypedValue<toucan::VecType>> vecDeclVals;
    // vec reads
    mlir::SmallVector<mlir::SmallVector<CGOpMetaInfo>> vecReadOps;
    mlir::SmallVector<mlir::Value> vecReadHandleVals;
    // mem reads. reorder is not necessary
    mlir::SmallVector<CGOpMetaInfo> memReadOps;

    mlir::SmallVector<CGOpMetaInfo> currentVecReadOps;
    mlir::SmallVector<CGOpMetaInfo> currentVecDeclOps;
    std::array<uint32_t, 3> lutOpValueIds;
    std::array<uint32_t, 4> vecReadOpIndexIds;
    std::array<uint32_t, 8> memReadOpIndexIds;

    for (size_t layerId = 1; layerId < partLevels[partId].size() - 1; layerId++) {
      auto &currentLevel = partLevels[partId][layerId];

      currentLevelOps.clear();

      lutOps.clear();
      vecReadOps.clear();
      vecReadHandleVals.clear();
      memReadOps.clear();
      vecDecls.clear();
      vecDeclVals.clear();

      for (auto vtxId: currentLevel) {
        auto tOpName = graph.g[vtxId].toucanOpName;
        auto rawOp = graph.g[vtxId].op;
        // auto vtxWeight = graph.g[vtxId].weight;

        CGOpMetaInfo opMeta;
        opMeta.opName = tOpName;
        opMeta.op = rawOp;
        opMeta.vtxId = vtxId;
        populateOpMetaDebugInfo(opMeta, rawOp);

        if (tOpName == CGToucanOPName::LUT) {
          // lut
          auto lutOp = cast<toucan::LUTOp>(rawOp);

          // fill input oprands
          auto lutInputs = lutOp.getInputs();
          auto numLutOprands = lutInputs.size();
          assert(numLutOprands <= 3);
          size_t pos = 0;
          for (; pos < 3 - numLutOprands; pos++) {
            // Note: the first elem in value pool is const 0
            lutOpValueIds[pos] = 0;
          }
          for (auto val: lutOp.getInputs()) {
            assert(partInfo.valueToValId.contains(val));
            auto valId = partInfo.valueToValId[val];
            lutOpValueIds[pos] = valId;
            pos++;
          }

          auto rawOpName = static_cast<uint32_t>(lutOp.getOpName());
          assert(rawOpName <= toucan::getMaxEnumValForLUTOpName());
          opMeta.lut.lutId = static_cast<uint8_t>(rawOpName);
          opMeta.lut.op0 = lutOpValueIds[0];
          opMeta.lut.op1 = lutOpValueIds[1];
          opMeta.lut.op2 = lutOpValueIds[2];
          opMeta.lut.numOprands = numLutOprands;

          populateOpMetaDebugInfo(opMeta, rawOp);

          lutOps.push_back(opMeta);
        } else if (tOpName == CGToucanOPName::VecDecl) {
          // vec decl, expand to list of nops
          currentVecDeclOps.clear();
          auto vecDeclOp = cast<toucan::DefVectorOp>(rawOp);
          // Why reverse? vector decl op elements are MSB first. Reorder to LSB first to make vecRead's life easier
          for (auto elemVal: llvm::reverse(vecDeclOp.getInputs())) {
            // Create a NOP
            CGOpMetaInfo opMeta;
            opMeta.opName = CGToucanOPName::LUT;
            opMeta.op = rawOp;
            opMeta.vtxId = vtxId;
            // nop, the op code should be 0
            opMeta.lut.lutId = static_cast<uint8_t>(LUTOpName::LUT_Nop);
            opMeta.lut.op0 = 0;
            opMeta.lut.op1 = 0;
            assert(partInfo.valueToValId.contains(elemVal));
            opMeta.lut.op2 = partInfo.valueToValId[elemVal];
            opMeta.lut.numOprands = 1;

            // No namehint

            currentVecDeclOps.push_back(opMeta);
          }
          // Nops are ordered.
          vecDecls.push_back(currentVecDeclOps);
          vecDeclVals.push_back(vecDeclOp.getHandle());
        } else if (tOpName == CGToucanOPName::VecRead) {
          currentVecReadOps.clear();

          auto vecReadOp = cast<toucan::VectorReadOp>(rawOp);
          auto vecHandle = vecReadOp.getHandle();
          assert(partInfo.valueToValId.contains(vecHandle));
          auto vecHandleId = partInfo.valueToValId[vecHandle];
          auto vecLength = vecHandle.getType().getLength();

          for (auto userOp: vecHandle.getUsers()) {
            // Note: Only 1 user has vtxId. Other ops are not included in currentLevelOps
            auto userReadOp = cast<toucan::VectorReadOp>(userOp);

            // offset: i16
            auto offset = userReadOp.getOffset().getZExtValue();
            auto outRangeValue = userReadOp.getOutRangeValue();


            // collect index vecReadOpIndexIds
            auto indexValues = userReadOp.getIndicies();
            auto numIndexValues = indexValues.size();
            assert(numIndexValues <= 4 && "Index is too long");

            size_t pos = 0;
            for (; pos < 4 - numIndexValues; pos++) {
              // Note: the first elem in value pool is const 0
              vecReadOpIndexIds[pos] = 0;
            }
            for (auto val: indexValues) {
              assert(partInfo.valueToValId.contains(val));
              auto valId = partInfo.valueToValId[val];
              vecReadOpIndexIds[pos] = valId;
              pos++;
            }


            CGOpMetaInfo opMeta;
            opMeta.opName = CGToucanOPName::VecRead;
            opMeta.op = userOp;
            // Share same vtxId
            opMeta.vtxId = vtxId;
            // nop, the op code should be 0
            opMeta.vec.vecBase = vecHandleId;
            opMeta.vec.vecLength = vecLength;
            opMeta.vec.index0 = vecReadOpIndexIds[0];
            opMeta.vec.index1 = vecReadOpIndexIds[1];
            opMeta.vec.index2 = vecReadOpIndexIds[2];
            opMeta.vec.index3 = vecReadOpIndexIds[3];

            assert(partInfo.valueToValId.contains(outRangeValue));
            opMeta.vec.outRangeValue = partInfo.valueToValId[outRangeValue];
            opMeta.vec.offset = static_cast<uint16_t>(offset);

            populateOpMetaDebugInfo(opMeta, userOp);

            currentVecReadOps.push_back(opMeta);
          }

          // Sort by offset for performance reason
          std::sort(currentVecReadOps.begin(), currentVecReadOps.end(), 
            [](const CGOpMetaInfo& a, const CGOpMetaInfo& b) {return a.vec.offset < b.vec.offset;});

          vecReadOps.push_back(std::move(currentVecReadOps));
          vecReadHandleVals.push_back(vecHandle);

        } else if (tOpName == CGToucanOPName::MemRead) {
          // a memread
          auto memReadOp = cast<toucan::MemReadOp>(rawOp);
          auto memVal = memReadOp.getMem();
          auto memValId = codeGenInfo.toucanMemToId[memVal];
          auto memEnVal = memReadOp.getEn();
          assert(partInfo.valueToValId.contains(memEnVal));
          auto memEnId = partInfo.valueToValId[memEnVal];

          auto memAddrs = memReadOp.getAddrs();
          auto numMemAddrs = memAddrs.size();
          assert(numMemAddrs <= 8 && "Memory address is too long");

          size_t pos = 0;
          for (; pos < 8 - numMemAddrs; pos++) {
            // Note: the first elem in value pool is const 0
            memReadOpIndexIds[pos] = 0;
          }
          for (auto val: memAddrs) {
            assert(partInfo.valueToValId.contains(val));
            auto valId = partInfo.valueToValId[val];
            memReadOpIndexIds[pos] = valId;
            pos++;
          }

          opMeta.memRead.hasMultipleWriter = codeGenInfo.memPool[memValId].hasMultipleWriter;
          opMeta.memRead.memBase = codeGenInfo.memPool[memValId].memBase;
          opMeta.memRead.memDepth = codeGenInfo.memPool[memValId].memDepth;

          opMeta.memRead.en = memEnId;
          opMeta.memRead.addr0 = memReadOpIndexIds[0];
          opMeta.memRead.addr1 = memReadOpIndexIds[1];
          opMeta.memRead.addr2 = memReadOpIndexIds[2];
          opMeta.memRead.addr3 = memReadOpIndexIds[3];
          opMeta.memRead.addr4 = memReadOpIndexIds[4];
          opMeta.memRead.addr5 = memReadOpIndexIds[5];
          opMeta.memRead.addr6 = memReadOpIndexIds[6];
          opMeta.memRead.addr7 = memReadOpIndexIds[7];

          populateOpMetaDebugInfo(opMeta, rawOp);
          
          memReadOps.push_back(opMeta);
        } else {
          assert(false && "other type of op should not appear in middle levels");
        }
      }

      // reorder luts for this layer
      auto findFirstOpValId = [](const CGOpMetaInfo &opMeta) {
        switch (opMeta.lut.numOprands) {
          case 1: return opMeta.lut.op2;
          case 2: return std::min(opMeta.lut.op1, opMeta.lut.op2);
          case 3: return std::min(std::min(opMeta.lut.op0, opMeta.lut.op1), opMeta.lut.op2);
          default:
            assert(false && "numOprands could only be 1, 2, or 3");
        }
      };
      // order by smallest input value id
      std::sort(lutOps.begin(), lutOps.end(), [&](const CGOpMetaInfo &a, const CGOpMetaInfo &b) {
        return findFirstOpValId(a) < findFirstOpValId(b);
      });


      // Record number of luts/memreads/vecreads for later performance tuning
      CGLayerValueStatistics stats;
      std::memset(&stats, 0, sizeof(CGLayerValueStatistics));
      stats.numMemReads = memReadOps.size();
      stats.numVecReads = vecReadOps.size();
      stats.numLuts = lutOps.size();
      // Note: vecDecl ops are lowered to Nops, which is also lut type
      for (auto &eachVecDeclOps: vecDecls) {
        stats.numLuts += eachVecDeclOps.size();
        partInfo.opStatistics.numLutNops += eachVecDeclOps.size();
      }
      partInfo.opStatisticsPerLevel.push_back(stats);

      partInfo.opStatistics.numMemReads += stats.numMemReads;
      partInfo.opStatistics.numVecReads += stats.numVecReads;
      partInfo.opStatistics.numLuts += stats.numLuts;

      // Within each layer, place memReads first, then vecReads, luts are the last
      // Allocate storage for this layer
      auto expectedOpCount = stats.numMemReads + stats.numVecReads + stats.numLuts;
      currentLevelOps.reserve(expectedOpCount);

      assert(currentLevelOps.size() == 0);
      // save op, allocate result storage
      auto allocateResultForMiddleLayer = [&](CGOpMetaInfo &opMeta) {
        auto valId = partInfo.valuePool.size();
        auto resultVal = opMeta.op->getResult(0);

        CGValueMetaInfo valMeta;
        valMeta.isConst = false;
        valMeta.isPlaceholder = false;
        valMeta.value = 0;
        valMeta.definingOp = opMeta.op;
        valMeta.levelId = layerId;
        valMeta.opId = currentLevelOps.size();
        valMeta.bitWidth = static_cast<uint8_t>(hw::getBitWidth(opMeta.op->getResult(0).getType()));
        valMeta.namehint = opMeta.namehint;
        valMeta.fragment_id = opMeta.fragment_id;

        assert(!partInfo.valueToValId.contains(resultVal));
        partInfo.valueToValId[resultVal] = valId;
        partInfo.valuePool.push_back(valMeta);

        opMeta.setResult(valId);
        currentLevelOps.push_back(opMeta);
      };

      // place mem reads
      for (auto &opMeta: memReadOps) {
        assert(opMeta.opName == CGToucanOPName::MemRead);
        allocateResultForMiddleLayer(opMeta);
      }

      // place vecReads
      assert(vecReadOps.size() == vecReadHandleVals.size());
      for (auto [singleVecReadOps, vecHandle]: llvm::zip(vecReadOps, vecReadHandleVals)) {
        // each elem inside singleVecReadOps reads vector vecHandle
        for (auto &opMeta: singleVecReadOps) {
          assert(opMeta.opName == CGToucanOPName::VecRead);
          allocateResultForMiddleLayer(opMeta);
        }
      }

      // place luts
      for (auto &opMeta: lutOps) {
        assert(opMeta.opName == CGToucanOPName::LUT);
        allocateResultForMiddleLayer(opMeta);
      }

      // place vecdefs
      for (auto [singleVecDefOps, vecHandle]: llvm::zip(vecDecls, vecDeclVals)) {
        auto bitWidth = static_cast<uint8_t>(vecHandle.getType().getElementWidth());
        for (size_t i = 0; i < singleVecDefOps.size(); ++i) {
          auto &opMeta = singleVecDefOps[i];
          assert(opMeta.opName == CGToucanOPName::LUT);
          assert(opMeta.lut.lutId == static_cast<uint8_t>(toucan::LUTOpName::LUT_Nop));

          auto valId = partInfo.valuePool.size();

          CGValueMetaInfo valMeta;
          valMeta.isConst = false;
          valMeta.value = 0;
          valMeta.definingOp = opMeta.op;
          valMeta.levelId = layerId;
          valMeta.opId = currentLevelOps.size();
          valMeta.namehint = opMeta.namehint;
          valMeta.bitWidth = bitWidth;
          valMeta.fragment_id = opMeta.fragment_id;
          // First op produces the vecHandle. 
          // Other ops produces an invisible placeholder value
          valMeta.isPlaceholder = (i != 0);
          if (i == 0) {
            assert(!partInfo.valueToValId.contains(vecHandle));
            partInfo.valueToValId[vecHandle] = valId;
          } 
          partInfo.valuePool.push_back(valMeta);

          opMeta.setResult(valId);
          currentLevelOps.push_back(opMeta);
        }
      }

      // Save ops
      partInfo.opPool.push_back(std::move(currentLevelOps));
    }

    // Code gen for last level
    {
      mlir::SmallVector<CGOpMetaInfo> regWriteOps;
      mlir::SmallVector<CGOpMetaInfo> memWriteOps;
      mlir::SmallVector<CGOpMetaInfo> printOps;
      mlir::SmallVector<CGOpMetaInfo> stopOps;

      std::array<uint32_t, 8> memReadOpIndexIds;

      currentLevelOps.clear();
      currentLevelOps.reserve(lastLevel.size());

      for (auto vtxId: lastLevel) {
        auto vtxOpName = graph.g[vtxId].toucanOpName;
        auto rawOp = graph.g[vtxId].op;

        CGOpMetaInfo opMeta;
        opMeta.op = rawOp;
        opMeta.opName = vtxOpName;
        opMeta.vtxId = vtxId;

        if (vtxOpName == CGToucanOPName::RegWrite) {
          // regwrite
          auto regWriteOp = cast<toucan::RegWriteOp>(rawOp);
          auto regVal = regWriteOp.getReg();
          auto dataVal = regWriteOp.getData();

          assert(codeGenInfo.toucanRegToId.contains(regVal) && "A register never seen was written!");
          auto regValId = codeGenInfo.toucanRegToId[regVal];
          assert(partInfo.valueToValId.contains(dataVal) && "A value never seen was read!");
          auto dataValId = partInfo.valueToValId[dataVal];

          opMeta.regWrite.reg = regValId;
          opMeta.regWrite.dat = dataValId;

          // Namehint not needed for last level ops
          regWriteOps.push_back(opMeta);

        } else if (vtxOpName == CGToucanOPName::MemWrite) {
          // memWrite
          auto memWriteOp = cast<toucan::MemWriteOp>(rawOp);
          auto memVal = memWriteOp.getMem();
          auto dataVal = memWriteOp.getData();
          auto enVal = memWriteOp.getEn();

          assert(codeGenInfo.toucanMemToId.contains(memVal));
          auto memValId = codeGenInfo.toucanMemToId[memVal];
          assert(partInfo.valueToValId.contains(dataVal));
          auto dataValId = partInfo.valueToValId[dataVal];
          assert(partInfo.valueToValId.contains(enVal));
          auto enValId = partInfo.valueToValId[enVal];

          auto memAddrs = memWriteOp.getAddrs();
          auto numMemAddrs = memAddrs.size();
          assert(numMemAddrs <= 8 && "Memory address is too long");

          size_t pos = 0;
          for (; pos < 8 - numMemAddrs; pos++) {
            // Note: the first elem in value pool is const 0
            memReadOpIndexIds[pos] = 0;
          }
          for (auto val: memAddrs) {
            assert(partInfo.valueToValId.contains(val));
            auto valId = partInfo.valueToValId[val];
            memReadOpIndexIds[pos] = valId;
            pos++;
          }

          opMeta.memWrite.hasMultipleWriter = codeGenInfo.memPool[memValId].hasMultipleWriter;
          opMeta.memWrite.memBase = codeGenInfo.memPool[memValId].memBase;
          opMeta.memWrite.memDepth = codeGenInfo.memPool[memValId].memDepth;

          opMeta.memWrite.addr0 = memReadOpIndexIds[0];
          opMeta.memWrite.addr1 = memReadOpIndexIds[1];
          opMeta.memWrite.addr2 = memReadOpIndexIds[2];
          opMeta.memWrite.addr3 = memReadOpIndexIds[3];
          opMeta.memWrite.addr4 = memReadOpIndexIds[4];
          opMeta.memWrite.addr5 = memReadOpIndexIds[5];
          opMeta.memWrite.addr6 = memReadOpIndexIds[6];
          opMeta.memWrite.addr7 = memReadOpIndexIds[7];

          opMeta.memWrite.dat = dataValId;
          opMeta.memWrite.en = enValId;

          // Namehint not needed for last level ops
          memWriteOps.push_back(opMeta);

        } else if (vtxOpName == CGToucanOPName::Print) {
          // print
          auto printOp = cast<toucan::PrintOp>(rawOp);
          auto printStr = printOp.getMsg();
          auto enVal = printOp.getEn();

          assert(partInfo.valueToValId.contains(enVal));
          auto enValId = partInfo.valueToValId[enVal];
          
          assert(codeGenInfo.printStrings.contains(printStr));
          auto printStrId = codeGenInfo.printStrings[printStr];

          opMeta.print.en = enValId;
          opMeta.print.msg = printStrId;

          // Namehint not needed for last level ops
          printOps.push_back(opMeta);

        } else if (vtxOpName == CGToucanOPName::Stop) {
          // stop
          auto stopOp = cast<toucan::StopOp>(rawOp);
          auto enVal = stopOp.getEn();

          assert(partInfo.valueToValId.contains(enVal));
          auto enValId = partInfo.valueToValId[enVal];

          opMeta.stop.en = enValId;

          // Namehint not needed for last level ops
          stopOps.push_back(opMeta);

        } else {
          llvm_unreachable("Other type of ops should not appear in last level");
        }
      }

      // Since non of last level ops produces any output, we don't need allocate storage.
      // Simply push back
      currentLevelOps.append(regWriteOps.begin(), regWriteOps.end());
      currentLevelOps.append(memWriteOps.begin(), memWriteOps.end());
      currentLevelOps.append(printOps.begin(), printOps.end());
      currentLevelOps.append(stopOps.begin(), stopOps.end());

      partInfo.opPool.push_back(std::move(currentLevelOps));

      CGLayerValueStatistics stats;
      std::memset(&stats, 0, sizeof(CGLayerValueStatistics));
      stats.numRegWrites = regWriteOps.size();
      stats.numMemWrites = memWriteOps.size();
      stats.numPrints = printOps.size();
      stats.numStops = stopOps.size();

      partInfo.opStatisticsPerLevel.push_back(stats);
      partInfo.opStatistics.numRegWrites = stats.numRegWrites;
      partInfo.opStatistics.numMemWrites = stats.numMemWrites;
      partInfo.opStatistics.numPrints = stats.numPrints;
      partInfo.opStatistics.numStops = stats.numStops;
    }

    // TODO: check correctness



    codeGenInfo.partitionInfo.push_back(std::move(partInfo));
  }
}

void SingleRegionScheduler::fillDebugInfo() {
  // Consider parallel

  // Collect io signals and extern module signals
  for (size_t regId = 0; regId < codeGenInfo.regPool.size(); regId++) {
    auto &regMeta = codeGenInfo.regPool[regId];
    if (regMeta.isIO) {
      assert(regMeta.namehint && "An io signal must have a name");
      auto namehint = regMeta.namehint.value();
      codeGenInfo.ioSignals.insert(namehint);
    }
  }


  // collect reg info
  for (size_t regId = 0; regId < codeGenInfo.regPool.size(); regId++) {
    auto &regMeta = codeGenInfo.regPool[regId];
    if (regMeta.namehint) {
      // has name hint
      auto namehint = regMeta.namehint.value();
      // auto fragment_id = regMeta.fragment_id;
      if (!codeGenInfo.regDebugInfo.contains(namehint)) {
        codeGenInfo.regDebugInfo.try_emplace(namehint);
      }
      codeGenInfo.regDebugInfo[namehint].push_back(regId);
    }
  }
  // sort by fragment id
  for (auto &elem: codeGenInfo.regDebugInfo) {
    auto &v = elem.getSecond();
    std::sort(v.begin(), v.end(), [=](const uint32_t a, const uint32_t b) {
      return codeGenInfo.regPool[a].fragment_id > codeGenInfo.regPool[b].fragment_id;
    });
  }



  // collect mem info
  for (size_t memId = 0; memId < codeGenInfo.memPool.size(); memId++) {
    auto &memMeta = codeGenInfo.memPool[memId];
    if (memMeta.namehint) {
      // has name hint
      auto namehint = memMeta.namehint.value();
      if (!codeGenInfo.memDebugInfo.contains(namehint)) {
        codeGenInfo.memDebugInfo.try_emplace(namehint);
      }
      codeGenInfo.memDebugInfo[namehint].push_back(memId);
    }
  }
  // sort by fragment id
  for (auto &elem: codeGenInfo.memDebugInfo) {
    auto &v = elem.getSecond();
    std::sort(v.begin(), v.end(), [=](const uint32_t a, const uint32_t b) {
      return codeGenInfo.memPool[a].fragment_id > codeGenInfo.memPool[b].fragment_id;
    });
  }


  // collect signal info
  for (size_t partId = 0; partId < codeGenInfo.partitionInfo.size(); partId++) {
    auto &partOpPool = codeGenInfo.partitionInfo[partId].opPool;

    for (size_t levelId = 0; levelId < partOpPool.size(); levelId++) {
      auto &currentLevelOps = partOpPool[levelId];
      for (auto &opMeta: currentLevelOps) {
        if (opMeta.hasResult()) {
          if (opMeta.namehint) {
            auto namehint = opMeta.namehint.value();
            if (!codeGenInfo.signalDebugInfo.contains(namehint)) {
              codeGenInfo.signalDebugInfo.try_emplace(namehint);
            }
            auto resultValId = opMeta.getResult();
            codeGenInfo.signalDebugInfo[namehint].push_back({partId, resultValId});
          }
        }
      }
    }
  }

  // sort by fragment_id
  for (auto &elem: codeGenInfo.signalDebugInfo) {
    auto &v = elem.getSecond();
    std::sort(v.begin(), v.end(), [=](const std::tuple<uint32_t, uint32_t>& a, const std::tuple<uint32_t, uint32_t> &b) {
      auto a_partId = std::get<0>(a);
      auto b_partId = std::get<0>(b);

      auto a_valId = std::get<1>(a);
      auto b_valId = std::get<1>(b);

      auto a_fragmentId = codeGenInfo.partitionInfo[a_partId].valuePool[a_valId].fragment_id;
      auto b_fragmentId = codeGenInfo.partitionInfo[b_partId].valuePool[b_valId].fragment_id;
      return a_fragmentId > b_fragmentId;
    });
  }
}

