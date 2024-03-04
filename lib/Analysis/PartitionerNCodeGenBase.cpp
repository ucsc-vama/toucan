#include "circt/Dialect/HW/HWTypes.h"
#include "circt/Support/LLVM.h"
#include "circt/Dialect/Comb/CombDialect.h"
#include "circt/Dialect/Comb/CombOps.h"
#include "circt/Dialect/Seq/SeqOps.h"

#include "mlir/IR/Value.h"
#include "mlir/Pass/AnalysisManager.h"
#include "mlir/Support/LLVM.h"
#include "toucan/ToucanAnalysis.h"
#include "toucan/ToucanOps.h"
#include "toucan/ToucanTypes.h"
#include "toucan/ToucanUtils.h"

#include "llvm/Support/Casting.h"
#include "llvm/Support/Debug.h"
#include <cstddef>
#include <cstdint>

#include <boost/graph/topological_sort.hpp>
#include <iterator>

using namespace toucan;

using namespace mlir;
using namespace llvm;
using namespace circt;

void PartitionerNCodeGenBase::levelizePartitions(DesignGraph &graph) {
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

  // init
  for (size_t partId = 0; partId < partitions.size(); ++partId) {
    partLevels.push_back({});
    partLevels[partId].push_back({});
  }

  // save level in partition
  for (uint32_t vtx = 0; vtx < boost::num_vertices(graph.g); ++vtx) {
    auto vtxLevel = levels[vtx];
    if (vtxLevel == UINT32_MAX) continue;

    auto vtxPartition = vtxIdToPartId[vtx];
    // grow container if needed
    for (size_t levelId = partLevels[vtxPartition].size(); levelId <= vtxLevel; levelId++) {
      partLevels[vtxPartition].push_back({});
    }
    partLevels[vtxPartition][vtxLevel].push_back(vtx);
  }

  // Condense, remove empty levels
  partLevels.erase(std::remove_if(partLevels.begin(), partLevels.end(), [](auto innerVec) {
    return innerVec.empty();
  }), partLevels.end());

  // push all sink vtx to last level
  for (size_t partId = 0; partId < partitions.size(); ++partId) {
    // An empty level to hold all sink vtx
    partLevels[partId].push_back({});
  }
  for (auto sinkVtx: sinkVtxes) {
    auto partId = vtxIdToPartId[sinkVtx];
    auto numParts = partLevels.size();
    assert(partId < numParts);
    partLevels[partId].end()->push_back(sinkVtx);
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
      // }
    }
  }
  return;
}

// void PartitionerNCodeGenBase::collectPrintMsgs(DesignGraph &graph) {
//   mlir::DenseMap<mlir::StringRef, uint32_t> stringToId;

//   for (uint32_t vtxId = 0; vtxId < boost::num_vertices(graph.g); vtxId++) {
//     auto vtxOpName = graph.g[vtxId].toucanOpName;
//     if (vtxOpName == CGToucanOPName::Print) {
//       auto printOp = cast<toucan::PrintOp>(graph.g[vtxId].op);
//       auto printMsgRef = printOp.getMsg();
//     }
//   }
// }

void PartitionerNCodeGenBase::generateRegMemLayout(DesignGraph &graph, uint32_t partitionRegPaddingSpace, uint32_t memPaddingSpace) {
  // collect all reg and memory, generate layout
  for (size_t partId = 0; partId < partitions.size(); partId++) {
    auto &lastLevel = *partLevels[partId].end();
    for (auto vtxId: lastLevel) {
      // for each vtx in current partition
      auto vtxOpName = graph.g[vtxId].toucanOpName;
      if (vtxOpName == CGToucanOPName::RegWrite) {
        auto regWriteOp = cast<toucan::RegWriteOp>(graph.g[vtxId].op);
        auto regVal = regWriteOp.getReg();
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

        auto regId = codeGenInfo.regPool.size();
        codeGenInfo.regPool.push_back(regMeta);
        codeGenInfo.toucanRegToId[regVal] = regId;
        // TODO: fill reg debug info
      }
    }
    // Add extra padding
    for (size_t i = 0; i < partitionRegPaddingSpace; i++) {
      CGRegMetaInfo paddingRegMeta;

      paddingRegMeta.isPadding = true;
      paddingRegMeta.bitWidth = 0;
      paddingRegMeta.fragment_id = 0;

      // auto regId = codeGenInfo.regPool.size();
      codeGenInfo.regPool.push_back(paddingRegMeta);
    }
  }

  // collect all reg and memory, generate layout
  uint64_t memBaseAddr = 0;
  for (size_t vtxId = 0; vtxId < boost::num_vertices(graph.g); vtxId++) {
    // TODO: if this is too slow (unlikely), consider find all DefMem nodes
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
      // TODO: fill mem debug info
    }
  }
  codeGenInfo.totalMemSize = memBaseAddr;
}

void PartitionerNCodeGenBase::collectConstant(DesignGraph &graph, CGPartitionMetaInfo &partInfo, uint32_t partId) {
  // Collect all consts, populate value pool
  for (uint32_t vtxId = 0; vtxId < boost::num_vertices(graph.g); vtxId++) {
    if (vtxIdToPartId[vtxId] == partId) {
      // for each vtx in current partition
      auto vtxOpName = graph.g[vtxId].toucanOpName;
      if (vtxOpName == CGToucanOPName::ConstDecl) {
        auto op = graph.g[vtxId].op;
        partInfo.opToResultValueId[op] = partInfo.valuePool.size();

        if (auto constOp = dyn_cast<toucan::ConstantOp>(op)) {
          // regular const
          auto constVal = constOp.getValue();
          assert(constVal.getBitWidth() <= 4);
          auto rawVal = static_cast<uint8_t>(constVal.getZExtValue());
          partInfo.valuePool.push_back({true, rawVal, op});
        } else {
          // it must be a vector const
          auto defConstVecOp = cast<toucan::DefConstVectorOp>(op);
          for (auto &vecValElem: defConstVecOp.getValues().getValue()) {
            auto elemVal = cast<mlir::IntegerAttr>(vecValElem).getValue();
            assert(elemVal.getBitWidth() <= 4);
            auto rawVal = static_cast<uint8_t>(elemVal.getZExtValue());
            partInfo.valuePool.push_back({true, rawVal, op});
          }
        }
      }
    }
  }
}

void PartitionerNCodeGenBase::generateMemoryLayout(DesignGraph &graph, uint32_t partitionRegPaddingSpace, uint32_t memPaddingSpace) {

  // Collect information for code gen
  // This function also determines register layout
  generateRegMemLayout(graph, partitionRegPaddingSpace, memPaddingSpace);



  for (size_t partId = 0; partId < partitions.size(); partId++) {
    CGPartitionMetaInfo partInfo;

    // A const zero for all luts
    CGValueMetaInfo zeroConst = {true, 0, nullptr};
    partInfo.valuePool.push_back(zeroConst);

    // Collect all constants
    collectConstant(graph, partInfo, partId);
    partInfo.numConstsInValuePool = partInfo.valuePool.size() + 1;

    // Note: first level should contains regread and memread
    // middle levels should contain lut or vecread
    // last level should contains print, stop, regwrite, memwrite
    auto &firstLevel = partLevels[partId][0];
    auto &lastLevel = *partLevels[partId].end();

    // First level. should only contains regread or const decl, the later will be pushed into const pool
    mlir::SmallVector<CGOpMetaInfo> firstLevelOps;
    for (auto vtxId: firstLevel) {
      auto tOpName = graph.g[vtxId].toucanOpName;
      assert(tOpName == CGToucanOPName::RegRead || tOpName == CGToucanOPName::MemRead);

      auto op = graph.g[vtxId].op;
      CGOpMetaInfo opMeta;
      opMeta.opName = tOpName;

      if (tOpName == CGToucanOPName::RegRead) {
        // a regread
        auto regReadOp = cast<toucan::RegReadOp>(op);
        auto regVal = regReadOp.getReg();
        auto regValId = codeGenInfo.toucanRegToId[regVal];

        opMeta.regRead.reg = regValId;
        firstLevelOps.push_back(opMeta);
      } else if (tOpName == CGToucanOPName::ConstDecl) {
        // A constant decl. Do nothing.
      } else {
        assert(false && "Should not reach here");
      }
    }
    // TODO: Allocate result storage

    // Codegen for middle layers
    // At least 3 layers: 1 for input, 1 for lut, the last 1 for output
    assert(partLevels[partId].size() > 2);
    for (size_t layerId = 1; layerId < partLevels[partId].size() - 1; layerId++) {
      auto &currentLevel = partLevels[partId][layerId];

      for (auto vtxId: currentLevel) {
        auto tOpName = graph.g[vtxId].toucanOpName;
        auto op = graph.g[vtxId].op;
        auto vtxWeight = graph.g[vtxId].weight;
        // lut 
        // vec, expand

        // mem read
        if (tOpName == CGToucanOPName::MemRead) {
          CGOpMetaInfo opMeta;
          // a memread
          auto memReadOp = cast<toucan::MemReadOp>(op);
          auto memVal = memReadOp.getMem();
          auto memValId = codeGenInfo.toucanMemToId[memVal];

          opMeta.memRead.hasMultipleWriter = codeGenInfo.memPool[memValId].hasMultipleWriter;
          opMeta.memRead.memBase = codeGenInfo.memPool[memValId].memBase;

          // TODO: collect addr
        }
      }
    }

    // TODO: last level
    for (auto vtxId: lastLevel) {
      //
    }



    codeGenInfo.partitionInfo.push_back(std::move(partInfo));
  }
}
