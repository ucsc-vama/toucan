
#include "circt/Dialect/SV/SVDialect.h"
#include "circt/Dialect/OM/OMDialect.h"
#include "circt/Dialect/Seq/SeqDialect.h"
#include "circt/Support/LLVM.h"

#include "mlir/IR/AsmState.h"
#include "mlir/IR/BuiltinOps.h"
#include "mlir/IR/BuiltinTypes.h"
#include "mlir/IR/OpImplementation.h"
#include "mlir/IR/Operation.h"
#include "mlir/IR/Visitors.h"
#include "mlir/Interfaces/SideEffectInterfaces.h"
#include "mlir/Support/LLVM.h"
#include "mlir/IR/Threading.h"
#include "mlir/Support/LogicalResult.h"
#include "toucan/MicroPartitioner.h"
#include "toucan/MultiRegionMicroPartScheduler.h"
#include "toucan/PartitioningGraph.h"
#include "toucan/RepCutPartitioner.h"
#include "toucan/ToucanAnalysis.h"
#include "toucan/ToucanCodeGenInfo.h"

#include "llvm/ADT/DenseMap.h"
#include "llvm/ADT/SmallPtrSet.h"
#include "llvm/ADT/SmallVector.h"
#include "llvm/ADT/STLExtras.h"

#include "llvm/ADT/StringExtras.h"
#include "llvm/Support/Casting.h"
#include "llvm/Support/Debug.h"
#include "llvm/Support/ErrorHandling.h"
#include "llvm/Support/raw_ostream.h"

#include <cstddef>
#include <cstdint>
#include <algorithm>
#include <cstdlib>
#include <iterator>
#include <memory>
#include <filesystem>
#include <fstream>
#include <sstream>
#include <tuple>
#include <vector>


#define GEN_PASS_DEF_GPUCODEGEN
#include "toucan/ToucanPassCommon.h"

#include "toucan/MicroPartitioner.h"
#include "toucan/CodeGenCommon.h"
#include "ToucanGPUSim/ToucanGPUGenDataTypes.h"

#include "toucan/PartitioningManager.h"

using namespace toucan;
using namespace circt;
using namespace mlir;
using namespace llvm;

#define DEBUG_TYPE "GPUCodeGenPass"


struct GPUCodeGenPass : toucan::impl::GPUCodeGenBase<GPUCodeGenPass>, CodeGenHelper {
  using GPUCodeGenBase<GPUCodeGenPass>::GPUCodeGenBase;

  toucanGPUSim::SimDesignInfo designInfo;
  toucanGPUSim::SimDebugInfo debugInfo;


  void populateMicroPartInfo(const CGMicroPartInfo &mp, toucanGPUSim::CGMicroPartInfo &cmp) {


    switch (mp.opType) {
      case CGToucanOPName::LUT: {
        cmp.isLUTPart = true;
        for (const auto &eachTopOp: mp.topLevel) {
          toucanGPUSim::CGMicroPartLUTTopLevelOp op;
          op.lutIndex = lutPos.at(static_cast<uint32_t>(eachTopOp.opName));
          op.op0 = eachTopOp.op0;
          op.op1 = eachTopOp.op1;
          op.op2 = eachTopOp.op2;

          cmp.topLevel.push_back(op);
        }
        for (const auto &eachMiddleLevel: mp.middleLevels) {
          cmp.middleLevels.emplace_back();
          for (const auto &eachMiddleOp: eachMiddleLevel) {
            // ensure values does not exceed expectation
            assert(lutPos.at(static_cast<uint32_t>(eachMiddleOp.opName)) < (1 << 14));
            assert(eachMiddleOp.op0 < (1 << 6));
            assert(eachMiddleOp.op1 < (1 << 6));
            assert(eachMiddleOp.op2 < (1 << 6));

            toucanGPUSim::CGMicroPartLUTMiddleLevelOp op(
              lutPos.at(static_cast<uint32_t>(eachMiddleOp.opName)),
              eachMiddleOp.op0,
              eachMiddleOp.op1,
              eachMiddleOp.op2
            );

            cmp.middleLevels.back().push_back(op);
          }
        }
        for (const auto &eachLastOp: mp.lastLevel) {
          toucanGPUSim::CGMicroPartLUTLastLevelWriteBack op;
          op.shuffleId = eachLastOp.shuffleId;
          op.result = eachLastOp.result;

          cmp.lastLevel.push_back(op);
        }
        break;
      }
      case CGToucanOPName::VecRead: {
        cmp.isLUTPart = false;
        assert(mp.vecRead.size() > 0);

        for (const auto &eachOp: mp.vecRead) {
          toucanGPUSim::CGMicroPartVecRead op;

          op.vecBase = eachOp.vecBase;
          op.vecLength = eachOp.vecLength;
          op.offset = eachOp.offset;
          op.isConstVec = eachOp.isConstVec;

          op.index0 = eachOp.index0;
          op.index1 = eachOp.index1;
          op.index2 = eachOp.index2;
          op.index3 = eachOp.index3;
          op.outRangeValue = eachOp.outRangeValue;
          op.result = eachOp.result;

          cmp.vecRead.push_back(op);
        }
        break;
      }
      case CGToucanOPName::MemRead: {
        cmp.isLUTPart = false;
        assert(mp.memRead.size() > 0);

        for (const auto &eachOp: mp.memRead) {
          toucanGPUSim::CGMicroPartMemRead op;

          op.hasMultipleWriter = eachOp.hasMultipleWriter;
          op.memBase = eachOp.memBase;
          op.en = eachOp.en;
          op.addrVec = eachOp.addrVec;
          op.result = eachOp.result;

          cmp.memRead.push_back(op);
        }
        break;
      }
      case CGToucanOPName::VecLogic: 
      case CGToucanOPName::VecArith: {
        llvm_unreachable("VecLogic and VecArith is not expected to appear here!");
      }
      default: {
        llvm_unreachable("Should not reach here");
      }
    }
  }

  void populateVecArithAndVecLogicMicroParts(mlir::SmallVector<toucanGPUSim::CGMicroPartInfo> &newMps, const mlir::SmallVector<CGMicroPartVecArith> &allVecArithOps, const mlir::SmallVector<CGMicroPartVecLogic> &allVecLogicOps) {
    mlir::SmallVector<toucanGPUSim::CGMicroPartVecArithOrLogic> allOps;

    for (const auto &eachVecArithOp: allVecArithOps) {
      toucanGPUSim::CGMicroPartVecArithOrLogic op;
      op.isV1V2Const = 0;
      if (eachVecArithOp.isVec1Const) op.isV1V2Const |= 0b10;
      if (eachVecArithOp.isVec2Const) op.isV1V2Const |= 1;
      op.vec1Base = eachVecArithOp.vec1Base;
      op.vec2Base = eachVecArithOp.vec2Base;
      // For now, vecLength is limited by TOUCAN_VEC_OP_MAX_WIDTH
      assert(eachVecArithOp.vecLength < UINT16_MAX);
      op.vecLength = eachVecArithOp.vecLength;
      op.result = eachVecArithOp.result;

      switch (eachVecArithOp.opName) {
        case VecArithOpName::VecArith_Add: {
          op.opName = VEC_ARITH_ADD;
          break;
        }
        case VecArithOpName::VecArith_Sub:{
          op.opName = VEC_ARITH_SUB;
          break;
        }
        case VecArithOpName::VecArith_Mul:{
          op.opName = VEC_ARITH_MUL;
          break;
        }
        // default: llvm_unreachable("Whats this");
      }

      allOps.push_back(op);
    }

    for (const auto &eachVecLogicOp: allVecLogicOps) {
      toucanGPUSim::CGMicroPartVecArithOrLogic op;
      op.isV1V2Const = 0;
      if (eachVecLogicOp.isVec1Const) op.isV1V2Const |= 0b10;
      if (eachVecLogicOp.isVec2Const) op.isV1V2Const |= 1;
      op.vec1Base = eachVecLogicOp.vec1Base;
      op.vec2Base = eachVecLogicOp.vec2Base;
      // For now, vecLength is limited by TOUCAN_VEC_OP_MAX_WIDTH
      assert(eachVecLogicOp.vecLength < UINT16_MAX);
      op.vecLength = eachVecLogicOp.vecLength;
      op.result = eachVecLogicOp.result;

      switch (eachVecLogicOp.opName) {
        case VecLogicOpName::VecLogic_Eq: {
          op.opName = VEC_LOGIC_EQ;
          break;
        }
        case VecLogicOpName::VecLogic_Lt: {
          op.opName = VEC_LOGIC_LT;
          break;
        }
        case VecLogicOpName::VecLogic_Le: {
          op.opName = VEC_LOGIC_LE;
          break;
        }
        // default: llvm_unreachable("Whats this");
      }

      allOps.push_back(op);
    }

    assert(allOps.size() > 0);
    assert(newMps.size() == 0);
    size_t pos = 0;
    while (pos < allOps.size()) {
      auto remainingOps = allOps.size() - pos;
      auto newPartSize = std::min(remainingOps, static_cast<size_t>(32));

      newMps.emplace_back();
      newMps.back().isLUTPart = false;
      newMps.back().vecArithAndLogic.assign(allOps.begin() + pos, allOps.begin() + (pos + newPartSize));
      assert(newMps.back().vecArithAndLogic.size() == newPartSize);

      pos += newPartSize;
    }
    assert(pos == allOps.size());

  }


  void populateSinglePartition(const CGPartitionMetaInfo &part, uint32_t partId) {
    toucanGPUSim::SimPartitionInfo partInfo;
    // at lease 1 level
    assert(part.microPartOps.size() >= 1);
    partInfo.valuePool.resize(part.numConstsInValuePool);
    for (size_t i = 0; i < part.numConstsInValuePool; i++) {
      partInfo.valuePool[i] = part.constValuePool[i];
    }
    assert(part.constValuePool.size() == part.numConstsInValuePool);
    partInfo.valuePoolSize = part.numTotalValues;

    if (partInfo.valuePoolSize > UINT16_MAX) {
      llvm::errs() << "Value pool size is " << partInfo.valuePoolSize << ", which exceeds UINT16_MAX. This should not happen.\n";
    }
    assert(partInfo.valuePoolSize <= UINT16_MAX && "Value pool is too large");

    // copy const vec data
    std::copy(part.constVecPool.begin(), part.constVecPool.end(), std::back_inserter(partInfo.constVecPool));

    assert(part.exchangeReadOps.empty() || part.regReadOps.empty());

    // reg reads
    partInfo.ops_l0_regRead.reserve(part.regReadOps.size());
    for (const auto &opMeta: part.regReadOps) {
      toucanGPUSim::CGRegReadMetaInfo rr;
      rr.reg = opMeta.regRead.reg;
      rr.result = opMeta.regRead.result;
      partInfo.ops_l0_regRead.push_back(rr);
    }

    // ensure reg reads mem locations in ascending order
    if (partInfo.ops_l0_regRead.size() > 1) {
      auto rr_reg_last = partInfo.ops_l0_regRead.front().reg;
      auto rr_result_last = partInfo.ops_l0_regRead.front().result;

      for (size_t i = 1; i < partInfo.ops_l0_regRead.size(); i++) {
        auto rr_reg = partInfo.ops_l0_regRead[i].reg;
        auto rr_result = partInfo.ops_l0_regRead[i].result;
        assert(rr_reg > rr_reg_last);
        assert(rr_result > rr_result_last);
        rr_reg_last = rr_reg;
        rr_result_last = rr_result;
      }
    }


    // exchange reads
    partInfo.ops_l0_exchangeRead.reserve(part.exchangeReadOps.size());
    for (const auto &opMeta: part.exchangeReadOps) {
      toucanGPUSim::CGExchangeReadMetaInfo er;
      er.exchange = opMeta.exgRead.exchangeVal;
      er.result = opMeta.exgRead.localVal;
      partInfo.ops_l0_exchangeRead.push_back(er);
    }

    // ensure reg reads mem locations in ascending order
    if (partInfo.ops_l0_exchangeRead.size() > 1) {
      auto er_exg_last = partInfo.ops_l0_exchangeRead.front().exchange;
      auto er_local_last = partInfo.ops_l0_exchangeRead.front().result;

      for (size_t i = 1; i < partInfo.ops_l0_exchangeRead.size(); i++) {
        auto er_exg = partInfo.ops_l0_exchangeRead[i].exchange;
        auto er_local = partInfo.ops_l0_exchangeRead[i].result;
        assert(er_exg > er_exg_last);
        assert(er_local > er_local_last);
        er_exg_last = er_exg;
        er_local_last = er_local;
      }
    }

    // Micro parts
    for (const auto &eachMPLevel: part.microPartOps) {
      partInfo.exec_mParts.emplace_back();

      mlir::SmallVector<CGMicroPartVecArith> allVecArithOps;
      mlir::SmallVector<CGMicroPartVecLogic> allVecLogicOps;

      for (const auto &eachMP: eachMPLevel) {
        if (eachMP.opType != CGToucanOPName::VecArith && eachMP.opType != CGToucanOPName::VecLogic) {
          partInfo.exec_mParts.back().emplace_back();
          populateMicroPartInfo(eachMP, partInfo.exec_mParts.back().back());
        } else {
          // Note: VecArith and VecLogic have same byte code format
          if (eachMP.opType == CGToucanOPName::VecArith) {
            assert(eachMP.vecArith.size() > 0);
            allVecArithOps.append(eachMP.vecArith);
          } else {
            assert(eachMP.opType == CGToucanOPName::VecLogic);
            assert(eachMP.vecLogic.size() > 0);
            allVecLogicOps.append(eachMP.vecLogic);
          }
        }
      }

      if (allVecArithOps.size() + allVecLogicOps.size() > 0) {
        mlir::SmallVector<toucanGPUSim::CGMicroPartInfo> newMps;

        populateVecArithAndVecLogicMicroParts(newMps, allVecArithOps, allVecLogicOps);
        assert(newMps.size() != 0);
        assert(newMps.front().vecArithAndLogic.size() != 0);

        partInfo.exec_mParts.back().insert(partInfo.exec_mParts.back().end(), newMps.begin(), newMps.end());
      }

      // sort for performance
      assert(partInfo.exec_mParts.size() < UINT16_MAX);
      auto getPartSize = [](const toucanGPUSim::CGMicroPartInfo &p) {
        if (p.isLUTPart) {
          int ret = 0;
          ret += p.topLevel.size();
          for (const auto &eachLevel: p.middleLevels) {
            ret += eachLevel.size();
          }
          ret += p.lastLevel.size();

          return ret;
        }

        int numOps = 0;
        if (p.memRead.size() != 0) numOps = p.memRead.size();
        else if (p.vecRead.size() != 0) numOps = p.vecRead.size();
        else numOps = p.vecArithAndLogic.size();
        assert(numOps != 0);
        return numOps;
      };
      std::sort(partInfo.exec_mParts.back().begin(), partInfo.exec_mParts.back().end(), [getPartSize](const auto &a, const auto &b) {
        auto weight_a = getPartSize(a);
        auto weight_b = getPartSize(b);
        return weight_a > weight_b;
      });
      auto getPartWeight = [](const toucanGPUSim::CGMicroPartInfo &p) {
        if (p.isLUTPart) return p.middleLevels.size();
        size_t numOps = 0;
        if (p.memRead.size() != 0) numOps = p.memRead.size();
        else if (p.vecRead.size() != 0) numOps = p.vecRead.size();
        else numOps = p.vecArithAndLogic.size();
        assert(numOps != 0);
        return UINT16_MAX + numOps;
      };
      std::stable_sort(partInfo.exec_mParts.back().begin(), partInfo.exec_mParts.back().end(), [getPartWeight](const auto &a, const auto &b) {
        auto weight_a = getPartWeight(a);
        auto weight_b = getPartWeight(b);
        return weight_a > weight_b;
      });

    }


    // mem writes
    for (const auto &opMeta: part.memWriteOps) {
      assert(opMeta.memWrite.addrVec <= UINT16_MAX);
      assert(opMeta.memWrite.dat <= UINT16_MAX);
      assert(opMeta.memWrite.en <= UINT16_MAX);

      toucanGPUSim::CGMemWriteMetaInfo info;
      info.hasMultipleWriter = opMeta.memWrite.hasMultipleWriter;
      // info.memDepth = opMeta.memWrite.memDepth;
      info.memBase = opMeta.memWrite.memBase;
      info.addrVec = opMeta.memWrite.addrVec;
      info.dat = opMeta.memWrite.dat;
      info.en = opMeta.memWrite.en;
      
      partInfo.ops_last_memWrite.push_back(info);
    }

    // print
    for (const auto &opMeta: part.printOps) {
      assert(opMeta.print.en <= UINT16_MAX);
      assert(opMeta.print.msg <= UINT16_MAX);

      toucanGPUSim::CGPrintMetaInfo info;
      info.en = opMeta.print.en;
      info.msg = opMeta.print.msg;

      partInfo.ops_last_print.push_back(info);
    }

    // stop
    for (const auto &opMeta: part.stopOps) {
      assert(opMeta.stop.en <= UINT16_MAX);

      toucanGPUSim::CGStopMetaInfo info;
      info.en = opMeta.stop.en;

      partInfo.ops_last_stop.push_back(info);
    }

    // reg write
    uint32_t rw_bulk_size = 0;
    uint32_t rw_bulk_start_dat = 0;
    uint32_t rw_bulk_start_reg = 0;
    for (const auto &opMeta: part.regWriteOps) {
      assert(opMeta.regWrite.dat <= UINT16_MAX);

      if (rw_bulk_size == 0) {
        rw_bulk_start_dat = opMeta.regWrite.dat;
        rw_bulk_start_reg = opMeta.regWrite.reg;
        rw_bulk_size = 1;
      } else {
        if (opMeta.regWrite.dat == rw_bulk_start_dat + rw_bulk_size && opMeta.regWrite.reg == rw_bulk_start_reg + rw_bulk_size) {
          // bulk
          rw_bulk_size += 1;
        } else {
          // a new bulk. This should not happen
          assert(false && "Each partition should only write to a contiguous range of registers");
        }
      }
    }
    // add extra padding
    if (rw_bulk_size != 0) {
      rw_bulk_size += getExtraAlignmentSpace(rw_bulk_size, 16);
    }
    // special handling for regWrites

    if (rw_bulk_size != 0) {
      llvm::outs() << "Partition " << partId << " reg writes: from data(shared mem) " << rw_bulk_start_dat << " to reg(global mem) " << rw_bulk_start_reg << ", size " << rw_bulk_size << "B\n";
    }


    toucanGPUSim::CGRegWriteMetaInfo rwInfo;
    if (rw_bulk_size != 0) {
      // Note: This is guarenteed by LocalValueAllocator
      assert(rw_bulk_start_dat == 16 && "Local data should starts from shared memory addr 16");
      assert(rw_bulk_start_reg % 16 == 0 && "Register should be aligned to 16B");
      assert(rw_bulk_size < UINT16_MAX && "Register write count should fit in uint16");

      rwInfo.reg = rw_bulk_start_reg;
      rwInfo.dat = rw_bulk_start_dat;
      rwInfo.count = rw_bulk_size;
    } else {
      rwInfo.reg = 0;
      rwInfo.dat = 0;
      rwInfo.count = 0;
    }

    partInfo.op_last_regWrite = rwInfo;


    // exchange write
    uint32_t ew_bulk_size = 0;
    uint32_t ew_bulk_start_dat = 0;
    uint32_t ew_bulk_start_exg = 0;
    for (const auto &opMeta: part.exchangeWriteOps) {
      assert(opMeta.exgWrite.localVal <= UINT16_MAX);

      if (ew_bulk_size == 0) {
        ew_bulk_start_dat = opMeta.exgWrite.localVal;
        ew_bulk_start_exg = opMeta.exgWrite.exchangeVal;
        ew_bulk_size = 1;
      } else {
        if (opMeta.exgWrite.localVal == ew_bulk_start_dat + ew_bulk_size && opMeta.exgWrite.exchangeVal == ew_bulk_start_exg + ew_bulk_size) {
          // bulk
          ew_bulk_size += 1;
        } else {
          // a new bulk. This should not happen
          assert(false && "Each partition should only write to a contiguous range of exchange values");
        }
      }
    }
    // add extra padding
    if (ew_bulk_size != 0) {
      ew_bulk_size += getExtraAlignmentSpace(ew_bulk_size, 16);
    }

    if (ew_bulk_size != 0) {
      llvm::outs() << "Partition " << partId << " exchange writes: from data(shared mem) " << ew_bulk_start_dat << " to exchange pool (global mem) " << ew_bulk_start_exg << ", size " << ew_bulk_size << "B\n";
    }

    toucanGPUSim::CGExchangeWriteMetaInfo ewInfo;
    if (ew_bulk_size != 0) {
      // Note: This is guarenteed by LocalValueAllocator
      assert(ew_bulk_start_dat == 16 && "Local data should starts from shared memory addr 16");
      assert(ew_bulk_start_exg % 16 == 0 && "Exchange should be aligned to 16B");
      assert(ew_bulk_size < UINT16_MAX && "Exchange write count should fit in uint16");

      ewInfo.exchange = ew_bulk_start_exg;
      ewInfo.dat = ew_bulk_start_dat;
      ewInfo.count = ew_bulk_size;
    } else {
      ewInfo.exchange = 0;
      ewInfo.dat = 0;
      ewInfo.count = 0;
    }
    partInfo.op_last_exchangeWrite = ewInfo;

    designInfo.parts.push_back(std::move(partInfo));
  }

  void populateDebugInfo(const CGInfo& codeGenInfo) {
    std::vector<std::tuple<uint32_t, uint32_t>> eachRegDbgInfo;
    std::vector<std::tuple<uint32_t, uint32_t, uint32_t>> eachMemDbgInfo;
    std::vector<std::tuple<uint32_t, uint32_t, uint32_t>> eachSignalDbgInfo;

    for (auto [regNameRef, ids]: codeGenInfo.regDebugInfo) {
      eachRegDbgInfo.clear();
      for (auto regId: ids) {
        eachRegDbgInfo.push_back(std::make_tuple(regId, codeGenInfo.regPool[regId].bitWidth));
      }
      debugInfo.regDebugInfo[regNameRef.str()] = eachRegDbgInfo;
    }
    for (auto [memNameRef, ids]: codeGenInfo.memDebugInfo) {
      eachMemDbgInfo.clear();
      for (auto memId: ids) {
        auto startPos = codeGenInfo.memPool[memId].memBase;
        auto bitWidth = codeGenInfo.memPool[memId].bitWidth;
        auto memDepth = codeGenInfo.memPool[memId].memDepth;
        eachMemDbgInfo.push_back(std::make_tuple(startPos, bitWidth, memDepth));
      }
      debugInfo.memDebugInfo[memNameRef.str()] = eachMemDbgInfo;
    }
    for (auto [sigNameRef, sigLocs]: codeGenInfo.signalDebugInfo) {
      eachSignalDbgInfo.clear();
      assert(sigLocs.size() != 0);

      bool signalIsComplete = true;

      for (auto sigLoc: sigLocs) {
        auto partId = std::get<0>(sigLoc);
        auto sigId = std::get<1>(sigLoc);
        auto sigBitWidth = std::get<2>(sigLoc);
        eachSignalDbgInfo.push_back(std::make_tuple(partId, sigId, sigBitWidth));

        if (partId == UINT32_MAX) {
          signalIsComplete = false;
        }
      }

      if (signalIsComplete) {
        debugInfo.signalDebugInfo[sigNameRef.str()] = eachSignalDbgInfo;
      }
    }
  }

  void saveDebugGraph(std::filesystem::path graphDirectory, const PartitioningGraph &rawGraph, const PartitioningGraph &mpGraph) {
    const char *rawGraphName = "rawGraph.graph";
    const char* mpGraphName = "mpGraph.graph";

    auto rawGraphSucc = RepCutPartitioner::dumpGraphToFile(rawGraph, graphDirectory / rawGraphName);
    assert(mlir::succeeded(rawGraphSucc));

    mlir::SmallVector<uint32_t> allNodes;
    for (auto v: boost::make_iterator_range(boost::vertices(mpGraph))) {
      allNodes.push_back(v);
    }
    auto mpGraphSucc = PartitioningManager::dumpGraphToFileForMicroPartitioner(mpGraph, allNodes, graphDirectory / mpGraphName);

    assert(mlir::succeeded(mpGraphSucc));
  }


  void runOnOperation() final {
    // Mark all analyses as preserved. This is a read only pass
    markAllAnalysesPreserved();

    auto graph = getAnalysis<DesignGraph>();
    PartitioningManager pm;
    pm.context = &getContext();

    auto init_succeed = pm.init(outputDirectory.getValue(), 2);
    if (failed(init_succeed)) {
      signalPassFailure();
      return;
    }

    
    {
      // Note: In this step, small parts at same level are not merged. This is done in scheduler
      auto ret = pm.runStage1MicroPartitioner(graph.g);
      if (failed(ret)) {
        signalPassFailure();
        return;
      }

      ret = pm.buildMicroPartGraph(graph.g);
      if (failed(ret)) {
        signalPassFailure();
        return;
      }

      // saveDebugGraph(outputDirectory.getValue(), graph.g, *pm.microPartGraph);

      auto cutPoint = pm.findCutPoint();

      pm.cutGraph(cutPoint);
      pm.breakDirectIOConnection(graph.g);

      pm.updateGraphWeight_r0();
      pm.updateGraphWeight_r1();

      assert(ibFactor > 0.0f);
      ret = pm.runStage2RepCutPartitioner(partSizeRatio, ibFactor);
      if (failed(ret)) {
        signalPassFailure();
        return;
      }

      // create codegen data for scheduler. Also merges special MParts.
      pm.collectRepCutPartitionCodeGenData();

      // verify exchange index. can be removed
      assert(pm.exchangeValPool.size() == pm.exchangeValues.size());
      for (const auto &[val, idx]: pm.exchangeValues) {
        assert(idx < pm.exchangeValPool.size());
        assert(pm.exchangeValPool[idx] == val);
      }
    }


    // Schedule
    llvm::outs() << "================== Schedule operations ==================\n";

    auto scheduler = MultiRegionMicroPartScheduler();
    // 2 regions for now
    scheduler.regionPartData.resize(2);
    std::swap(scheduler.regionPartData[0], pm.partCodeGenData_r0);
    std::swap(scheduler.regionPartData[1], pm.partCodeGenData_r1);
    scheduler.buildDummyVtxIndexInVec(*(pm.mp));
    scheduler.copyVecTables(*(pm.mp));

    // After this point, pm is not used and can be released.
    scheduler.schedule(&getContext(), graph.g, pm.exchangeValPool);

    // To use this, run with -debug-only=GPUCodeGenPass
    LLVM_DEBUG(
      llvm::dbgs() << "============ Partitioning Details =============\n";
      for (size_t partId = 0; partId < scheduler.codeGenInfo.partitionInfo.size(); partId++) {
        llvm::dbgs() << "Part " << partId << "\n";
        scheduler.printPartInfo(scheduler.codeGenInfo.partitionInfo[partId]);
      }
    );


    llvm::outs() << "======================= Code Gen =======================\n";


    // Fill lut
    populateLUT();
    designInfo.lut.assign(lutContent.begin(), lutContent.end());
    llvm::outs() << "LUT size " << designInfo.lut.size() << "B\n";

    // copy pool size
    designInfo.regPoolSize = scheduler.codeGenInfo.regPool.size();
    designInfo.memPoolSize = scheduler.codeGenInfo.totalMemSize;
    designInfo.exchangePoolSize = scheduler.codeGenInfo.exchangePool.size();

    assert(designInfo.regPoolSize != 0);

    designInfo.regionPartitionIds.clear();
    uint32_t partId = 0;
    for (const auto &eachRegionParts: scheduler.codeGenInfo.regionPartitionIds) {
      designInfo.regionPartitionIds.emplace_back();
      for (const auto &eachPartId: eachRegionParts) {
        assert(eachPartId == partId);

        // save partition region info
        designInfo.regionPartitionIds.back().push_back(partId);
        // codegen for a single partition
        populateSinglePartition(scheduler.codeGenInfo.partitionInfo[partId], partId);
        partId++;
      }
    }
    assert(designInfo.regionPartitionIds.size() == 2);

    // Fill print msgs
    designInfo.printMsgs.resize(scheduler.codeGenInfo.printStrings.size());
    for (auto [k, v]: scheduler.codeGenInfo.printStrings) {
      assert(designInfo.printMsgs[v].empty());
      designInfo.printMsgs[v] = k;
    }
    llvm::outs() << "Design has " << designInfo.printMsgs.size() << " unique print messages\n";

    // // Fill debug info
    populateDebugInfo(scheduler.codeGenInfo);
    assert(scheduler.codeGenInfo.ioSignals.size() != 0);

    llvm::outs() << "Symbol file has " 
      << debugInfo.regDebugInfo.size() << " register debug info, " 
      << debugInfo.memDebugInfo.size() << " memory debug info, " 
      << debugInfo.signalDebugInfo.size() << " signal debug info\n";


    llvm::outs() << "=================== Serialize Netlist ===================\n";
    // Save. serialize
    auto outputDesignFileFullName = std::filesystem::path(outputDirectory.getValue()) / outputDesignFilename.getValue();
    std::ofstream ofs_design(outputDesignFileFullName, std::ios::binary | std::ios::out);
    toucanGPUSim::serializeSimDesignInfo(ofs_design, designInfo);
    ofs_design.close();

    // debug: deserialize it, ensure everything is same
    std::ifstream ifs_design(outputDesignFileFullName, std::ios::binary);
    toucanGPUSim::SimDesignInfo design_read_back;
    toucanGPUSim::deserializeSimDesignInfo(ifs_design, design_read_back);
    assert(isSimDesignInfoIdentical(designInfo, design_read_back));


    // Debug symbols
    auto outputSymbolFileFullName = std::filesystem::path(outputDirectory.getValue()) / outputSymbolFilename.getValue();
    std::ofstream ofs_symbol(outputSymbolFileFullName, std::ios::binary | std::ios::out);
    toucanGPUSim::serializeSimDebugInfo(ofs_symbol, debugInfo);
    ofs_symbol.close();

    // IO signal only
    debugInfo.memDebugInfo.clear();
    debugInfo.signalDebugInfo.clear();
    std::erase_if(debugInfo.regDebugInfo, [&](auto const &item) {
      auto const& [k, v] = item;
      return !(scheduler.codeGenInfo.ioSignals.contains(k));
    });

    auto outputIOSymbolFileFullName = std::filesystem::path(outputDirectory.getValue()) / outputIOSymbolFilename.getValue();
    std::ofstream ofs_io_symbol(outputIOSymbolFileFullName, std::ios::binary | std::ios::out);
    toucanGPUSim::serializeSimDebugInfo(ofs_io_symbol, debugInfo);
    ofs_io_symbol.close();

    llvm::outs() << "Done\n";
  }

};

std::unique_ptr<mlir::Pass> toucan::createGPUCodeGenPass(GPUCodeGenOptions option) {
  return std::make_unique<GPUCodeGenPass>(option);
}
