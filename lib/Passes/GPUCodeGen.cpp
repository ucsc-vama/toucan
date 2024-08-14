
#include "circt/Dialect/SV/SVDialect.h"
#include "circt/Dialect/OM/OMDialect.h"
#include "circt/Dialect/Seq/SeqDialect.h"
#include "circt/Support/LLVM.h"

#include "mlir/IR/BuiltinOps.h"
#include "mlir/IR/BuiltinTypes.h"
#include "mlir/IR/OpImplementation.h"
#include "mlir/IR/Operation.h"
#include "mlir/IR/Visitors.h"
#include "mlir/Interfaces/SideEffectInterfaces.h"
#include "mlir/Support/LLVM.h"
#include "mlir/IR/Threading.h"
#include "mlir/Support/LogicalResult.h"
#include "toucan/PartitioningGraph.h"
#include "toucan/ToucanAnalysis.h"

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
#include <memory>
#include <filesystem>
#include <fstream>
#include <tuple>
#include <vector>


#define GEN_PASS_DEF_GPUCODEGEN
#include "toucan/ToucanPassCommon.h"

#include "toucan/CodeGenCommon.h"
#include "ToucanGPUSim/ToucanGPUGenDataTypes.h"

using namespace toucan;
using namespace circt;
using namespace mlir;
using namespace llvm;

#define DEBUG_TYPE "GPUCodeGenPass"

struct GPUCodeGenPass : toucan::impl::GPUCodeGenBase<GPUCodeGenPass>, CodeGenHelper {
  using GPUCodeGenBase<GPUCodeGenPass>::GPUCodeGenBase;

  toucanGPUSim::SimDesignInfo designInfo;
  toucanGPUSim::SimDebugInfo debugInfo;


  void populateSinglePartition(const CGPartitionMetaInfo &part) {
    toucanGPUSim::SimPartitionInfo partInfo;
    // have at least 2 levels. one for input, one for output
    assert(part.opPool.size() >= 2);

    partInfo.valuePool.resize(part.numConstsInValuePool);
    for (size_t i = 0; i < part.numConstsInValuePool; i++) {
      assert(part.valuePool[i].isConst);
      partInfo.valuePool[i] = part.valuePool[i].value;
    } 
    partInfo.valuePoolSize = part.valuePool.size();
    partInfo.numConstsInValuePool = part.numConstsInValuePool;
    if (partInfo.valuePoolSize > UINT16_MAX) {
      llvm::errs() << "Value pool size is " << partInfo.valuePoolSize << ", which exceeds UINT16_MAX. This should not happen.\n";
    }
    assert(partInfo.valuePoolSize <= UINT16_MAX && "Value pool is too large");

    for (size_t i = 0; i < part.opPool[0].size(); i++) {
      auto &opMeta = part.opPool[0][i];

      switch (opMeta.opName) {
        case CGToucanOPName::RegRead: {
          assert(opMeta.regRead.result <= UINT16_MAX);

          toucanGPUSim::CGRegReadMetaInfo info;
          info.reg = opMeta.regRead.reg;
          info.result = static_cast<uint16_t>(opMeta.regRead.result);

          partInfo.ops_l0_regRead.push_back(info);
          break;
        }
        case CGToucanOPName::ExchangeRead: {
          assert(opMeta.exgRead.localVal <= UINT16_MAX);

          toucanGPUSim::CGExchangeReadMetaInfo info;
          info.exchangeVal = opMeta.exgRead.exchangeVal;
          info.localVal = opMeta.exgRead.localVal;

          partInfo.ops_l0_exgRead.push_back(info);
          break;
        }
        default: {
          llvm_unreachable("Unknow op name appears in first level");
        }
      }
    }

    // ops, middle levels
    for (size_t levelId = 1; levelId < part.opPool.size() - 1; levelId++) {
      auto execOpsIdx = levelId - 1;
      auto &currentLevel = part.opPool[levelId];

      partInfo.ops_exec_memRead.emplace_back();
      partInfo.ops_exec_vecRead.emplace_back();
      partInfo.ops_exec_lut.emplace_back();

      for (size_t i = 0; i < currentLevel.size(); i++) {
        auto &opMeta = currentLevel[i];
        switch (opMeta.opName) {
          case CGToucanOPName::LUT: {
            assert(opMeta.lut.op0 <= UINT16_MAX);
            assert(opMeta.lut.op1 <= UINT16_MAX);
            assert(opMeta.lut.op2 <= UINT16_MAX);
            assert(opMeta.lut.result <= UINT16_MAX);
            toucanGPUSim::CGLUTMetaInfo info;

            auto lutIndex = lutPos[static_cast<uint32_t>(opMeta.lut.lutId)];
            info.lutIndex = lutIndex;
            info.op0 = opMeta.lut.op0;
            info.op1 = opMeta.lut.op1;
            info.op2 = opMeta.lut.op2;
            info.result = opMeta.lut.result;

            partInfo.ops_exec_lut.back().push_back(info);
            break;
          }
          case CGToucanOPName::VecRead: {
            assert(opMeta.vec.index0 <= UINT16_MAX);
            assert(opMeta.vec.index1 <= UINT16_MAX);
            assert(opMeta.vec.index2 <= UINT16_MAX);
            assert(opMeta.vec.index3 <= UINT16_MAX);
            assert(opMeta.vec.outRangeValue <= UINT16_MAX);
            assert(opMeta.vec.result <= UINT16_MAX);

            assert(opMeta.vec.vecLength < UINT16_MAX && "Vector is too long");

            toucanGPUSim::CGVecReadMetaInfo info;

            info.vecBase = opMeta.vec.vecBase;
            info.vecLength = opMeta.vec.vecLength;
            info.index0 = opMeta.vec.index0;
            info.index1 = opMeta.vec.index1;
            info.index2 = opMeta.vec.index2;
            info.index3 = opMeta.vec.index3;
            info.outRangeValue = opMeta.vec.outRangeValue;
            info.offset = opMeta.vec.offset;
            info.result = opMeta.vec.result;

            partInfo.ops_exec_vecRead.back().push_back(info);
            break;
          }
          case CGToucanOPName::MemRead: {
            assert(opMeta.memRead.en <= UINT16_MAX);
            assert(opMeta.memRead.addrVec <= UINT16_MAX);
            assert(opMeta.memRead.result <= UINT16_MAX);

            toucanGPUSim::CGMemReadMetaInfo info;

            info.hasMultipleWriter = opMeta.memRead.hasMultipleWriter;
            info.memDepth = opMeta.memRead.memDepth;
            info.memBase = opMeta.memRead.memBase;
            info.en = opMeta.memRead.en;
            info.addrVec = opMeta.memRead.addrVec;
            info.result = opMeta.memRead.result;

            partInfo.ops_exec_memRead.back().push_back(info);
            break;
          }
          default: {
            llvm::dbgs() << static_cast<uint32_t>(opMeta.opName);
            llvm_unreachable("Other ops should not appear here");
          }
        }
      }
    }

    // ops, last level
    for (size_t i = 0; i < part.opPool.back().size(); i++) {
      auto &opMeta = part.opPool.back()[i];
      switch (opMeta.opName) {
        case CGToucanOPName::Print: {
          assert(opMeta.print.en <= UINT16_MAX);
          assert(opMeta.print.msg <= UINT16_MAX);

          toucanGPUSim::CGPrintMetaInfo info;
          info.en = opMeta.print.en;
          info.msg = opMeta.print.msg;
          
          partInfo.ops_last_print.push_back(info);
          break;
        }
        case CGToucanOPName::Stop: {
          assert(opMeta.stop.en <= UINT16_MAX);

          toucanGPUSim::CGStopMetaInfo info;
          info.en = opMeta.stop.en;
          
          partInfo.ops_last_stop.push_back(info);
          break;
        }
        case CGToucanOPName::RegWrite: {
          assert(opMeta.regWrite.dat <= UINT16_MAX);

          toucanGPUSim::CGRegWriteMetaInfo info;
          info.reg = opMeta.regWrite.reg;
          info.dat = opMeta.regWrite.dat;
          
          partInfo.ops_last_regWrite.push_back(info);
          break;
        }
        case CGToucanOPName::MemWrite: {
          assert(opMeta.memWrite.addrVec <= UINT16_MAX);
          assert(opMeta.memWrite.dat <= UINT16_MAX);
          assert(opMeta.memWrite.en <= UINT16_MAX);

          toucanGPUSim::CGMemWriteMetaInfo info;
          info.hasMultipleWriter = opMeta.memWrite.hasMultipleWriter;
          info.memDepth = opMeta.memWrite.memDepth;
          info.memBase = opMeta.memWrite.memBase;
          info.addrVec = opMeta.memWrite.addrVec;
          info.dat = opMeta.memWrite.dat;
          info.en = opMeta.memWrite.en;
          
          partInfo.ops_last_memWrite.push_back(info);
          break;
        }
        case CGToucanOPName::ExchangeWrite: {
          assert(opMeta.exgWrite.localVal <= UINT16_MAX);

          toucanGPUSim::CGExchangeWriteMetaInfo info;
          info.localVal = opMeta.exgWrite.localVal;
          info.exchangeVal = opMeta.exgWrite.exchangeVal;
          
          partInfo.ops_last_exgWrite.push_back(info);
          break;
        }
        default: {
          llvm_unreachable("Other type of ops should not appear in last level!");
        }
      }
    }

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

      for (auto sigLoc: sigLocs) {
        auto partId = std::get<0>(sigLoc);
        auto sigId = std::get<1>(sigLoc);
        auto sigBitWidth = codeGenInfo.partitionInfo[partId].valuePool[sigId].bitWidth;
        eachSignalDbgInfo.push_back(std::make_tuple(partId, sigId, sigBitWidth));
      }
      debugInfo.signalDebugInfo[sigNameRef.str()] = eachSignalDbgInfo;
    }
  }

  void runOnOperation() final {
    // Mark all analyses as preserved. This is a read only pass
    markAllAnalysesPreserved();

    auto graph = getAnalysis<DesignGraph>();

    auto p = RepCutPartitioner(outputDirectory.getValue());

    auto rawGraphNumVtxes = boost::num_vertices(graph.g);
    auto eachRegionVtxes = rawGraphNumVtxes / numRegions;
    auto preferredPartitionSize = 10000;
    auto preferredPartCount = eachRegionVtxes / preferredPartitionSize;
    llvm::outs() << "Preferred Part count is " << preferredPartCount << "\n";
    llvm::outs() << "Create " << numSMs << " partitions\n";

    // TODO: What's the best policy to determine numPartsInEachRegion?
    // For now, simply use numSMs
    p.setPartitionTarget(numRegions, numSMs);
    assert(ibFactor > 0.0f);
    p.targetIb = ibFactor;

    auto result = p.partitionAndSchedule(&getContext(), graph);

    if (failed(result)) {
      return signalPassFailure();
    }


    // Fill lut
    populateLUT();
    designInfo.lut.assign(lutContent.begin(), lutContent.end());

    // copy pool size
    designInfo.regPoolSize = p.codeGenInfo.regPool.size();
    designInfo.memPoolSize = p.codeGenInfo.totalMemSize;
    designInfo.exchangePoolSize = p.codeGenInfo.exchangePool.size();

    uint32_t partId = 0;
    for (const auto &eachRegionParts: p.codeGenInfo.regionPartitionIds) {
      designInfo.regionPartitionIds.emplace_back();
      for (const auto &eachPartId: eachRegionParts) {
        assert(eachPartId == partId);

        // save partition region info
        designInfo.regionPartitionIds.back().push_back(partId);
        // codegen for a single partition
        populateSinglePartition(p.codeGenInfo.partitionInfo[partId]);
        partId++;
      }
    }

    // Fill print msgs
    designInfo.printMsgs.resize(p.codeGenInfo.printStrings.size());
    for (auto [k, v]: p.codeGenInfo.printStrings) {
      assert(designInfo.printMsgs[v].empty());
      designInfo.printMsgs[v] = k;
    }

    // Fill debug info
    populateDebugInfo(p.codeGenInfo);



    // Save. serialize
    auto outputDesignFileFullName = std::filesystem::path(outputDirectory.getValue()) / outputDesignFilename.getValue();
    std::ofstream ofs_design(outputDesignFileFullName, std::ios::binary | std::ios::out);
    toucanGPUSim::serializeSimDesignInfo(ofs_design, designInfo);
    ofs_design.close();


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
      return !(p.codeGenInfo.ioSignals.contains(k));
    });

    auto outputIOSymbolFileFullName = std::filesystem::path(outputDirectory.getValue()) / outputIOSymbolFilename.getValue();
    std::ofstream ofs_io_symbol(outputIOSymbolFileFullName, std::ios::binary | std::ios::out);
    toucanGPUSim::serializeSimDebugInfo(ofs_io_symbol, debugInfo);
    ofs_io_symbol.close();

  }

};

std::unique_ptr<mlir::Pass> toucan::createGPUCodeGenPass(GPUCodeGenOptions option) {
  return std::make_unique<GPUCodeGenPass>(option);
}