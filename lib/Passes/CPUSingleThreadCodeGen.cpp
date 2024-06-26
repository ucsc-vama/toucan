
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


#define GEN_PASS_DEF_CPUSINGLETHREADCODEGEN
#include "toucan/ToucanPassCommon.h"

#include "toucan/CodeGenCommon.h"
#include "ToucanSim/ToucanGenDataTypes.h"

using namespace toucan;
using namespace circt;
using namespace mlir;
using namespace llvm;

#define DEBUG_TYPE "CPUSingleThreadCodeGenPass"

struct CPUSingleThreadCodeGenPass : toucan::impl::CPUSingleThreadCodeGenBase<CPUSingleThreadCodeGenPass>, CodeGenHelper {
  using CPUSingleThreadCodeGenBase<CPUSingleThreadCodeGenPass>::CPUSingleThreadCodeGenBase;


  void runOnOperation() final {
    // Mark all analyses as preserved. This is a read only pass
    markAllAnalysesPreserved();

    auto graph = getAnalysis<DesignGraph>();

    auto p = NaivePartitioner();
    auto result = p.partitionAndSchedule(graph);
    assert(succeeded(result));

    toucanSim::SimDesignInfo designInfo;
    toucanSim::SimDebugInfo debugInfo;

    // fill lut
    populateLUT();
    designInfo.lut.assign(lutContent.begin(), lutContent.end());

    designInfo.regPoolSize = p.codeGenInfo.regPool.size();
    designInfo.memPoolSize = p.codeGenInfo.totalMemSize;

    for (size_t partId = 0; partId < p.codeGenInfo.partitionInfo.size(); partId++) {
      auto &part = p.codeGenInfo.partitionInfo[partId];

      toucanSim::SimPartitionInfo partInfo;
      partInfo.valuePool.resize(part.numConstsInValuePool);
      for (size_t i = 0; i < part.numConstsInValuePool; i++) {
        assert(part.valuePool[i].isConst);
        partInfo.valuePool[i] = part.valuePool[i].value;
      } 
      partInfo.valuePoolSize = part.valuePool.size();

      partInfo.ops_l0.resize(part.opPool[0].size());
      for (size_t i = 0; i < part.opPool[0].size(); i++) {
        auto &opMeta = part.opPool[0][i];
        assert(opMeta.opName == CGToucanOPName::RegRead);
        partInfo.ops_l0[i].reg = opMeta.regRead.reg;
        partInfo.ops_l0[i].result = opMeta.regRead.result;
      }

      // ops, middle levels
      assert(part.opPool.size() >= 3);
      partInfo.ops_exec.resize(part.opPool.size() - 2);
      for (size_t levelId = 1; levelId < part.opPool.size() - 1; levelId++) {
        auto execOpsIdx = levelId - 1;
        auto &currentLevel = part.opPool[levelId];

        uint32_t numMemReads = 0;
        uint32_t numVecReads = 0;
        uint32_t numLuts = 0;

        partInfo.ops_exec[execOpsIdx].resize(currentLevel.size());
        for (size_t i = 0; i < currentLevel.size(); i++) {
          auto &opMeta = currentLevel[i];
          partInfo.ops_exec[execOpsIdx][i].opType = static_cast<uint8_t>(opMeta.opName);
          switch (opMeta.opName) {
          case CGToucanOPName::LUT: {
            auto lutIndex = lutPos[static_cast<uint32_t>(opMeta.lut.lutId)];
            partInfo.ops_exec[execOpsIdx][i].lut.lutIndex = lutIndex;
            partInfo.ops_exec[execOpsIdx][i].lut.op0 = opMeta.lut.op0;
            partInfo.ops_exec[execOpsIdx][i].lut.op1 = opMeta.lut.op1;
            partInfo.ops_exec[execOpsIdx][i].lut.op2 = opMeta.lut.op2;
            partInfo.ops_exec[execOpsIdx][i].lut.result = opMeta.lut.result;

            numLuts++;
            break;
          }
          case CGToucanOPName::VecRead: {
            assert(opMeta.vec.vecLength < UINT16_MAX && "Vector is too long");

            partInfo.ops_exec[execOpsIdx][i].vec.vecBase = opMeta.vec.vecBase;
            partInfo.ops_exec[execOpsIdx][i].vec.vecLength = opMeta.vec.vecLength;
            partInfo.ops_exec[execOpsIdx][i].vec.index0 = opMeta.vec.index0;
            partInfo.ops_exec[execOpsIdx][i].vec.index1 = opMeta.vec.index1;
            partInfo.ops_exec[execOpsIdx][i].vec.index2 = opMeta.vec.index2;
            partInfo.ops_exec[execOpsIdx][i].vec.index3 = opMeta.vec.index3;
            partInfo.ops_exec[execOpsIdx][i].vec.outRangeValue = opMeta.vec.outRangeValue;
            partInfo.ops_exec[execOpsIdx][i].vec.offset = opMeta.vec.offset;
            partInfo.ops_exec[execOpsIdx][i].vec.result = opMeta.vec.result;

            numVecReads++;
            break;
          }
          case CGToucanOPName::MemRead: {
            partInfo.ops_exec[execOpsIdx][i].mem.hasMultipleWriter = opMeta.memRead.hasMultipleWriter;
            partInfo.ops_exec[execOpsIdx][i].mem.memDepth = opMeta.memRead.memDepth;
            partInfo.ops_exec[execOpsIdx][i].mem.memBase = opMeta.memRead.memBase;
            partInfo.ops_exec[execOpsIdx][i].mem.en = opMeta.memRead.en;
            partInfo.ops_exec[execOpsIdx][i].mem.addrVec = opMeta.memRead.addrVec;
            partInfo.ops_exec[execOpsIdx][i].mem.result = opMeta.memRead.result;

            numMemReads++;
            break;
          }
          default:
            llvm::dbgs() << static_cast<uint32_t>(opMeta.opName);
            llvm_unreachable("Other ops should not appear here");
          }
        }

        partInfo.opInfo_exec.push_back(std::tuple(numMemReads, numVecReads, numLuts));
        assert(partInfo.ops_exec[execOpsIdx].size() == (numMemReads + numVecReads + numLuts));
      }

      // ops, last level
      uint32_t numRegWrites = 0;
      uint32_t numMemWrites = 0;
      uint32_t numPrints = 0;
      uint32_t numStops = 0;

      partInfo.ops_last.resize(part.opPool.back().size());
      for (size_t i = 0; i < part.opPool.back().size(); i++) {
        auto &opMeta = part.opPool.back()[i];
        partInfo.ops_last[i].opType = static_cast<uint8_t>(opMeta.opName);
        switch (opMeta.opName) {
        case CGToucanOPName::Print: {
          partInfo.ops_last[i].print.en = opMeta.print.en;
          partInfo.ops_last[i].print.msg = opMeta.print.msg;
          numPrints++;
          break;
        }
        case CGToucanOPName::Stop: {
          partInfo.ops_last[i].stop.en = opMeta.stop.en;
          numStops++;
          break;
        }
        case CGToucanOPName::RegWrite: {
          partInfo.ops_last[i].regWrite.reg = opMeta.regWrite.reg;
          partInfo.ops_last[i].regWrite.dat = opMeta.regWrite.dat;
          numRegWrites++;
          break;
        }
        case CGToucanOPName::MemWrite: {
          partInfo.ops_last[i].memWrite.hasMultipleWriter = opMeta.memWrite.hasMultipleWriter;
          partInfo.ops_last[i].memWrite.memDepth = opMeta.memWrite.memDepth;
          partInfo.ops_last[i].memWrite.memBase = opMeta.memWrite.memBase;
          partInfo.ops_last[i].memWrite.addrVec = opMeta.memWrite.addrVec;
          partInfo.ops_last[i].memWrite.dat = opMeta.memWrite.dat;
          partInfo.ops_last[i].memWrite.en = opMeta.memWrite.en;
          numMemWrites++;
          break;
        }
        default:
          llvm_unreachable("Other type of ops should not appear in last level!");
        }
      }

      partInfo.opInfo_last = std::tuple(numRegWrites, numMemWrites, numPrints, numStops);
      assert(partInfo.ops_last.size() == (numRegWrites + numMemWrites + numPrints + numStops));

      designInfo.parts.push_back(std::move(partInfo));
    }

    designInfo.printMsgs.resize(p.codeGenInfo.printStrings.size());
    for (auto [k, v]: p.codeGenInfo.printStrings) {
      assert(designInfo.printMsgs[v].empty());
      designInfo.printMsgs[v] = k;
    }

    // debug info
    std::vector<std::tuple<uint32_t, uint32_t>> eachRegDbgInfo;
    std::vector<std::tuple<uint32_t, uint32_t, uint32_t>> eachMemDbgInfo;
    std::vector<std::tuple<uint32_t, uint32_t, uint32_t>> eachSignalDbgInfo;

    for (auto [regNameRef, ids]: p.codeGenInfo.regDebugInfo) {
      eachRegDbgInfo.clear();
      for (auto regId: ids) {
        eachRegDbgInfo.push_back(std::make_tuple(regId, p.codeGenInfo.regPool[regId].bitWidth));
      }
      debugInfo.regDebugInfo[regNameRef.str()] = eachRegDbgInfo;
    }
    for (auto [memNameRef, ids]: p.codeGenInfo.memDebugInfo) {
      eachMemDbgInfo.clear();
      for (auto memId: ids) {
        auto startPos = p.codeGenInfo.memPool[memId].memBase;
        auto bitWidth = p.codeGenInfo.memPool[memId].bitWidth;
        auto memDepth = p.codeGenInfo.memPool[memId].memDepth;
        eachMemDbgInfo.push_back(std::make_tuple(startPos, bitWidth, memDepth));
      }
      debugInfo.memDebugInfo[memNameRef.str()] = eachMemDbgInfo;
    }
    for (auto [sigNameRef, sigLocs]: p.codeGenInfo.signalDebugInfo) {
      eachSignalDbgInfo.clear();

      for (auto sigLoc: sigLocs) {
        auto partId = std::get<0>(sigLoc);
        auto sigId = std::get<1>(sigLoc);
        auto sigBitWidth = p.codeGenInfo.partitionInfo[partId].valuePool[sigId].bitWidth;
        eachSignalDbgInfo.push_back(std::make_tuple(partId, sigId, sigBitWidth));
      }
      debugInfo.signalDebugInfo[sigNameRef.str()] = eachSignalDbgInfo;
    }



    // serialize
    auto outputDesignFileFullName = std::filesystem::path(outputDirectory.getValue()) / outputDesignFilename.getValue();
    std::ofstream ofs_design(outputDesignFileFullName, std::ios::binary | std::ios::out);
    toucanSim::serializeSimDesignInfo(ofs_design, designInfo);
    ofs_design.close();

    // Debug symbols
    auto outputSymbolFileFullName = std::filesystem::path(outputDirectory.getValue()) / outputSymbolFilename.getValue();
    std::ofstream ofs_symbol(outputSymbolFileFullName, std::ios::binary | std::ios::out);
    toucanSim::serializeSimDebugInfo(ofs_symbol, debugInfo);
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
    toucanSim::serializeSimDebugInfo(ofs_io_symbol, debugInfo);
    ofs_io_symbol.close();


  }

};

std::unique_ptr<mlir::Pass> toucan::createCPUSingleThreadCodeGenPass(CPUSingleThreadCodeGenOptions option) {
  return std::make_unique<CPUSingleThreadCodeGenPass>(option);
}