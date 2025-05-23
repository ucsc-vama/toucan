
#pragma once

#include "mlir/IR/BuiltinAttributes.h"
#include "mlir/IR/Operation.h"

#include "mlir/IR/Value.h"

#include "mlir/Support/LLVM.h"
#include "llvm/ADT/APInt.h"
#include "llvm/Support/Debug.h"
#include "llvm/Support/ErrorHandling.h"



#include <cstdint>
#include <optional>


#include "toucan/ToucanAttributes.h"
#include "toucan/ToucanOps.h"
#include "toucan/ToucanTypes.h"
#include "toucan/PartitioningGraph.h"


#include <boost/graph/adjacency_list.hpp>

#include <unordered_map>
#include <filesystem>
#include <vector>

namespace toucan {

  struct CGValueMetaInfo {
    bool isConst;
    bool isPlaceholder;
    // value is valid if isConst
    uint8_t value;
    mlir::Operation *definingOp;

    uint32_t levelId;
    // uint32_t opId;
    uint8_t bitWidth;

    std::optional<mlir::StringRef> namehint;
    uint32_t fragment_id;
  };


  struct CGLUTOpMetaInfo {
    uint8_t lutId;
    // assert size == 3
    uint32_t op0;
    uint32_t op1;
    uint32_t op2;

    uint8_t numOprands;

    uint32_t result;
  };
  struct CGVectorReadOpMetaInfo {
    uint32_t vecBase;
    // Note: vecLength is static
    uint16_t vecLength;
    // Note: This offset is a static value!!!!
    bool isConstVec;
    uint16_t offset;
    
    // max addr width: 16
    uint32_t index0;
    uint32_t index1;
    uint32_t index2;
    uint32_t index3;
    uint32_t outRangeValue;

    uint32_t result;
  };

  struct CGPrintOpMetaInfo {
    uint32_t en;
    // Index of target string in string pool
    uint32_t msg;
  };
  struct CGStopOpMetaInfo {
    uint32_t en;
  };
  struct CGRegReadOpMetaInfo {
    // reg: index in global regPool
    uint32_t reg;
    uint32_t result;
  };
  struct CGRegWriteOpMetaInfo {
    uint32_t reg;
    uint32_t dat;
  };
  struct CGMemReadOpMetaInfo {
    // static
    bool hasMultipleWriter;
    // static
    uint32_t memDepth;
    // static
    uint64_t memBase;

    uint32_t en;
    uint32_t addrVec;

    uint32_t result;
  };
  struct CGMemWriteOpMetaInfo {
    bool hasMultipleWriter;
    uint32_t memDepth;
    uint64_t memBase;

    uint32_t addrVec;
    uint32_t dat;
    uint32_t en;
  };

  struct CGExchangeReadMetaInfo {
    uint32_t exchangeVal;
    uint32_t localVal;
  };

  struct CGExchangeWriteMetaInfo {
    uint32_t localVal;
    uint32_t exchangeVal;
  };


  struct CGOpMetaInfo {
    CGToucanOPName opName;
    
    union {
      // top level
      CGMemReadOpMetaInfo memRead;
      CGRegReadOpMetaInfo regRead;
      CGExchangeReadMetaInfo exgRead;

      // middle levels, exec
      CGLUTOpMetaInfo lut;
      CGVectorReadOpMetaInfo vec;

      // last level
      CGMemWriteOpMetaInfo memWrite;
      CGRegWriteOpMetaInfo regWrite;
      CGPrintOpMetaInfo print;
      CGStopOpMetaInfo stop;
      CGExchangeWriteMetaInfo exgWrite;
    };

    std::optional<mlir::StringRef> namehint;
    uint32_t fragment_id;

    mlir::Operation *op;
    uint32_t vtxId;

    void setResult(uint32_t result) {
      switch (opName) {
        case CGToucanOPName::LUT: {
          lut.result = result;
          return;
        }
        case CGToucanOPName::VecRead: {
          vec.result = result;
          return;
        }
        case CGToucanOPName::RegRead: {
          regRead.result = result;
          return;
        }
        case CGToucanOPName::MemRead: {
          memRead.result = result;
          return;
        }
        case CGToucanOPName::ExchangeRead: {
          exgRead.localVal = result;
          return;
        }
        case CGToucanOPName::ExchangeWrite: {
          exgWrite.exchangeVal = result;
          return;
        }

        // ConstDecl & VecDecl: Should not have any op with such type
        case CGToucanOPName::ConstDecl:
        case CGToucanOPName::VecDecl:

        case CGToucanOPName::ShouldNotAppear:
        case CGToucanOPName::Print:
        case CGToucanOPName::Stop:
        case CGToucanOPName::RegWrite:
        case CGToucanOPName::MemWrite: {
          llvm::dbgs() << "Error: should not have result\n";
          assert(false);
        }
        default: llvm_unreachable("Unexpected op");
      }
      llvm::dbgs() << "Error: Unknow codegen op name " << static_cast<uint32_t>(opName) << "\n";
      llvm_unreachable("Should not reach here");
    }
    uint32_t getResult() {
      switch (opName) {
        case CGToucanOPName::LUT: return lut.result;
        case CGToucanOPName::VecRead: return vec.result;
        case CGToucanOPName::RegRead: return regRead.result;
        case CGToucanOPName::MemRead: return memRead.result;
        case CGToucanOPName::ExchangeRead: return exgRead.localVal;
        case CGToucanOPName::ExchangeWrite: return exgWrite.exchangeVal;

        // ConstDecl & VecDecl: Should not have any op with such type
        case CGToucanOPName::ConstDecl:
        case CGToucanOPName::VecDecl:

        case CGToucanOPName::ShouldNotAppear:
        case CGToucanOPName::Print:
        case CGToucanOPName::Stop:
        case CGToucanOPName::RegWrite:
        case CGToucanOPName::MemWrite: {
          llvm::dbgs() << "Error: should not have result\n";
          assert(false);
        }
        default: llvm_unreachable("Unexpected op");
      }
      llvm::dbgs() << "Error: Unknow codegen op name " << static_cast<uint32_t>(opName) << "\n";
      llvm_unreachable("Unsupported op");
    }
    bool hasResult() {
      switch (opName) {
        case CGToucanOPName::LUT:
        case CGToucanOPName::VecRead:
        case CGToucanOPName::RegRead:
        case CGToucanOPName::MemRead: 
        case CGToucanOPName::ExchangeRead:
        case CGToucanOPName::ExchangeWrite:
          return true;

        default: return false;
      }
    }
  };

  struct CGRegMetaInfo {
    std::optional<mlir::StringRef> namehint;
    uint32_t bitWidth;
    uint32_t fragment_id;
    bool isPadding;
    bool isIO;
    mlir::Value val;
  };

  struct CGMemMetaInfo {
    std::optional<mlir::StringRef> namehint;
    uint32_t bitWidth;
    uint64_t memDepth;
    uint32_t fragment_id;
    bool hasMultipleWriter;
    uint64_t memBase;
  };



  struct CGExchangeValueMetaInfo {
    bool isPadding;
    // write id (new)
    uint32_t writerId;
    uint32_t writerRegionId;
    mlir::Value val;
    // region, vtxId (new)
    mlir::SmallVector<std::tuple<uint32_t, uint32_t>> readerIds;
  };

  struct CGLayerValueStatistics {
    // only in first level
    uint32_t numRegReads;
    // only in middle levels
    uint32_t numMemReads;
    uint32_t numVecReads;
    uint32_t numLuts;
    // only in last level
    uint32_t numRegWrites;
    uint32_t numMemWrites;
    uint32_t numPrints;
    uint32_t numStops;
    uint32_t numExchangeReads;
    uint32_t numExchangeWrites;
  };

  struct CGOpStatistics {
    uint32_t numRegReads;
    uint32_t numMemReads;
    uint32_t numVecReads;
    uint32_t numLuts;
    uint32_t numLutNops;
    uint32_t numRegWrites;
    uint32_t numMemWrites;
    uint32_t numPrints;
    uint32_t numStops;
    uint32_t numExchangeReads;
    uint32_t numExchangeWrites;

    void print() const;
  };



  struct CGMicroPartLUTTopLevelOp {
    LUTOpName opName;
    // assert size == 3
    uint16_t op0;
    uint16_t op1;
    uint16_t op2;
  };
  struct CGMicroPartLUTMiddleLevelOp {
    LUTOpName opName;
    uint8_t op0;
    uint8_t op1;
    uint8_t op2;
  };
  struct CGMicroPartLUTLastLevelWriteBack {
    uint8_t shuffleId;

    uint16_t result;
  };

  struct CGMicroPartVecRead {
    uint16_t vecBase;
    // Note: vecLength is static
    uint16_t vecLength;
    // Note: This offset is a static value!!!!
    bool isConstVec;
    uint16_t offset;
    
    // max addr width: 16
    uint16_t index0;
    uint16_t index1;
    uint16_t index2;
    uint16_t index3;
    uint16_t outRangeValue;

    uint16_t result;
  };

  struct CGMicroPartVecArith {
    uint16_t vec1Base;
    uint16_t vec2Base;
    
    uint16_t vecLength;

    bool isVec1Const;
    bool isVec2Const;
    
    toucan::VecArithOpName opName;

    uint16_t result;
  };

  struct CGMicroPartVecLogic {
    uint16_t vec1Base;
    uint16_t vec2Base;
    
    uint16_t vecLength;

    bool isVec1Const;
    bool isVec2Const;
    
    toucan::VecLogicOpName opName;

    uint16_t result;
  };

  struct CGMicroPartMemRead {
    // static
    bool hasMultipleWriter;
    // static
    uint32_t memDepth;
    // static
    uint64_t memBase;

    uint16_t en;
    uint16_t addrVec;

    uint16_t result;
  };


  struct CGMicroPartInfo {
    CGToucanOPName opType;

    mlir::SmallVector<CGMicroPartLUTTopLevelOp> topLevel;
    mlir::SmallVector<mlir::SmallVector<CGMicroPartLUTMiddleLevelOp>> middleLevels;
    mlir::SmallVector<CGMicroPartLUTLastLevelWriteBack> lastLevel;

    // special type
    mlir::SmallVector<CGMicroPartVecRead> vecRead;
    mlir::SmallVector<CGMicroPartVecArith> vecArith;
    mlir::SmallVector<CGMicroPartVecLogic> vecLogic;
    mlir::SmallVector<CGMicroPartMemRead> memRead;

    void clear() {
      topLevel.clear();
      middleLevels.clear();
      lastLevel.clear();
      vecRead.clear();
      vecArith.clear();
      vecLogic.clear();
      memRead.clear();
    }
  };

  // Information needed for code gen, each level
  struct CGPartitionMetaInfo {
    // Values generated by each layer. Will also be used as input of next level
    mlir::SmallVector<CGValueMetaInfo> valuePool;
    uint32_t numTotalValues;
    mlir::SmallVector<uint8_t> constValuePool;
    mlir::SmallVector<uint8_t> constVecPool;
    // Beginning N values are constant
    uint32_t numConstsInValuePool;
    // Operator pool in each level
    // mlir::SmallVector<mlir::SmallVector<CGOpMetaInfo>> opPool;

    mlir::SmallVector<CGOpMetaInfo> regReadOps, regWriteOps, memWriteOps, stopOps, printOps;
    std::vector<std::vector<CGMicroPartInfo>> microPartOps;

    // data structure for scheduling purpose
    mlir::DenseMap<mlir::Value, uint32_t> valueToValId;

    mlir::SmallVector<CGLayerValueStatistics> opStatisticsPerLevel;
    CGOpStatistics opStatistics;
  };

  struct CGInfo {
    // Global register region. 
    mlir::SmallVector<CGRegMetaInfo> regPool;
    // Memories
    mlir::SmallVector<CGMemMetaInfo> memPool;
    // Exchange values
    mlir::SmallVector<CGExchangeValueMetaInfo> exchangePool;

    mlir::SmallVector<mlir::SmallVector<uint32_t>> regionPartitionIds;

    uint64_t totalMemSize;

    mlir::DenseMap<mlir::StringRef, uint32_t> printStrings;

    // debug info
    // name -> (fragment 0, 1, 2, ...)
    mlir::DenseMap<mlir::StringRef, mlir::SmallVector<uint32_t>> regDebugInfo;
    // name -> ((part, valId), (part, valId), ..)
    mlir::DenseMap<mlir::StringRef, mlir::SmallVector<std::tuple<uint32_t, uint32_t>>> signalDebugInfo;
    // name -> vector of all mem ids with same namehint. Start from fragment 0 to n
    mlir::DenseMap<mlir::StringRef, mlir::SmallVector<uint32_t>> memDebugInfo;

    // For developing purpose
    mlir::DenseMap<mlir::TypedValue<toucan::RegType>, uint32_t> toucanRegToId;
    mlir::DenseMap<mlir::TypedValue<toucan::MemType>, uint32_t> toucanMemToId;

    mlir::SmallVector<CGPartitionMetaInfo, 4> partitionInfo;

    mlir::DenseSet<mlir::StringRef> ioSignals;
  };
}