#pragma once

#include <cstdint>
#include <string>
#include <unordered_map>
#include <vector>
#include <tuple>

#include <fstream>
#include <cassert>


namespace toucanGPUSim {

  // const uint16_t MAX_REGREAD_BYTECOUNT = 2;

  // Reg read, only appear in first level
  struct CGRegReadMetaInfo {
    uint32_t reg;
    uint16_t result;
  };

  struct CGExchangeReadMetaInfo {
    uint32_t exchange;
    uint16_t result;
  };
  struct CGExchangeWriteMetaInfo {
    uint32_t exchange;
    uint16_t dat;
    uint16_t count;
  };

  // last level: regWrite, memWrite, stop, print
  struct CGPrintMetaInfo {
    uint16_t en;
    uint16_t msg;
  };
  struct CGStopMetaInfo {
    uint16_t en;
  };
  struct CGRegWriteMetaInfo {
    uint32_t reg;
    uint16_t dat;
    uint16_t count;
  };
  struct CGMemWriteMetaInfo {
    // const
    bool hasMultipleWriter;
    // const
    // uint32_t memDepth;
    uint64_t memBase;

    uint16_t addrVec;
    uint16_t dat;
    uint16_t en;
  };




  struct CGMicroPartLUTTopLevelOp {
    // lut size: 4898
    uint16_t lutIndex;
    uint16_t op0;
    uint16_t op1;
    uint16_t op2;
  };

  // Note: old version
  // struct CGMicroPartLUTMiddleLevelOp {
  //   // lut size: 4898
  //   uint16_t lutIndex;
  //   // 0~15: const, 32 ~ 63: value from other thread
  //   uint8_t op0;
  //   uint8_t op1;
  //   uint8_t op2;
  // };


  // Note: pack all fields into 32 bits (performance consideration)
  // Should work on both CPU and GPU (nvcc)
  #ifdef __CUDACC__
  #define HOST_DEVICE __host__ __device__
  #else
  #define HOST_DEVICE
  #endif


  #pragma pack(push, 1)
  struct CGMicroPartLUTMiddleLevelOp {
  private:
    uint32_t packed_data;

  public:
    HOST_DEVICE CGMicroPartLUTMiddleLevelOp() : packed_data(0) {}

    HOST_DEVICE explicit CGMicroPartLUTMiddleLevelOp(
      uint16_t lutIndex, uint8_t op0, uint8_t op1, uint8_t op2
    ) : packed_data(0) {
      setLutIndex(lutIndex);
      setOp0(op0);
      setOp1(op1);
      setOp2(op2);
    }

    HOST_DEVICE uint16_t lutIndex() const { return (packed_data >> 18) & 0x3FFF; }
    HOST_DEVICE uint8_t op0() const { return (packed_data >> 12) & 0x3F; }
    HOST_DEVICE uint8_t op1() const { return (packed_data >> 6) & 0x3F; }
    HOST_DEVICE uint8_t op2() const { return packed_data & 0x3F; }

    HOST_DEVICE void setLutIndex(uint16_t val) {
      assert(val <= 0x3FFF); // 14-bit
      packed_data = (packed_data & ~(0x3FFF << 18)) | ((val & 0x3FFF) << 18);
    }

    HOST_DEVICE void setOp0(uint8_t val) {
      assert(val <= 0x3F); // 6b
      packed_data = (packed_data & ~(0x3F << 12)) | ((val & 0x3F) << 12);
    }

    HOST_DEVICE void setOp1(uint8_t val) {
      assert(val <= 0x3F);
      packed_data = (packed_data & ~(0x3F << 6)) | ((val & 0x3F) << 6);
    }

    HOST_DEVICE void setOp2(uint8_t val) {
      assert(val <= 0x3F);
      packed_data = (packed_data & ~0x3F) | (val & 0x3F);
    }

    HOST_DEVICE uint32_t getPacked() const   { return packed_data; }
    HOST_DEVICE void setPacked(uint32_t val) { packed_data = val; }
  };
  #pragma pack(pop)

  static_assert(sizeof(CGMicroPartLUTMiddleLevelOp) == 4, "Struct size must be 4 bytes");




  struct CGMicroPartLUTLastLevelWriteBack {
    // 0~31
    uint8_t shuffleId;

    uint16_t result;
  };

  struct CGMicroPartVecRead {
    uint16_t vecBase;
    // Note: vecLength is static
    uint16_t vecLength;
    // Note: This offset is a static value!!!!
    uint16_t offset;
    bool isConstVec;
    
    // max addr width: 16
    uint16_t index0;
    uint16_t index1;
    uint16_t index2;
    uint16_t index3;
    uint16_t outRangeValue;

    uint16_t result;
  };

  #define VEC_ARITH_ADD 0
  #define VEC_ARITH_SUB 1
  #define VEC_ARITH_MUL 2

  #define VEC_LOGIC_EQ 3
  #define VEC_LOGIC_LT 4
  #define VEC_LOGIC_LE 5
  struct CGMicroPartVecArithOrLogic {
    uint16_t vec1Base;
    uint16_t vec2Base;

    uint16_t vecLength;
    uint8_t opName;
    // 00, 01, 10, 11
    uint8_t isV1V2Const;

    uint16_t result;
  };


  // struct CGMicroPartVecLogic {
  //   bool isVec1Const;
  //   bool isVec2Const;

  //   uint16_t vec1Base;
  //   uint16_t vec2Base;

  //   uint8_t vecLength;
  //   uint8_t opName;

  //   uint16_t result;
  // };

  struct CGMicroPartMemRead {
    // static
    bool hasMultipleWriter;
    // static
    // uint32_t memDepth;
    // static
    uint64_t memBase;

    uint16_t en;
    uint16_t addrVec;

    uint16_t result;
  };

  struct CGMicroPartInfo {
    bool isLUTPart;

    // valid for lut part
    std::vector<CGMicroPartLUTTopLevelOp> topLevel;
    std::vector<std::vector<CGMicroPartLUTMiddleLevelOp>> middleLevels;
    std::vector<CGMicroPartLUTLastLevelWriteBack> lastLevel;

    // special type
    std::vector<CGMicroPartVecRead> vecRead;
    std::vector<CGMicroPartVecArithOrLogic> vecArithAndLogic;
    std::vector<CGMicroPartMemRead> memRead;
  };


  struct SimDebugInfo {
    // name -> ((id, wdith), (id, width), ...)
    std::unordered_map<std::string, std::vector<std::tuple<uint32_t, uint32_t>> > regDebugInfo;
    // name -> ((part, valId, width), (part, valId, width), ..)
    std::unordered_map<std::string, std::vector<std::tuple<uint32_t, uint32_t, uint32_t> > > signalDebugInfo;
    // name -> ((start pos, bit width, length), ...)
    std::unordered_map<std::string, std::vector<std::tuple<uint32_t, uint32_t, uint32_t>> > memDebugInfo;
  };

  struct SimPartitionInfo {
    // valuePool is filled with consts
    std::vector<uint8_t> valuePool;
    uint32_t valuePoolSize;

    // Vector holds const vec elements
    std::vector<uint8_t> constVecPool;

    std::vector<CGRegReadMetaInfo> ops_l0_regRead;
    std::vector<CGExchangeReadMetaInfo> ops_l0_exchangeRead;

    std::vector<std::vector<CGMicroPartInfo>> exec_mParts;

    CGRegWriteMetaInfo op_last_regWrite;
    std::vector<CGMemWriteMetaInfo> ops_last_memWrite;
    std::vector<CGPrintMetaInfo> ops_last_print;
    std::vector<CGStopMetaInfo> ops_last_stop;

    CGExchangeWriteMetaInfo op_last_exchangeWrite;

  };
  struct SimDesignInfo {
    std::vector<uint8_t> lut;

    // Note: regPool and memPool are empty.
    // Leave randomization to simulator.
    std::vector<uint8_t> regPool;
    std::vector<uint8_t> memPool;

    uint32_t regPoolSize;
    uint32_t memPoolSize;
    uint32_t exchangePoolSize;

    std::vector<SimPartitionInfo> parts;

    std::vector<std::vector<uint32_t>> regionPartitionIds;

    bool shouldStop;

    std::vector<std::string> printMsgs;
    // debug info
    // name -> (fragment 0, 1, 2, ...)

    void Init();
    void Randomize(uint32_t seed, SimDebugInfo &symbols);
  };


  // Why write my own serialize/deserialize?
  // Because I cannot find any serialize lib that don't require RTTI and exception, which are disabled by llvm
  void serializeSimDesignInfo(std::ostream& out, const SimDesignInfo& info);
  void deserializeSimDesignInfo(std::istream& in, SimDesignInfo& info);
  void serializeSimDebugInfo(std::ostream& out, const toucanGPUSim::SimDebugInfo& info);
  void deserializeSimDebugInfo(std::istream& in, toucanGPUSim::SimDebugInfo& info);
  

  bool isSimDesignInfoIdentical(const SimDesignInfo& a, const SimDesignInfo& b);
};