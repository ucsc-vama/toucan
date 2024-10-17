#pragma once

#include <cstdint>
#include <string>
#include <unordered_map>
#include <vector>
#include <tuple>

#include <fstream>


namespace toucanGPUSim {

  // Reg read, only appear in first level
  struct CGRegReadMetaInfo {
    uint32_t reg;
    uint16_t result;
  };

  struct CGExchangeReadMetaInfo {
    uint32_t exchangeVal;
    uint16_t localVal;
  };

  struct CGExchangeWriteMetaInfo {
    uint16_t count;
    uint16_t localVal;
    uint32_t exchangeVal;
  };


  struct CGLUTMetaInfo {
    // lut size: 4898
    uint16_t lutIndex;

    uint16_t op0;
    uint16_t op1;
    uint16_t op2;

    uint16_t result;
  };

  struct CGVecReadMetaInfo {
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

  struct CGMemReadMetaInfo {
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
    uint32_t memDepth;
    uint64_t memBase;

    uint16_t addrVec;
    uint16_t dat;
    uint16_t en;
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
    uint32_t numConstsInValuePool;

    // Vector holds const vec elements
    std::vector<uint8_t> constVecPool;

    std::vector<CGRegReadMetaInfo> ops_l0_regRead;
    std::vector<CGExchangeReadMetaInfo> ops_l0_exgRead;

    std::vector<std::vector<CGMemReadMetaInfo>> ops_exec_memRead;
    std::vector<std::vector<CGVecReadMetaInfo>> ops_exec_vecRead;
    std::vector<std::vector<CGLUTMetaInfo>> ops_exec_lut;

    std::vector<CGExchangeWriteMetaInfo> ops_last_exgWrite;
    std::vector<CGRegWriteMetaInfo> ops_last_regWrite;
    std::vector<CGMemWriteMetaInfo> ops_last_memWrite;
    std::vector<CGPrintMetaInfo> ops_last_print;
    std::vector<CGStopMetaInfo> ops_last_stop;

  };
  struct SimDesignInfo {
    std::vector<uint8_t> lut;

    // Note: regPool and memPool are empty.
    // Leave randomization to simulator.
    std::vector<uint8_t> regPool;
    std::vector<uint8_t> memPool;
    std::vector<uint8_t> exchangePool;

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
  


};