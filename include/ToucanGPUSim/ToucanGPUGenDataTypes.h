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
    uint32_t result;
  };


  struct CGLUTMetaInfo {
    // lut size: 4898
    uint16_t lutIndex;

    uint32_t op0;
    uint32_t op1;
    uint32_t op2;

    uint32_t result;
  };

  struct CGVecReadMetaInfo {
    uint32_t vecBase;
    // Note: vecLength is static
    uint16_t vecLength;
    // Note: This offset is a static value!!!!
    uint16_t offset;
    
    // max addr width: 16
    uint32_t index0;
    uint32_t index1;
    uint32_t index2;
    uint32_t index3;
    uint32_t outRangeValue;

    uint32_t result;
  };

  struct CGMemReadMetaInfo {
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

  // exec level nodes
  struct CGExecLevelMetaInfo {
    uint8_t opType;
    union {
      CGLUTMetaInfo lut;
      CGVecReadMetaInfo vec;
      CGMemReadMetaInfo mem;
    };
  };

  // last level: regWrite, memWrite, stop, print
  struct CGPrintMetaInfo {
    uint32_t en;
    uint32_t msg;
  };
  struct CGStopMetaInfo {
    uint32_t en;
  };
  struct CGRegWriteMetaInfo {
    uint32_t reg;
    uint32_t dat;
  };
  struct CGMemWriteMetaInfo {
    // const
    bool hasMultipleWriter;
    // const
    uint32_t memDepth;
    uint64_t memBase;

    uint32_t addrVec;
    uint32_t dat;
    uint32_t en;
  };

  struct CGLastLevelMetaInfo {
    uint8_t opType;
    union {
      CGPrintMetaInfo print;
      CGStopMetaInfo stop;
      CGRegWriteMetaInfo regWrite;
      CGMemWriteMetaInfo memWrite;
    };
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

    std::vector<CGRegReadMetaInfo> ops_l0;
    std::vector<std::vector<CGExecLevelMetaInfo> > ops_exec;
    std::vector<CGLastLevelMetaInfo> ops_last;

    // Within each layer, place memReads first, then vecReads, luts are the last
    std::vector<std::tuple<uint32_t, uint32_t, uint32_t>> opInfo_exec;
    // Last level: regWrite, memWrite, print, stop
    std::tuple<uint32_t, uint32_t, uint32_t, uint32_t> opInfo_last;
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