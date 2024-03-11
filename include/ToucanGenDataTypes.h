#pragma once

#include <cstdint>
#include <string>
#include <unordered_map>
#include <vector>
#include <tuple>


namespace toucanSim {

  // Reg read, only appear in first level
  struct CGRegReadMetaInfo {
    uint32_t reg;
    uint32_t result;
  };


  struct CGLUTMetaInfo {
    uint8_t lutOpName;

    uint32_t op0;
    uint32_t op1;
    uint32_t op2;

    uint32_t result;
  };

  struct CGVecReadMetaInfo {
    uint32_t vecBase;
    // Note: vecLength is statc
    uint32_t vecLength;
    
    // max addr width: 16
    uint32_t index0;
    uint32_t index1;
    uint32_t index2;
    uint32_t index3;
    uint32_t outRangeValue;
    // Note: This offset is a static value!!!!
    uint16_t offset;

    uint32_t result;
  };

  struct CGMemReadMetaInfo {
    // static
    bool hasMultipleWriter;
    // static
    uint32_t memDepth;
    // static
    uint64_t memBase;

    // max addr width: 32
    uint32_t addr0;
    uint32_t addr1;
    uint32_t addr2;
    uint32_t addr3;
    uint32_t addr4;
    uint32_t addr5;
    uint32_t addr6;
    uint32_t addr7;

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

    // max addr width: 32
    uint32_t addr0;
    uint32_t addr1;
    uint32_t addr2;
    uint32_t addr3;
    uint32_t addr4;
    uint32_t addr5;
    uint32_t addr6;
    uint32_t addr7;

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




  struct SimPartitionInfo {
    std::vector<uint8_t> constValues;
    // Note: valuePool is empty. 
    // First values are consts, fill using constValues
    std::vector<uint8_t> valuePool;
    uint32_t valuePoolSize;

    std::vector<CGRegReadMetaInfo> ops_l0;
    std::vector<std::vector<CGExecLevelMetaInfo> > ops_exec;
    std::vector<CGLastLevelMetaInfo> ops_last;
  };
  struct SimDesignInfo {
    std::vector<uint8_t> lut;
    std::vector<uint32_t> lutIndex;

    // Note: regPool and memPool are empty.
    // Leave randomization to simulator.
    std::vector<uint8_t> regPool;
    std::vector<uint8_t> memPool;
    uint32_t regPoolSize;
    uint32_t memPoolSize;

    std::vector<SimPartitionInfo> parts;

    bool shouldStop;

    std::vector<std::string> printMsgs;
    // debug info
    // name -> (fragment 0, 1, 2, ...)
    std::unordered_map<std::string, std::vector<uint32_t> > regDebugInfo;
    // name -> ((part, valId), (part, valId), ..)
    std::unordered_map<std::string, std::vector<std::tuple<uint32_t, uint32_t> > > signalDebugInfo;
    // name -> (start pos, bit width, length)
    std::unordered_map<std::string, std::vector<uint32_t> > memDebugInfo;

  };


};