#include "ToucanGPUSim/ToucanGPUGenDataTypes.h"
#include <fstream>

#include <iostream>
#include <cassert>
#include <vector>
#include <cstring>

using namespace toucanGPUSim;


// // Endianness representation
// enum class Endianness : uint8_t {
//     LittleEndian = 0,
//     BigEndian = 1
// };

// // Detect host endianness
// static Endianness getHostEndianness() {
//     const uint32_t one = 1;
//     return (*(reinterpret_cast<const uint8_t*>(&one)) == 1) ? Endianness::LittleEndian : Endianness::BigEndian;
// }


namespace toucanGPUSim {

// Primitive type serialization
template<typename T>
static void serializePrimitive(std::ostream& out, const T& value) {
  out.write(reinterpret_cast<const char*>(&value), sizeof(value));
}

template<typename T>
static void deserializePrimitive(std::istream& in, T& value) {
  in.read(reinterpret_cast<char*>(&value), sizeof(value));
}

// Vector serialization
template<typename T>
static void serializeVector(std::ostream& out, const std::vector<T>& vec) {
  serializePrimitive(out, vec.size());
  for (const auto& item : vec) {
    serializePrimitive(out, item);
  }
}

template<typename T>
static void deserializeVector(std::istream& in, std::vector<T>& vec) {
  size_t size;
  deserializePrimitive(in, size);
  vec.resize(size);
  for (size_t i = 0; i < size; ++i) {
    deserializePrimitive(in, vec[i]);
  }
}

// String serialization
static void serializeString(std::ostream& out, const std::string& str) {
  serializeVector(out, std::vector<char>(str.begin(), str.end()));
}

static void deserializeString(std::istream& in, std::string& str) {
  std::vector<char> vec;
  deserializeVector(in, vec);
  str.assign(vec.begin(), vec.end());
}



template<typename T1, typename T2>
static void serializeTuple(std::ostream& out, const std::tuple<T1, T2>& value) {
  serializePrimitive(out, std::get<0>(value));
  serializePrimitive(out, std::get<1>(value));
}

template<typename T1, typename T2>
static void deserializeTuple(std::istream& in, std::tuple<T1, T2>& value) {
  T1 first;
  T2 second;
  deserializePrimitive(in, first);
  deserializePrimitive(in, second);
  value = std::make_tuple(first, second);
}

template<typename T1, typename T2, typename T3>
static void serializeTuple(std::ostream& out, const std::tuple<T1, T2, T3>& value) {
  serializePrimitive(out, std::get<0>(value));
  serializePrimitive(out, std::get<1>(value));
  serializePrimitive(out, std::get<2>(value));
}

template<typename T1, typename T2, typename T3>
static void deserializeTuple(std::istream& in, std::tuple<T1, T2, T3>& value) {
  T1 first;
  T2 second;
  T3 third;
  deserializePrimitive(in, first);
  deserializePrimitive(in, second);
  deserializePrimitive(in, third);
  value = std::make_tuple(first, second, third);
}




static void serializeCGMicroPartInfo(std::ostream& out, const CGMicroPartInfo& info) {
  serializePrimitive(out, info.isLUTPart);

  if (info.isLUTPart) {
    serializeVector(out, info.topLevel);
    serializePrimitive(out, info.middleLevels.size());
    for (const auto& level : info.middleLevels) {
      serializeVector(out, level);
    }
    serializeVector(out, info.lastLevel);
  } else {
    serializeVector(out, info.vecRead);
    serializeVector(out, info.vecArithAndLogic);
    serializeVector(out, info.memRead);
  }
}

static void deserializeCGMicroPartInfo(std::istream& in, CGMicroPartInfo& info) {
  deserializePrimitive(in, info.isLUTPart);

  if (info.isLUTPart) {
    deserializeVector(in, info.topLevel);
    size_t middleLevelsSize;
    deserializePrimitive(in, middleLevelsSize);
    info.middleLevels.resize(middleLevelsSize);
    for (auto& level : info.middleLevels) {
      deserializeVector(in, level);
    }
    deserializeVector(in, info.lastLevel);
    info.vecRead.clear();
    info.vecArithAndLogic.clear();
    info.memRead.clear();
  } else {
    deserializeVector(in, info.vecRead);
    deserializeVector(in, info.vecArithAndLogic);
    deserializeVector(in, info.memRead);
  }
}



static void serializeSimPartitionInfo(std::ostream& out, const SimPartitionInfo& info) {
  serializeVector(out, info.valuePool);
  serializePrimitive(out, info.valuePoolSize);
  serializeVector(out, info.constVecPool);
  serializeVector(out, info.ops_l0_regRead);

  serializePrimitive(out, info.exec_mParts.size());
  for (const auto& mParts : info.exec_mParts) {
    serializePrimitive(out, mParts.size());
    for (const auto& mPart : mParts) {
      serializeCGMicroPartInfo(out, mPart);
    }
  }

  serializePrimitive(out, info.op_last_regWrite);
  serializeVector(out, info.ops_last_memWrite);
  serializeVector(out, info.ops_last_print);
  serializeVector(out, info.ops_last_stop);
}

static void deserializeSimPartitionInfo(std::istream& in, SimPartitionInfo& info) {
  deserializeVector(in, info.valuePool);
  deserializePrimitive(in, info.valuePoolSize);
  deserializeVector(in, info.constVecPool);
  deserializeVector(in, info.ops_l0_regRead);

  size_t exec_mPartsSize;
  deserializePrimitive(in, exec_mPartsSize);
  info.exec_mParts.resize(exec_mPartsSize);

  for (auto& mParts : info.exec_mParts) {
    size_t mPartsSize;
    deserializePrimitive(in, mPartsSize);
    mParts.resize(mPartsSize);

    for (auto& mPart : mParts) {
      deserializeCGMicroPartInfo(in, mPart);
    }
  }

  deserializePrimitive(in, info.op_last_regWrite);
  deserializeVector(in, info.ops_last_memWrite);
  deserializeVector(in, info.ops_last_print);
  deserializeVector(in, info.ops_last_stop);
}




void serializeSimDesignInfo(std::ostream& out, const SimDesignInfo& info) {
  serializePrimitive(out, info.regPoolSize);
  serializePrimitive(out, info.memPoolSize);
  serializePrimitive(out, info.shouldStop);

  serializeVector(out, info.lut);
  serializeVector(out, info.regPool);
  serializeVector(out, info.memPool);

  serializePrimitive(out, info.parts.size());
  for (const auto& part : info.parts) {
    serializeSimPartitionInfo(out, part);
  }

  serializePrimitive(out, info.regionPartitionIds.size());
  for (const auto& partitionIds : info.regionPartitionIds) {
    serializeVector(out, partitionIds);
  }

  serializePrimitive(out, info.printMsgs.size());
  for (const auto& msg : info.printMsgs) {
    serializeString(out, msg);
  }
}


void deserializeSimDesignInfo(std::istream& in, SimDesignInfo& info) {
  deserializePrimitive(in, info.regPoolSize);
  deserializePrimitive(in, info.memPoolSize);
  deserializePrimitive(in, info.shouldStop);

  deserializeVector(in, info.lut);
  deserializeVector(in, info.regPool);
  deserializeVector(in, info.memPool);

  size_t partsSize;
  deserializePrimitive(in, partsSize);
  info.parts.resize(partsSize);

  for (auto& part : info.parts) {
    deserializeSimPartitionInfo(in, part);
  }

  size_t regionPartitionIdsSize;
  deserializePrimitive(in, regionPartitionIdsSize);
  info.regionPartitionIds.resize(regionPartitionIdsSize);

  for (auto& partitionIds : info.regionPartitionIds) {
    deserializeVector(in, partitionIds);
  }

  // Deserialize printMsgs
  size_t printMsgsSize;
  deserializePrimitive(in, printMsgsSize);
  info.printMsgs.resize(printMsgsSize);
  for (auto& msg : info.printMsgs) {
    deserializeString(in, msg);
  }
}




void serializeSimDebugInfo(std::ostream& out, const SimDebugInfo& info) {
  // Serialize regDebugInfo
  size_t regMapSize = info.regDebugInfo.size();
  serializePrimitive(out, regMapSize);
  for (const auto& pair : info.regDebugInfo) {
    serializeString(out, pair.first);
    size_t vectorSize = pair.second.size();
    serializePrimitive(out, vectorSize);
    for (const auto& tuple : pair.second) {
      serializeTuple(out, tuple);
    }
  }

  // Serialize signalDebugInfo
  size_t signalMapSize = info.signalDebugInfo.size();
  serializePrimitive(out, signalMapSize);
  for (const auto& pair : info.signalDebugInfo) {
    serializeString(out, pair.first);
    size_t vectorSize = pair.second.size();
    serializePrimitive(out, vectorSize);
    for (const auto& tuple : pair.second) {
      serializeTuple(out, tuple);
    }
  }

  // Serialize memDebugInfo similarly to regDebugInfo
  size_t memMapSize = info.memDebugInfo.size();
  serializePrimitive(out, memMapSize);
  for (const auto& pair : info.memDebugInfo) {
    serializeString(out, pair.first);
    size_t vectorSize = pair.second.size();
    serializePrimitive(out, vectorSize);
    for (const auto& tuple : pair.second) {
      serializeTuple(out, tuple);
    }
  }
}


void deserializeSimDebugInfo(std::istream& in, SimDebugInfo& info) {
  // Deserialize regDebugInfo
  size_t regMapSize;
  deserializePrimitive(in, regMapSize);
  for (size_t i = 0; i < regMapSize; ++i) {
    std::string key;
    deserializeString(in, key);
    size_t vectorSize;
    deserializePrimitive(in, vectorSize);
    std::vector<std::tuple<uint32_t, uint32_t>> vector;
    for (size_t j = 0; j < vectorSize; ++j) {
      std::tuple<uint32_t, uint32_t> tuple;
      deserializeTuple(in, tuple);
      vector.push_back(tuple);
    }
    info.regDebugInfo[key] = vector;
  }

  // Deserialize signalDebugInfo
  size_t signalMapSize;
  deserializePrimitive(in, signalMapSize);
  for (size_t i = 0; i < signalMapSize; ++i) {
    std::string key;
    deserializeString(in, key);
    size_t vectorSize;
    deserializePrimitive(in, vectorSize);
    std::vector<std::tuple<uint32_t, uint32_t, uint32_t>> vector;
    for (size_t j = 0; j < vectorSize; ++j) {
      std::tuple<uint32_t, uint32_t, uint32_t> tuple;
      deserializeTuple(in, tuple);
      vector.push_back(tuple);
    }
    info.signalDebugInfo[key] = vector;
  }

  // Deserialize memDebugInfo similarly to regDebugInfo
  size_t memMapSize;
  deserializePrimitive(in, memMapSize);
  for (size_t i = 0; i < memMapSize; ++i) {
    std::string key;
    deserializeString(in, key);
    size_t vectorSize;
    deserializePrimitive(in, vectorSize);
    std::vector<std::tuple<uint32_t, uint32_t, uint32_t>> vector;
    for (size_t j = 0; j < vectorSize; ++j) {
      std::tuple<uint32_t, uint32_t, uint32_t> tuple;
      deserializeTuple(in, tuple);
      vector.push_back(tuple);
    }
    info.memDebugInfo[key] = vector;
  }
}



template<typename T>
static bool isPODEqual(const T& a, const T& b) {
  return std::memcmp(&a, &b, sizeof(T)) == 0;
}

template<typename T>
static bool isVectorEqual(const std::vector<T>& a, const std::vector<T>& b) {
  return a.size() == b.size() && 
           (a.empty() || std::memcmp(a.data(), b.data(), a.size() * sizeof(T)) == 0);
}

static bool isVectorOfStringEqual(const std::vector<std::string> &a, const std::vector<std::string> &b) {
  if (a.size() != b.size()) return false;
  for (size_t i = 0; i < a.size(); ++i) {
    if (!(a[i] == b[i])) return false;
  }
  return true;
}

static bool isCGMicroPartInfoEqual(const CGMicroPartInfo& a, const CGMicroPartInfo& b) {
  return a.isLUTPart == b.isLUTPart &&
         std::memcmp(a.topLevel.data(), b.topLevel.data(), 
                    a.topLevel.size() * sizeof(CGMicroPartLUTTopLevelOp)) == 0 &&
         a.middleLevels.size() == b.middleLevels.size() &&
         [&](){
           for (size_t i = 0; i < a.middleLevels.size(); ++i) {
             if (std::memcmp(a.middleLevels[i].data(), b.middleLevels[i].data(),
                            a.middleLevels[i].size() * sizeof(CGMicroPartLUTMiddleLevelOp)) != 0)
               return false;
           }
           return true;
         }() &&
         std::memcmp(a.lastLevel.data(), b.lastLevel.data(),
                    a.lastLevel.size() * sizeof(CGMicroPartLUTLastLevelWriteBack)) == 0 &&
         std::memcmp(a.vecRead.data(), b.vecRead.data(),
                    a.vecRead.size() * sizeof(CGMicroPartVecRead)) == 0 &&
         std::memcmp(a.vecArithAndLogic.data(), b.vecArithAndLogic.data(),
                    a.vecArithAndLogic.size() * sizeof(CGMicroPartVecArithOrLogic)) == 0 &&
         std::memcmp(a.memRead.data(), b.memRead.data(),
                    a.memRead.size() * sizeof(CGMicroPartMemRead)) == 0;
}

bool isSimDesignInfoIdentical(const SimDesignInfo& a, const SimDesignInfo& b) {
  assert(a.regPoolSize == b.regPoolSize);
  assert(a.memPoolSize == b.memPoolSize);
  assert(a.shouldStop == b.shouldStop);

  assert(isVectorEqual(a.lut, b.lut));
  assert(isVectorEqual(a.regPool, b.regPool));
  assert(isVectorEqual(a.memPool, b.memPool));
  assert(isVectorOfStringEqual(a.printMsgs, b.printMsgs));

  assert(a.parts.size() == b.parts.size());
  for (size_t i = 0; i < a.parts.size(); ++i) {
    const auto& partA = a.parts[i];
    const auto& partB = b.parts[i];

    assert(isVectorEqual(partA.valuePool, partB.valuePool));
    assert(partA.valuePoolSize == partB.valuePoolSize);
    assert(isVectorEqual(partA.constVecPool, partB.constVecPool));
    assert(isVectorEqual(partA.ops_l0_regRead, partB.ops_l0_regRead));

    assert(partA.exec_mParts.size() == partB.exec_mParts.size());
    for (size_t j = 0; j < partA.exec_mParts.size(); ++j) {
      assert(partA.exec_mParts[j].size() == partB.exec_mParts[j].size());
      for (size_t k = 0; k < partA.exec_mParts[j].size(); k++) {
        assert(isCGMicroPartInfoEqual(partA.exec_mParts[j][k], partB.exec_mParts[j][k]));
      }
    }

    assert(isPODEqual(partA.op_last_regWrite, partB.op_last_regWrite));
    assert(isVectorEqual(partA.ops_last_memWrite, partB.ops_last_memWrite));
    assert(isVectorEqual(partA.ops_last_print, partB.ops_last_print));
    assert(isVectorEqual(partA.ops_last_stop, partB.ops_last_stop));
  }

  assert(a.regionPartitionIds.size() == b.regionPartitionIds.size());
  for (size_t i = 0; i < a.regionPartitionIds.size(); ++i) {
    assert(isVectorEqual(a.regionPartitionIds[i], b.regionPartitionIds[i]));
  }

  return true;
}


} // namespace toucanGPUSim