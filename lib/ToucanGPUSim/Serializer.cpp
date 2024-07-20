#include "ToucanGPUSim/ToucanGPUGenDataTypes.h"
#include <fstream>

#include <iostream>

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




template<typename K, typename V>
static void serializeMap(std::ostream& out, const std::unordered_map<K, std::vector<V>>& map) {
  size_t mapSize = map.size();
  serializePrimitive(out, mapSize);
  for (const auto& pair : map) {
    serializeString(out, pair.first);
    serializeVector(out, pair.second); // Assumes V can be serialized directly or is already handled
  }
}

template<typename K, typename V>
static void deserializeMap(std::istream& in, std::unordered_map<K, std::vector<V>>& map) {
  size_t mapSize;
  deserializePrimitive(in, mapSize);
  for (size_t i = 0; i < mapSize; ++i) {
    K key;
    deserializeString(in, key); // Assumes K is string
    std::vector<V> vector;
    deserializeVector(in, vector); // Assumes V can be deserialized directly or is already handled
    map[key] = vector;
  }
}





static void serializeSimPartitionInfo(std::ostream& out, const toucanGPUSim::SimPartitionInfo& info) {
  serializeVector(out, info.valuePool);
  out.write(reinterpret_cast<const char*>(&info.valuePoolSize), sizeof(info.valuePoolSize));

  serializeVector(out, info.ops_l0);
  size_t execSize = info.ops_exec.size();
  out.write(reinterpret_cast<const char*>(&execSize), sizeof(execSize));
  for (const auto& execVec : info.ops_exec) {
    serializeVector(out, execVec);
  }

  serializeVector(out, info.ops_last);
  serializeVector(out, info.opInfo_exec);
  serializePrimitive(out, info.opInfo_last);
}

static void deserializeSimPartitionInfo(std::istream& in, toucanGPUSim::SimPartitionInfo& info) {
  deserializeVector(in, info.valuePool);
  in.read(reinterpret_cast<char*>(&info.valuePoolSize), sizeof(info.valuePoolSize));

  deserializeVector(in, info.ops_l0);
  size_t execSize;
  in.read(reinterpret_cast<char*>(&execSize), sizeof(execSize));
  info.ops_exec.resize(execSize);
  for (auto& execVec : info.ops_exec) {
    deserializeVector(in, execVec);
  }

  deserializeVector(in, info.ops_last);
  deserializeVector(in, info.opInfo_exec);
  deserializePrimitive(in, info.opInfo_last);
}



void toucanGPUSim::serializeSimDesignInfo(std::ostream& out, const toucanGPUSim::SimDesignInfo& info) {
  // Serialize lut and lutIndex vectors
  serializeVector(out, info.lut);

  // regPool and memPool vectors need randomization.

  // Serialize regPoolSize and memPoolSize
  serializePrimitive(out, info.regPoolSize);
  serializePrimitive(out, info.memPoolSize);
  serializePrimitive(out, info.exchangePoolSize);

  // Serialize parts (a vector of SimPartitionInfo)
  serializePrimitive(out, info.parts.size());
  for (const auto& part : info.parts) {
    // Serialize SimPartitionInfo members...
    serializeSimPartitionInfo(out, part);
  }

  serializePrimitive(out, info.regionPartitionIds.size());
  for (const auto& region: info.regionPartitionIds) {
    serializeVector(out, region);
  }

  // Serialize printMsgs (a vector of strings)
  
  serializePrimitive(out, info.printMsgs.size());
  for (const auto& msg : info.printMsgs) {
    serializeString(out, msg);
  }
}

void toucanGPUSim::deserializeSimDesignInfo(std::istream& in, toucanGPUSim::SimDesignInfo& info) {
  // Deserialize lut and lutIndex vectors
  deserializeVector(in, info.lut);

  // regPool and memPool vectors need randomization.

  // Deserialize regPoolSize and memPoolSize
  deserializePrimitive(in, info.regPoolSize);
  deserializePrimitive(in, info.memPoolSize);
  deserializePrimitive(in, info.exchangePoolSize);

  // Deserialize parts
  size_t partsSize;
  deserializePrimitive(in, partsSize);
  info.parts.resize(partsSize);
  for (auto& part : info.parts) {
    // Deserialize SimPartitionInfo members...
    deserializeSimPartitionInfo(in, part);
  }

  size_t regionSize;
  deserializePrimitive(in, regionSize);
  info.regionPartitionIds.resize(regionSize);
  for (auto& region: info.regionPartitionIds) {
    deserializeVector(in, region);
  }

  // Deserialize printMsgs
  size_t printMsgsSize;
  deserializePrimitive(in, printMsgsSize);
  info.printMsgs.resize(printMsgsSize);
  for (auto& msg : info.printMsgs) {
    deserializeString(in, msg);
  }
}





void toucanGPUSim::serializeSimDebugInfo(std::ostream& out, const toucanGPUSim::SimDebugInfo& info) {
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


void toucanGPUSim::deserializeSimDebugInfo(std::istream& in, toucanGPUSim::SimDebugInfo& info) {
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


