// Host-only regression: build --target check-micro-part-merging, then run it.
#include "toucan/MultiRegionMicroPartScheduler.h"
#include "llvm/Support/ErrorHandling.h"
#include "llvm/Support/raw_ostream.h"
#include <algorithm>
#include <array>
#include <map>
#include <random>
#include <vector>

using namespace toucan;

namespace {
void require(bool condition, const char *message) {
  if (!condition)
    llvm::report_fatal_error(message);
}

// Exact expression interning: NOP is identity, and other LUTs are arbitrary
// pure functions. Equal output expressions imply equal values for all inputs,
// without relying on random input samples or expression-hash equality.
struct Expressions {
  std::map<std::array<uint32_t, 4>, uint32_t> nodes;

  uint32_t apply(LUTOpName op, uint32_t a, uint32_t b, uint32_t c) {
    if (op == LUTOpName::LUT_Nop)
      return c;
    auto key = std::array<uint32_t, 4>{static_cast<uint32_t>(op), a, b, c};
    return nodes.emplace(key, 65536 + nodes.size()).first->second;
  }

  std::map<uint16_t, uint32_t> evaluate(const std::vector<CGMicroPartInfo> &parts) {
    std::map<uint16_t, uint32_t> outputs;
    for (const auto &part : parts) {
      require(part.opType == CGToucanOPName::LUT, "Expected regular LUT micro-part");
      require(part.maxActiveVars > 0 && part.maxActiveVars <= 32, "Invalid maximum lane count");
      auto checkWidth = [&](size_t width) {
        require(width <= static_cast<size_t>(part.maxActiveVars), "Level exceeds maximum lane count");
      };
      checkWidth(part.topLevel.size());
      checkWidth(part.lastLevel.size());
      std::vector<uint32_t> lanes;
      for (const auto &op : part.topLevel)
        lanes.push_back(apply(op.opName, op.op0, op.op1, op.op2));
      auto select = [&](uint32_t selector) {
        if (selector < 16)
          return selector;
        require(selector >= 32 && selector < 32 + lanes.size(), "Invalid shuffle selector");
        return lanes[selector - 32];
      };
      for (const auto &level : part.middleLevels) {
        checkWidth(level.size());
        std::vector<uint32_t> next;
        for (const auto &op : level)
          next.push_back(apply(op.opName, select(op.op0), select(op.op1), select(op.op2)));
        lanes = std::move(next);
      }
      for (const auto &write : part.lastLevel) {
        require(write.result >= 16, "Write to constant slot");
        require(outputs.emplace(write.result, select(write.shuffleId)).second, "Duplicate output slot");
      }
    }
    return outputs;
  }
};

uint64_t countNOPs(const std::vector<CGMicroPartInfo> &parts) {
  uint64_t count = 0;
  for (const auto &part : parts) {
    for (const auto &op : part.topLevel)
      count += op.opName == LUTOpName::LUT_Nop;
    for (const auto &level : part.middleLevels)
      for (const auto &op : level)
        count += op.opName == LUTOpName::LUT_Nop;
  }
  return count;
}

std::vector<CGMicroPartInfo> checkMerge(const std::vector<CGMicroPartInfo> &before) {
  Expressions expressions;
  const auto expected = expressions.evaluate(before);
  auto after = before;
  MultiRegionMicroPartScheduler::mergeSmallRegularMParts(after);
  require(expressions.evaluate(after) == expected, "Merging changed an output expression");
  require(after.size() <= before.size(), "Merging added micro-parts");
  return after;
}

CGMicroPartInfo makePart(unsigned id, unsigned width, unsigned depth) {
  CGMicroPartInfo part;
  part.opType = CGToucanOPName::LUT;
  part.maxActiveVars = width;
  for (unsigned lane = 0; lane < width; ++lane) {
    auto source = static_cast<uint16_t>(16 + id * 32 + lane);
    part.topLevel.push_back({LUTOpName::LUT_Xor, 0, source, static_cast<uint16_t>(lane % 16)});
  }
  for (unsigned level = 0; level < depth; ++level) {
    part.middleLevels.emplace_back();
    for (unsigned lane = 0; lane < width; ++lane)
      part.middleLevels.back().push_back(
          {LUTOpName::LUT_Xor, 0, static_cast<uint8_t>(32 + lane), static_cast<uint8_t>(32 + (lane + 1) % width)});
  }
  for (unsigned lane = 0; lane < width; ++lane)
    part.lastLevel.push_back({static_cast<uint8_t>(32 + lane), static_cast<uint16_t>(40000 + id * 32 + lane)});
  return part;
}

CGMicroPartInfo randomPart(unsigned id, std::mt19937 &random) {
  unsigned capacity = 1 + random() % 32;
  unsigned width = 1 + random() % capacity;
  auto part = makePart(id, width, 0);
  part.lastLevel.clear();
  unsigned depth = random() % 20;
  auto selector = [&]() -> uint8_t { return random() % 3 == 0 ? random() % 16 : 32 + random() % width; };
  for (unsigned level = 0; level < depth; ++level) {
    part.middleLevels.emplace_back();
    unsigned nextWidth = 1 + random() % capacity;
    for (unsigned lane = 0; lane < nextWidth; ++lane)
      part.middleLevels.back().push_back({static_cast<LUTOpName>(random() % 20), selector(), selector(), selector()});
    width = nextWidth;
    part.maxActiveVars = std::max(part.maxActiveVars, static_cast<int>(width));
  }
  unsigned writes = 1 + random() % capacity;
  for (unsigned lane = 0; lane < writes; ++lane)
    part.lastLevel.push_back({selector(), static_cast<uint16_t>(40000 + id * 32 + lane)});
  part.maxActiveVars = std::max(part.maxActiveVars, static_cast<int>(writes));
  return part;
}
} // namespace

int main() {
  require(checkMerge({}).empty(), "Empty execution level changed");
  require(checkMerge({makePart(0, 32, 2)}).size() == 1, "Full warp changed");
  require(checkMerge({makePart(0, 17, 3), makePart(1, 16, 3)}).size() == 2, "Overfilled a warp");

  // Width-first pairing used to match each deep part with a shallow one,
  // adding 320 carry NOPs. Equal-depth pairs need no padding and still use two warps.
  std::vector<CGMicroPartInfo> mixed = {makePart(0, 16, 10), makePart(1, 16, 10), makePart(2, 16, 0),
                                        makePart(3, 16, 0)};
  auto packed = checkMerge(mixed);
  require(packed.size() == 2 && countNOPs(packed) == 0, "Missed compatible equal-depth partners");
  require(packed[0].middleLevels.size() == 10 && packed[1].middleLevels.empty(), "Unexpected packed depths");

  // Unequal depths exercise padding and repeated lane remapping. The shallow
  // part writes a constant and the same lane twice as well as a distinct lane.
  auto shallow = makePart(1, 3, 0);
  shallow.lastLevel[0].shuffleId = 15;
  shallow.lastLevel[2].shuffleId = 33;
  auto unequal = checkMerge({makePart(0, 4, 5), shallow, makePart(2, 2, 2)});
  require(unequal.size() == 1, "Compatible unequal-depth parts were not merged");

  std::vector<CGMicroPartInfo> narrow;
  for (unsigned i = 0; i < 64; ++i)
    narrow.push_back(makePart(i, 1, 4));
  require(checkMerge(narrow).size() == 2, "Repeated merges failed to fill two warps");

  // Vary lane counts by level, with constants, duplicate selectors, all LUT
  // opcodes, and zero-depth parts. The seed fixes the generated schedules;
  // expression equality still covers every possible input to each schedule.
  std::mt19937 random(0x746f7563);
  for (unsigned trial = 0; trial < 500; ++trial) {
    std::vector<CGMicroPartInfo> parts;
    unsigned count = 1 + random() % 128;
    for (unsigned i = 0; i < count; ++i)
      parts.push_back(randomPart(i, random));
    checkMerge(parts);
  }
  llvm::outs() << "PASS: depth-aware packing, lane limits, padding, constants, and 500 symbolic equivalence cases\n";
  return 0;
}
