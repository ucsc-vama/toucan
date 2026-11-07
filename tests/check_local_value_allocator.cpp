// Host-only regression: build --target check-local-value-allocator, then run it.
#include "mlir/IR/Builders.h"
#include "mlir/IR/BuiltinOps.h"
#include "mlir/Parser/Parser.h"
#include "toucan/MicroPartLocalValueAllocator.h"
#include "llvm/Support/raw_ostream.h"
#include <memory>
#include <vector>

using namespace mlir;
using namespace toucan;

namespace {
struct LiveVector {
  Value value;
  unsigned length, start, end;
};

struct Fixture {
  MLIRContext context;
  OwningOpRef<ModuleOp> module;
  RepCutPartitionCodeGenData data;
  std::vector<LiveVector> live;
  MicroPartLocalValueAllocator allocator;

  Fixture(unsigned levels) {
    context.getOrLoadDialect<ToucanDialect>();
    module = parseSourceString<ModuleOp>("module { %c = toucan.const 0 : i4 : i4 }", &context);
    data.mpartLevels.resize(levels);
  }

  Value vector(unsigned length, unsigned start, unsigned end) {
    OpBuilder builder(&context);
    builder.setInsertionPointToEnd(module->getBody());
    SmallVector<Value> inputs(length, module->getBody()->front().getResult(0));
    auto value = builder.create<DefVectorOp>(module->getLoc(), inputs).getHandle();
    live.push_back({value, length, start, end});
    return value;
  }

  void step(unsigned level, ArrayRef<Value> inputs, ArrayRef<Value> outputs) {
    auto part = std::make_shared<MicroPart>();
    part->inputValues.insert(inputs.begin(), inputs.end());
    part->outputValueSet.insert(outputs.begin(), outputs.end());
    data.mpartLevels[level].push_back(std::move(part));
  }

  bool allocate() {
    allocator.collectValueLifetime(data);
    DenseMap<Value, uint32_t> constants;
    allocator.populateInitialPinnedVals(data, constants);
    allocator.allocateLocalValues();
    // Independent interval check: inputs and outputs of the same level may
    // execute in different warps, so even touching lifetimes must not overlap.
    for (const auto &a : live) {
      if (!allocator.valToValId.contains(a.value))
        return false;
      unsigned aStart = allocator.valToValId.at(a.value);
      if (aStart < 16 || aStart + a.length > allocator.numTotalValSize)
        return false;
      for (const auto &b : live) {
        if (a.value == b.value || a.end < b.start || b.end < a.start)
          continue;
        unsigned bStart = allocator.valToValId.at(b.value);
        if (aStart < bStart + b.length && bStart < aStart + a.length)
          return false;
      }
    }
    return true;
  }

  unsigned slot(Value value) { return allocator.valToValId.at(value); }
};

bool reuseReleasedInput(unsigned length) {
  Fixture f(5);
  auto input = f.vector(length, 0, 0);
  auto a = f.vector(length, 1, 2);
  auto b = f.vector(length, 2, 3);
  auto c = f.vector(length, 3, 4);
  f.data.allExgReadVals.push_back(input);
  f.step(0, {input}, {});
  f.step(1, {}, {a});
  f.step(2, {a}, {b});
  f.step(3, {b}, {c});
  f.step(4, {c}, {});
  return f.allocate() && f.slot(a) == f.slot(input) && f.slot(c) == f.slot(input) &&
         f.allocator.numTotalValSize <= 17 + 2 * length;
}

bool respectFutureOutput(unsigned temporaryEnd) {
  Fixture f(6);
  auto output = f.vector(8, 3, 6);
  auto temporary = f.vector(8, 0, temporaryEnd);
  f.data.allExgWriteVals.push_back(output);
  f.step(0, {}, {temporary});
  f.step(temporaryEnd, {temporary}, {});
  f.step(3, {}, {output});
  return f.allocate() && f.slot(output) == 16 &&
         ((f.slot(temporary) == f.slot(output)) == (temporaryEnd < 3));
}

bool skipFragmentedSpace() {
  Fixture f(4);
  auto first = f.vector(3, 0, 0);
  auto middle = f.vector(2, 0, 3);
  auto last = f.vector(3, 0, 0);
  auto temporary = f.vector(4, 1, 2);
  f.data.allExgReadVals = {first, middle, last};
  f.step(0, {first, last}, {});
  f.step(1, {}, {temporary});
  f.step(2, {temporary}, {});
  f.step(3, {middle}, {});
  return f.allocate() && f.slot(temporary) >= f.slot(last) + 3;
}

bool restartAfterIneligibleSlot() {
  Fixture f(5);
  auto first = f.vector(3, 4, 5);
  auto middle = f.vector(1, 1, 5);
  auto last = f.vector(4, 4, 5);
  auto temporary = f.vector(4, 0, 2);
  f.data.allExgWriteVals = {first, middle, last};
  f.step(0, {}, {temporary});
  f.step(1, {}, {middle});
  f.step(2, {temporary}, {});
  f.step(4, {}, {first, last});
  return f.allocate() && f.slot(temporary) == f.slot(last);
}
} // namespace

int main() {
  for (unsigned length : {1u, 2u, 3u, 8u, 16u, 32u}) {
    if (!reuseReleasedInput(length)) {
      llvm::errs() << "FAIL: reuse released vector of length " << length << '\n';
      return 1;
    }
  }
  for (unsigned end : {2u, 3u, 4u}) {
    if (!respectFutureOutput(end)) {
      llvm::errs() << "FAIL: future output reservation with temporary end " << end << '\n';
      return 1;
    }
  }
  if (!skipFragmentedSpace() || !restartAfterIneligibleSlot()) {
    llvm::errs() << "FAIL: fragmented or ineligible free slots\n";
    return 1;
  }
  llvm::outs() << "PASS: vector reuse, same-level liveness, future outputs, and fragmented free slots\n";
  return 0;
}
