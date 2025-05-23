#include "toucan/MicroPartLocalValueAllocator.h"
#include "circt/Dialect/HW/HWTypes.h"
#include "circt/Support/LLVM.h"
#include "circt/Dialect/Comb/CombDialect.h"
#include "circt/Dialect/Comb/CombOps.h"
#include "circt/Dialect/Seq/SeqOps.h"

#include "mlir/IR/Operation.h"
#include "mlir/IR/PatternMatch.h"
#include "mlir/IR/Value.h"
#include "mlir/Pass/AnalysisManager.h"
#include "mlir/Support/LLVM.h"
#include "mlir/IR/Builders.h"

#include "toucan/MicroPartitioner.h"
#include "toucan/PartitioningGraph.h"
#include "toucan/ToucanAnalysis.h"
#include "toucan/MicroPartLocalValueAllocator.h"
#include "toucan/ToucanOps.h"
#include "toucan/ToucanTypes.h"
#include "toucan/ToucanUtils.h"

#include "llvm/ADT/BitVector.h"
#include "llvm/ADT/STLExtras.h"
#include "llvm/ADT/SmallVector.h"
#include "llvm/Support/Casting.h"
#include "llvm/Support/Debug.h"
#include "llvm/Support/ErrorHandling.h"
#include "llvm/Support/raw_ostream.h"
#include <algorithm>
#include <cstddef>
#include <cstdint>

#include <set>

using namespace toucan;

using namespace mlir;
using namespace llvm;
using namespace circt;


// #define DEBUG_PRINT_LOCAL_VAL_ALLOC



void MicroPartLocalValueAllocator::allocateLocalValues() {

  assert(valToLifeTime.size() != 0);

  mlir::DenseSet<mlir::Value> unPinnedVals;
  unPinnedVals.reserve(valToLifeTime.size());

  mlir::SmallVector<mlir::SmallVector<mlir::Value>> lifeTimeStartToVal, lifeTimeEndToVal;
  lifeTimeStartToVal.resize(totalLevels + 1);
  lifeTimeEndToVal.resize(totalLevels + 1);

  // Ignore pre-allocated const and io vals
  for (auto [eachVal, lifeTime]: valToLifeTime) {

    bool isPinnedVal = pinnedInputVals.contains(eachVal) || pinnedOutputVals.contains(eachVal) || constVals.contains(eachVal);
    assert(isPinnedVal == valToValId.contains(eachVal));
    if (!isPinnedVal) {
      unPinnedVals.insert(eachVal);
    }

    // group vals by life time
    assert(lifeTime.start <= totalLevels);
    assert(lifeTime.end <= totalLevels);
    assert(lifeTime.start < lifeTime.end);

    lifeTimeStartToVal[lifeTime.start].push_back(eachVal);
    lifeTimeEndToVal[lifeTime.end].push_back(eachVal);
  }


  std::set<uint32_t> availableValIds;
  mlir::DenseMap<uint32_t, uint32_t> pinnedOutputValIdToNextOccupy;
  size_t nextAvailableValId = compactConstValPool.size();
  
  auto valOKToUse = [&](uint32_t freeValId, uint32_t valEndTime) {
    if (pinnedOutputValIdToNextOccupy.contains(freeValId)) {
      // this val is used by a pinned val in the future
      return (valEndTime < pinnedOutputValIdToNextOccupy[freeValId]);
    }
    // not pinned. Simply use it
    return true;
  };

  // for (auto eachVal: pinnedOutputVals) {
  //   assert(valToValId.contains(eachVal));
  // }

  // const vals and pinned vals (reg read, reg write)
  for (auto [eachVal, eachValId]: valToValId) {
    // pinned vals
    auto valLifeTime = valToLifeTime[eachVal];

    assert(!pinnedOutputValIdToNextOccupy.contains(eachValId));

    assert(valLifeTime.end <= totalLevels);
    if (pinnedOutputVals.contains(eachVal)) {
      // last level output vals
      pinnedOutputValIdToNextOccupy[eachValId] = valLifeTime.start;
    } else {
      // must be an input val or const
      assert(valLifeTime.start == 0);
    }
    nextAvailableValId = std::max(nextAvailableValId, static_cast<size_t>(eachValId));
  }
  assert(numOutputVals == pinnedOutputVals.size());
  assert(pinnedOutputValIdToNextOccupy.size() == numOutputVals);
  nextAvailableValId++;

  // pinned vals may be used later, as long as its value is released
  for (auto [_, valId]: valToValId) {
    // pinned vals
    if (valId >= numConsts) {
      assert(!availableValIds.contains(valId));
      availableValIds.insert(valId);
    }
  }

  for (size_t levelId = 0; levelId < totalLevels; levelId++) {
    auto &valsToAllocate = lifeTimeStartToVal[levelId];
    auto &valsToRelease = lifeTimeEndToVal[levelId];

    // sort in descending order of val life time ends
    // allocate long-lasting vals first
    std::sort(valsToAllocate.begin(), valsToAllocate.end(), [&](const mlir::Value &a, const mlir::Value &b) {
      auto a_end = valToLifeTime[a].end;
      auto b_end = valToLifeTime[b].end;
      return a_end > b_end;
    });

    // Move vec vals to first
    std::stable_sort(valsToAllocate.begin(), valsToAllocate.end(), [&](const mlir::Value &a, const mlir::Value &b) {
      auto a_is_vec = vecValToLength.contains(a);
      auto b_is_vec = vecValToLength.contains(b);
      return a_is_vec && (!b_is_vec);
    });

    // uint32_t pinnedCount = 0;

    // allocate for all values that starts active in this level
    for (auto eachVal: valsToAllocate) {
      if (unPinnedVals.contains(eachVal)) {
        assert(!valToValId.contains(eachVal));

        // Not pinned
        if (vecValToLength.contains(eachVal)) {
          assert(isa<toucan::DefVectorOp>(eachVal.getDefiningOp()));
          // a vector
          auto vecLength = vecValToLength[eachVal];
          auto vecValEndTime = valToLifeTime[eachVal].end;

          mlir::SmallVector<bool> candidateValIds;
          mlir::SmallVector<uint32_t> availableValIds_vec;
          candidateValIds.reserve(availableValIds.size());
          availableValIds_vec.reserve(availableValIds.size());
          for (auto freeValId: availableValIds) {
            bool valIsAvailableForVec = valOKToUse(freeValId, vecValEndTime);
            candidateValIds.push_back(valIsAvailableForVec);
            availableValIds_vec.push_back(freeValId);
          }

          // we need continuous [vecLength] vals.
          uint32_t continuousValSize = 0;
          uint32_t continuousValStartPos = 0;
          uint32_t continuousValStartId = 0;

          for (size_t i =0; i < candidateValIds.size(); i++) {
            if (continuousValSize == 0) {
              if (candidateValIds[i]) {
                // start of a new section
                continuousValStartId = availableValIds_vec[i];
                continuousValStartPos = i;
                continuousValSize = 1;
              }
            } else {
              // in the middle of a region
              if (candidateValIds[i] && availableValIds_vec[i] == continuousValStartId + 1) {
                continuousValSize++;
              } else {
                if (candidateValIds[i]) {
                  // start of a new section
                  continuousValStartId = availableValIds_vec[i];
                  continuousValStartPos = i;
                  continuousValSize = 1;
                } else {
                  continuousValSize = 0;
                }
              }
            }

            if (continuousValSize == vecLength) break;
          }


          assert(!valToValId.contains(eachVal));

          if (continuousValSize == vecLength) {
#ifdef DEBUG_PRINT_LOCAL_VAL_ALLOC
          llvm::dbgs() << "Alloc vec val of size " << continuousValSize << ", start from " << continuousValStartId << "\n";
#endif
            valToValId[eachVal] = continuousValStartId;

            for (uint32_t i = 0; i < vecLength; i++) {
              auto valId = continuousValStartId + i;
              assert(availableValIds_vec[continuousValStartPos + i] == valId);
              availableValIds.erase(valId);
            }
          } else {
            // fail to find sufficient space for entire vector. use some new vars
            uint32_t vecValIdStart = nextAvailableValId;
            valToValId[eachVal] = vecValIdStart;

            nextAvailableValId += vecLength;
#ifdef DEBUG_PRINT_LOCAL_VAL_ALLOC
            llvm::dbgs() << "Alloc " << vecLength << " new vals for a vector\n";
#endif
          }

        } else {
          assert(!isa<toucan::DefVectorOp>(eachVal.getDefiningOp()));
          uint32_t valId = 0;
          bool foundFreeSlot = false;

          for (auto freeValId: availableValIds) {
            auto valEndTime = valToLifeTime[eachVal].end;
            if (valOKToUse(freeValId, valEndTime)) {
              foundFreeSlot = true;
              valId = freeValId;
              break;
            }
          }

          if (!foundFreeSlot) {
            // we iterate all free vals but cannot fit.
            // ask for a new val id
            valId = nextAvailableValId;
            nextAvailableValId++;

          } else {
            // remove this val from free list
            availableValIds.erase(valId);
          }

#ifdef DEBUG_PRINT_LOCAL_VAL_ALLOC
          // llvm::dbgs() << "Alloc val " << valId << "\n";
#endif
          assert(valId != 0);
          assert(!valToValId.contains(eachVal));
          valToValId[eachVal] = valId;
        }


      } else {
        // Pinned val.
        auto pinnedValIds = valToValId[eachVal];
        assert(availableValIds.contains(pinnedValIds));
        availableValIds.erase(pinnedValIds);
        assert(!vecValToLength.contains(eachVal));
      }
    }


    // Release val ids that no longer used
    for (auto eachVal: valsToRelease) {
      assert(valToValId.contains(eachVal));

      if (vecValToLength.contains(eachVal)) {
        // a vector
        auto vecLength = vecValToLength[eachVal];
        auto vecValId = valToValId[eachVal];
        for (uint32_t valId = vecValId; valId < (vecValId + vecLength); valId++) {
          availableValIds.insert(valId);
        }
      } else {
        auto valIdToRelease = valToValId[eachVal];
        availableValIds.insert(valIdToRelease);
        assert(valIdToRelease >= numConsts);
      }

    }
  }

  numTotalValSize = nextAvailableValId;
}


void MicroPartLocalValueAllocator::populateInitialPinnedVals(const PartitioningGraph &graph, const mlir::DenseMap<mlir::Value, uint32_t> constValToRawValue, const MicroPartitioner &mpartitioner) {


  // In each value pool, populate all possible const values (4 bits, 0 to 15)
  compactConstValPool.clear();
  compactConstValPool.reserve(16);
  for (int i = 0; i < 16; i++) {
    compactConstValPool.push_back(i);
  }

  for (auto [eachConstVal, rawVal]: constValToRawValue) {
    assert(rawVal < 16);
    // for const vals, valId is the same as rawVal
    uint32_t valId = rawVal;
    valToValId[eachConstVal] = valId;
    constVals.insert(eachConstVal);
  }

  numConsts = compactConstValPool.size();
  numOutputVals = 0;
  numInputVals = 0;

  uint32_t nextValId = numConsts;


  // allocate for reg write. They should be placed at the begining to ensure performance
  for (const auto eachVtx: mpartitioner.allRegWrites) {
    assert(graph[eachVtx].toucanOpName == CGToucanOPName::RegWrite);
    assert(graph[eachVtx].op != nullptr);
    auto regWriteOp = cast<toucan::RegWriteOp>(graph[eachVtx].op);
    auto outputVal = regWriteOp.getData();

    if (!pinnedOutputVals.contains(outputVal)) {
      pinnedOutputVals.insert(outputVal);
      // insert this val
      auto realValId = nextValId;
      numOutputVals++;
      // Vector val cannot be directly used as partition output val
      assert(!vecValToLength.contains(outputVal));
      assert(!valToValId.contains(outputVal));
      valToValId[outputVal] = realValId;

      nextValId++;
    } else {
      llvm_unreachable("Should this happen?");
    }
  }

  assert(numOutputVals == pinnedOutputVals.size());


  // allocate reg reads
  for (const auto eachVtx: mpartitioner.allRegReads) {
    assert(graph[eachVtx].toucanOpName == CGToucanOPName::RegRead);
    assert(graph[eachVtx].op != nullptr);
    auto regReadOp = cast<toucan::RegReadOp>(graph[eachVtx].op);
    auto inputVal = regReadOp.getResult();

    assert(!vecValToLength.contains(inputVal));
    assert(!pinnedOutputVals.contains(inputVal));
    assert(!constVals.contains(inputVal));
    pinnedInputVals.insert(inputVal);
    auto realValId = nextValId;
    numInputVals++;

    assert(!valToValId.contains(inputVal));
    valToValId[inputVal] = realValId;
    nextValId++;
  }

  assert(numInputVals == pinnedInputVals.size());
}




void MicroPartLocalValueAllocator::collectValueLifetime(const PartitioningGraph &graph, const MicroPartitioner &mpartitioner) {
  // 1. For every reg read, result val life starts at 0
  for (auto eachVtx: mpartitioner.allRegReads) {
    auto regReadOp = cast<toucan::RegReadOp>(graph[eachVtx].op);
    auto resultVal = regReadOp.getResult();

    assert(!valToLifeTime.contains(resultVal));
    valToLifeTime[resultVal] = {0, 0};
  }

  
  for (uint32_t levelId = 0; levelId < mpartitioner.partLevels.size(); levelId++) {
    for (const auto &eachMPart: mpartitioner.partLevels[levelId]) {
      for (const auto &eachInVal: eachMPart.inputValues) {
        // update val end time accordingly
        if (!isa<toucan::DefConstVectorOp>(eachInVal.getDefiningOp())) {
          // ignore const vec. They are placed in constVecPool and always alive
          assert(valToLifeTime.contains(eachInVal));
          auto oldValEndTime = valToLifeTime[eachInVal].end;
          assert(oldValEndTime <= levelId);
          valToLifeTime[eachInVal].end = levelId;
        }
      }
      for (const auto &eachOutVal: eachMPart.outputValues) {
        // update val start time accordingly
        if (!isa<toucan::DefConstVectorOp>(eachOutVal.getDefiningOp())) {
          // ignore const vec. They are placed in constVecPool and always alive
          assert(!valToLifeTime.contains(eachOutVal));
          valToLifeTime[eachOutVal] = {levelId, levelId};

          if (isa<mlir::TypedValue<toucan::VecType>>(eachOutVal)) {
            // a vector
            auto vecLength = cast<mlir::TypedValue<toucan::VecType>>(eachOutVal).getType().getLength();
            if (vecValToLength.contains(eachOutVal)) {
              assert(vecValToLength[eachOutVal] == vecLength);
            } else {
              vecValToLength[eachOutVal] = vecLength;
            }
          }
        }
      }
    }
  }

  totalLevels = mpartitioner.partLevels.size();

  // regwrite, memwrite, stop, print survive to last level (totalLevels)
  for (auto eachVtx: mpartitioner.allRegWrites) {
    auto regWriteOp = cast<toucan::RegWriteOp>(graph[eachVtx].op);
    auto inputVal = regWriteOp.getData();

    assert(valToLifeTime.contains(inputVal));
    valToLifeTime[inputVal].end = totalLevels;
  }

  // mem write could be merged nodes
  // collect all of them
  mlir::SmallVector<toucan::MemWriteOp> mwOps;
  for (auto eachVtx: mpartitioner.allMemWrites) {
    auto opCount = graph[eachVtx].opCount;
    uint32_t realOpCount = 0;
    auto oneMWOp = cast<toucan::MemWriteOp>(graph[eachVtx].op);
    auto memVal = oneMWOp.getMem();

    for (auto eachUserOp: memVal.getUsers()) {
      if (auto mwOp = dyn_cast<toucan::MemWriteOp>(eachUserOp)) {
        realOpCount += 1;
        mwOps.push_back(mwOp);
      }
    }
  }
  
  for (auto mwOp: mwOps) {
    auto addrVecVal = mwOp.getAddrVec();
    auto enVal = mwOp.getEn();
    auto dataVal = mwOp.getData();

    if (!isa<toucan::ConstantOp>(enVal.getDefiningOp())) {
      assert(valToLifeTime.contains(enVal));
      valToLifeTime[enVal].end = totalLevels;
    }
    if (!isa<toucan::ConstantOp>(dataVal.getDefiningOp())) {
      assert(valToLifeTime.contains(dataVal));
      valToLifeTime[dataVal].end = totalLevels;
    }
    if (!isa<toucan::DefConstVectorOp>(addrVecVal.getDefiningOp())) {
      assert(valToLifeTime.contains(addrVecVal));
      valToLifeTime[addrVecVal].end = totalLevels;
    }
  }

  // stop
  for (auto eachVtx: mpartitioner.allStops) {
    auto stopOp = cast<toucan::StopOp>(graph[eachVtx].op);
    auto inputVal = stopOp.getEn();

    assert(valToLifeTime.contains(inputVal));
    valToLifeTime[inputVal].end = totalLevels;
  }

  // print
  for (auto eachVtx: mpartitioner.allPrints) {
    auto printOp = cast<toucan::PrintOp>(graph[eachVtx].op);
    auto inputVal = printOp.getEn();

    assert(valToLifeTime.contains(inputVal));
    valToLifeTime[inputVal].end = totalLevels;
  }

  
}
