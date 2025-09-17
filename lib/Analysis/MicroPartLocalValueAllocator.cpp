#include "toucan/MicroPartLocalValueAllocator.h"
#include "circt/Dialect/HW/HWTypes.h"
#include "circt/Support/LLVM.h"
#include "circt/Dialect/Comb/CombDialect.h"
#include "circt/Dialect/Comb/CombOps.h"
#include "circt/Dialect/Seq/SeqOps.h"

#include "mlir/IR/Operation.h"
#include "mlir/IR/PatternMatch.h"
#include "mlir/IR/Types.h"
#include "mlir/IR/Value.h"
#include "mlir/Pass/AnalysisManager.h"
#include "mlir/Support/LLVM.h"
#include "mlir/IR/Builders.h"

#include "toucan/MicroPartitioner.h"
#include "toucan/PartitioningGraph.h"
#include "toucan/ToucanAnalysis.h"
#include "toucan/MicroPartLocalValueAllocator.h"
#include "toucan/ToucanCodeGenInfo.h"
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



void MicroPartLocalValueAllocator::allocateLocalValuesWithoutReclaim() {

  assert(valToLifeTime.size() != 0);

  mlir::DenseSet<mlir::Value> unPinnedVals;
  unPinnedVals.reserve(valToLifeTime.size());


  uint32_t nextAvailableValId = 0;
  for (const auto &[eachVal, valId]: valToValId) {
    nextAvailableValId = std::max(nextAvailableValId, valId);
  }
  nextAvailableValId += 1;

  for (auto [eachVal, lifeTime]: valToLifeTime) {

    bool isPinnedVal = pinnedInputVals.contains(eachVal) || pinnedOutputVals.contains(eachVal) || constVals.contains(eachVal);
    assert(isPinnedVal == valToValId.contains(eachVal));
    if (!isPinnedVal) {
      unPinnedVals.insert(eachVal);
    }

    // group vals by life time
    assert(lifeTime.start <= totalLevels);
    assert(lifeTime.end <= totalLevels);
    if (isa<toucan::VectorReadOp>(eachVal.getDefiningOp())
      || isa<toucan::StaticVectorSegmentReadOp>(eachVal.getDefiningOp())) {
      // Result of VecRead & StaticVectorSegmentRead might not be used due to replication by RepCut
      // So here we allow lifeTime.start == lifeTime.end
      assert(lifeTime.start <= lifeTime.end);
    } else {
      // Otherwise it must be used
      assert(lifeTime.start < lifeTime.end);
    }

    if (!vecSegmentsToVecArith.contains(eachVal)) {
      if (!valToValId.contains(eachVal)) {
        // allocate, but dont recycle
        if (vecValToLength.contains(eachVal)) {
          // a vector
          auto vecLength = vecValToLength.at(eachVal);
          uint32_t vecValIdStart = nextAvailableValId;
          valToValId[eachVal] = vecValIdStart;

          nextAvailableValId += vecLength;
        } else {
          // regular val
          valToValId[eachVal] = nextAvailableValId;
          nextAvailableValId++;
        }
      }
    }

    activeValuesAtLast.insert(eachVal);
  }



  // Assign ID for segment values
  for (const auto &[eachVecVal, segmentVals]: vecArithResultToSegments) {
    assert(valToValId.contains(eachVecVal) && "Result vector of VecArith should be already allocated!");
    auto vecValId = valToValId[eachVecVal];
    auto vecLength = vecValToLength[eachVecVal];

    for (const auto &segmentVal: segmentVals) {
      assert(valToLifeTime.contains(segmentVal));
      if (valToValId.contains(segmentVal)) {
        dbgs() << "Value that should not appear! defined by:\n";
        segmentVal.getDefiningOp()->print(dbgs());
        dbgs() << "\n";
      }
      assert(!valToValId.contains(segmentVal));
      auto segReadOp = cast<toucan::StaticVectorSegmentReadOp>(segmentVal.getDefiningOp());
      assert(segReadOp.getHandle() == eachVecVal);

      auto segmentId = segReadOp.getSegmentId().getZExtValue();
      assert(segmentId < vecLength);

      valToValId[segmentVal] = vecValId + segmentId;
    }
  }

  numTotalValSize = nextAvailableValId;
}


void MicroPartLocalValueAllocator::allocateLocalValues() {

  assert(valToLifeTime.size() != 0);

  mlir::DenseSet<mlir::Value> unPinnedVals;
  unPinnedVals.reserve(valToLifeTime.size());

  mlir::SmallVector<mlir::SmallVector<mlir::Value>> lifeTimeStartToVal, lifeTimeEndToVal;
  lifeTimeStartToVal.resize(totalLevels + 1);
  lifeTimeEndToVal.resize(totalLevels + 1);

  // Ignore pre-allocated const and io vals
  for (auto [eachVal, lifeTime]: valToLifeTime) {
    bool isSegmentVal = isa<toucan::StaticVectorSegmentReadOp>(eachVal.getDefiningOp());
    bool isPinnedVal = pinnedInputVals.contains(eachVal) || pinnedOutputVals.contains(eachVal) || constVals.contains(eachVal);

    if (!isSegmentVal) {
      assert(isPinnedVal == valToValId.contains(eachVal));
      if (!isPinnedVal) {
        unPinnedVals.insert(eachVal);
      }
    }


    // group vals by life time
    assert(lifeTime.start <= totalLevels);
    assert(lifeTime.end <= totalLevels);
    if (isa<toucan::VectorReadOp>(eachVal.getDefiningOp())
      || isa<toucan::StaticVectorSegmentReadOp>(eachVal.getDefiningOp())) {
      // Result of VecRead & StaticVectorSegmentRead might not be used due to replication by RepCut
      // So here we allow lifeTime.start == lifeTime.end
      assert(lifeTime.start <= lifeTime.end);
    } else {
      // start == end: value not used
      // Now it's OK to have value not used, since micro part could be duplicated then partitioning, original user of certain output value may no longer in current repcut partition
      assert(lifeTime.start <= lifeTime.end);
    }

    lifeTimeStartToVal[lifeTime.start].push_back(eachVal);
    lifeTimeEndToVal[lifeTime.end].push_back(eachVal);

    activeValuesAtLast.insert(eachVal);
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


  // const vals and pinned vals (reg read, reg write)
  for (auto [eachVal, eachValId]: valToValId) {
    assert(!isa<toucan::DefConstVectorOp>(eachVal.getDefiningOp()));
    if (isa<toucan::ConstantOp>(eachVal.getDefiningOp())) continue;
    if (isa<toucan::StaticVectorSegmentReadOp>(eachVal.getDefiningOp())) continue;
    // pinned vals
    assert(valToLifeTime.contains(eachVal));
    auto valLifeTime = valToLifeTime.at(eachVal);

    bool isVecVal = isa<mlir::TypedValue<toucan::VecType>>(eachVal);
    size_t valByteCount = (isVecVal) ? vecValToLength[eachVal] : 1;

    assert(!pinnedOutputValIdToNextOccupy.contains(eachValId));

    assert(valLifeTime.end <= totalLevels);
    if (pinnedOutputVals.contains(eachVal)) {
      // last level output vals
      pinnedOutputValIdToNextOccupy[eachValId] = valLifeTime.start;

      if (isVecVal) {
        // a vec val
        for (size_t i = 0; i < valByteCount; i++) {
          pinnedOutputValIdToNextOccupy[eachValId + i] = valLifeTime.start;
        }

      } else {
        // if is not a segment val
        if (!vecArithAndSegmentValues.contains(eachVal)) {
          pinnedOutputValIdToNextOccupy[eachValId] = valLifeTime.start;
        }
      }
    } else {
      // must be an input val (like reg read) or const
      assert(valLifeTime.start == 0);
    }

    nextAvailableValId += valByteCount;
    // nextAvailableValId = std::max(nextAvailableValId, static_cast<size_t>(eachValId));
  }
  assert(numOutputVals == pinnedOutputVals.size());
  assert(pinnedOutputValIdToNextOccupy.size() >= numOutputVals);
  nextAvailableValId++;

  // pinned vals may be used later, as long as its value is released
  for (auto [eachVal, valId]: valToValId) {
    if (valId >= numConsts) {
      if (isa<mlir::TypedValue<toucan::VecType>>(eachVal)) {
        // a vec val
        assert(vecValToLength.contains(eachVal));
        auto vecLength = vecValToLength[eachVal];

        for (size_t i = 0; i < vecLength; i++) {
          assert(!availableValIds.contains(valId + i));
          availableValIds.insert(valId + i);
          assert(valId + i < nextAvailableValId);
        }
      } else {
        // if is not a segment val
        if (!vecArithAndSegmentValues.contains(eachVal)) {
          assert(!availableValIds.contains(valId));
          availableValIds.insert(valId);
          assert(valId < nextAvailableValId);
        }
      }
    }
  }

  // Since now pinnedInputVals are read before level 0, they should also allocate space before level 0
  // preallocate for pinnedInputVals (remove from available list)
  for (auto & eachVal: pinnedInputVals) {
    assert(!isa<toucan::StaticVectorSegmentReadOp>(eachVal.getDefiningOp()));
    auto pinnedValIds = valToValId.at(eachVal);
    size_t byteCount = (vecValToLength.contains(eachVal)) ? vecValToLength[eachVal] : 1;
    for (size_t i = 0; i < byteCount; i++) {
      auto valId = pinnedValIds + i;
      assert(availableValIds.contains(valId));
      availableValIds.erase(valId);
    }
  }

  for (size_t levelId = 0; levelId < totalLevels; levelId++) {
    auto &valsToAllocate = lifeTimeStartToVal[levelId];
    auto &valsToRelease = lifeTimeEndToVal[levelId];

    // sort in descending order of val life time ends
    auto getValueLength = [](const mlir::Value &val) -> uint64_t {
      if (auto vecVal = dyn_cast<mlir::TypedValue<toucan::VecType>>(val)) {
        // a vector
        return vecVal.getType().getLength();
      } 
      return 1;
    };
    // Move wide vals to first
    std::sort(valsToAllocate.begin(), valsToAllocate.end(), [&](const mlir::Value &a, const mlir::Value &b) {
      auto a_length = getValueLength(a);
      auto b_length = getValueLength(b);
      return a_length > b_length;
    });
    // allocate long-lasting vals first
    std::stable_sort(valsToAllocate.begin(), valsToAllocate.end(), [&](const mlir::Value &a, const mlir::Value &b) {
      assert(valToLifeTime.contains(a));
      assert(valToLifeTime.contains(b));
      auto a_end = valToLifeTime.at(a).end;
      auto b_end = valToLifeTime.at(b).end;
      return a_end > b_end;
    });
    // uint32_t pinnedCount = 0;

    // allocate for all values that starts active in this level
    for (auto eachVal: valsToAllocate) {

      // Don't allocate space for segment values. They should be part of the vector.
      if (vecSegmentsToVecArith.contains(eachVal)) continue;

      if (unPinnedVals.contains(eachVal)) {
        assert(!valToValId.contains(eachVal));

        // Not pinned
        if (vecValToLength.contains(eachVal)) {
          assert(isa<toucan::DefVectorOp>(eachVal.getDefiningOp()) || isa<toucan::VectorArithOp>(eachVal.getDefiningOp()));
          // a vector
          auto vecLength = vecValToLength[eachVal];
          assert(valToLifeTime.contains(eachVal));
          auto vecValEndTime = valToLifeTime.at(eachVal).end;

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
            assert(valToLifeTime.contains(eachVal));
            auto valEndTime = valToLifeTime.at(eachVal).end;
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
        if (pinnedInputVals.contains(eachVal)) continue;
        // assert(!vecArithAndSegmentValues.contains(eachVal));
        if (isa<mlir::TypedValue<toucan::VecType>>(eachVal)) {
          // a vec val
          assert(vecValToLength.contains(eachVal));
          auto vecLength = vecValToLength[eachVal];

          for (size_t i = 0; i < vecLength; i++) {
            auto pinnedValIds = valToValId.at(eachVal) + i;
            assert(availableValIds.contains(pinnedValIds));
            availableValIds.erase(pinnedValIds);
          }
        } else {
          // if is not a segment val
          if (!vecArithAndSegmentValues.contains(eachVal)) {
            auto pinnedValIds = valToValId.at(eachVal);
            assert(availableValIds.contains(pinnedValIds));
            availableValIds.erase(pinnedValIds);
          }
        }
      }
    }


    // Release val ids that no longer used
    for (auto eachVal: valsToRelease) {
      activeValuesAtLast.erase(eachVal);
      // Don't allocate space for segment values. They should be part of the vector.
      if (vecSegmentsToVecArith.contains(eachVal)) continue;

      assert(valToValId.contains(eachVal));

      if (vecValToLength.contains(eachVal)) {
        // a vector
        auto vecLength = vecValToLength[eachVal];
        auto vecValId = valToValId[eachVal];
        for (uint32_t valId = vecValId; valId < (vecValId + vecLength); valId++) {
          availableValIds.insert(valId);
        }
      } else {
        // a regular value
        auto valIdToRelease = valToValId[eachVal];
        availableValIds.insert(valIdToRelease);
        assert(valIdToRelease >= numConsts);
      }

    }
  }

  // Assign ID for segment values
  for (const auto &[eachVecVal, segmentVals]: vecArithResultToSegments) {
    assert(valToValId.contains(eachVecVal) && "Result vector of VecArith should be already allocated!");
    auto vecValId = valToValId[eachVecVal];
    auto vecLength = vecValToLength[eachVecVal];

    for (const auto &segmentVal: segmentVals) {
      assert(valToLifeTime.contains(segmentVal));
      if (valToValId.contains(segmentVal)) {
        dbgs() << "Value that should not appear! defined by:\n";
        segmentVal.getDefiningOp()->print(dbgs());
        dbgs() << "\n";
      }
      assert(!valToValId.contains(segmentVal));
      auto segReadOp = cast<toucan::StaticVectorSegmentReadOp>(segmentVal.getDefiningOp());
      assert(segReadOp.getHandle() == eachVecVal);

      auto segmentId = segReadOp.getSegmentId().getZExtValue();
      assert(segmentId < vecLength);

      valToValId[segmentVal] = vecValId + segmentId;
    }
  }

  numTotalValSize = nextAvailableValId;
}


void MicroPartLocalValueAllocator::populateInitialPinnedVals(RepCutPartitionCodeGenData &partData, const mlir::DenseMap<mlir::Value, uint32_t> constValToRawValue) {

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

  assert(partData.allRegWrites.empty() || partData.allExgWriteVals.empty());

  // allocate for reg write. They should be placed at the begining to ensure performance
  for (auto &regWriteOp: partData.allRegWrites) {
    auto outputVal = regWriteOp.getData();
    assert(!pinnedOutputVals.contains(outputVal));

    pinnedOutputVals.insert(outputVal);
    // insert this val
    auto realValId = nextValId;
    numOutputVals++;
    // Vector val cannot be directly used as partition output val
    assert(!vecValToLength.contains(outputVal));
    assert(!valToValId.contains(outputVal));
    valToValId[outputVal] = realValId;

    nextValId++;
  }
  // allocate for exchange write. They should be placed at the begining to ensure performance
  for (auto &outputVal: partData.allExgWriteVals) {
    assert(!isa<toucan::StaticVectorSegmentReadOp>(outputVal.getDefiningOp()));
    assert(!pinnedOutputVals.contains(outputVal));

    size_t byteCount = 1;
    bool isVec = false;
    if (auto vecVal = dyn_cast<mlir::TypedValue<toucan::VecType>>(outputVal)) {
      isVec = true;
      byteCount = vecVal.getType().getLength();
    }

    pinnedOutputVals.insert(outputVal);
    // insert this val
    auto realValId = nextValId;
    numOutputVals++;

    // assert(!vecValToLength.contains(outputVal));
    if (isVec) {
      // exchange vals can be a vector
      assert(vecValToLength.contains(outputVal));
      assert(vecValToLength[outputVal] == byteCount);
    }
    valToValId[outputVal] = realValId;

    nextValId += byteCount;
  }

  assert(numOutputVals == pinnedOutputVals.size());


  assert(partData.allRegReads.empty() || partData.allExgReadVals.empty());
  // allocate reg reads
  for (auto &regReadOp: partData.allRegReads) {
    auto inputVal = regReadOp.getResult();

    assert(!vecValToLength.contains(inputVal));
    assert(!pinnedOutputVals.contains(inputVal));
    assert(!pinnedInputVals.contains(inputVal));
    assert(!constVals.contains(inputVal));
    pinnedInputVals.insert(inputVal);
    auto realValId = nextValId;
    numInputVals++;

    assert(!valToValId.contains(inputVal));
    valToValId[inputVal] = realValId;
    nextValId++;
  }
  // allocate exchange reads
  for (auto &inputVal: partData.allExgReadVals) {
    assert(!pinnedOutputVals.contains(inputVal) && "Possibly an edge from input directly to output");
    assert(!pinnedInputVals.contains(inputVal));
    assert(!constVals.contains(inputVal));

    mlir::SmallVector<mlir::Value> segmentVals;

    size_t byteCount = 1;
    bool isVec = false;
    if (auto vecVal = dyn_cast<mlir::TypedValue<toucan::VecType>>(inputVal)) {
      isVec = true;
      byteCount = vecVal.getType().getLength();

      for (auto eachUser: vecVal.getUsers()) {
        if (auto segmentReadOp = dyn_cast<toucan::StaticVectorSegmentReadOp>(eachUser)) {
          auto segmentVal = segmentReadOp.getResult();
          segmentVals.push_back(segmentVal);
        }
      }
    }

    if (isVec) {
      // exchange vals can be a vector
      assert(vecValToLength.contains(inputVal));
      assert(vecValToLength[inputVal] == byteCount);
    } else {
      assert(!vecValToLength.contains(inputVal));
    }

    pinnedInputVals.insert(inputVal);
    auto realValId = nextValId;
    numInputVals++;

    assert(!valToValId.contains(inputVal));
    valToValId[inputVal] = realValId;
    nextValId += byteCount;


    // Is this necessary?
    // // map segment values as well
    // for (auto eachSegmentVal: segmentVals) {
    //   auto segOp = cast<toucan::StaticVectorSegmentReadOp>(eachSegmentVal.getDefiningOp());

    //   auto segId = segOp.getSegmentId().getZExtValue();
    //   assert(segId < byteCount);

    //   valToValId[eachSegmentVal] = realValId + segId;
    // }

  }

  assert(numInputVals == pinnedInputVals.size());
}




void MicroPartLocalValueAllocator::collectValueLifetime(RepCutPartitionCodeGenData &partData) {

  auto checkValueExistance = [&](mlir::Value val) {
    if (!valToLifeTime.contains(val)) {
      llvm::dbgs() << "Value not found in life time pool. Defining op:\n";
      val.getDefiningOp()->print(llvm::dbgs());

      llvm::dbgs() << "\n";
    }
  };
  // 1. For every reg read, result val life starts at 0
  for (auto &regReadOp: partData.allRegReads) {
    auto resultVal = regReadOp.getResult();

    assert(!valToLifeTime.contains(resultVal));
    valToLifeTime[resultVal] = {0, 0};
  }
  for (auto &exgReadVal: partData.allExgReadVals) {
    assert(!valToLifeTime.contains(exgReadVal));
    valToLifeTime[exgReadVal] = {0, 0};

    // Save vector size
    if (isa<mlir::TypedValue<toucan::VecType>>(exgReadVal)) {
      auto vecLength = cast<mlir::TypedValue<toucan::VecType>>(exgReadVal).getType().getLength();
      if (vecValToLength.contains(exgReadVal)) {
        assert(vecValToLength[exgReadVal] == vecLength);
      } else {
        vecValToLength[exgReadVal] = vecLength;
      }
    }

    if (auto vecArithOp = dyn_cast<toucan::VectorArithOp>(exgReadVal.getDefiningOp())) {
      mlir::SmallVector<mlir::Value> segmentVals;
      for (auto eachUser: exgReadVal.getUsers()) {
        if (auto segmentReadOp = dyn_cast<toucan::StaticVectorSegmentReadOp>(eachUser)) {
          auto segmentVal = segmentReadOp.getResult();
          segmentVals.push_back(segmentVal);
        }
      }

      if (segmentVals.size() != 0) {
        auto vecVal = vecArithOp.getResult();

        for (auto eachSegmentVal: segmentVals) {
          assert(!vecSegmentsToVecArith.contains(eachSegmentVal));
          vecSegmentsToVecArith[eachSegmentVal] = vecVal;

          assert(!vecArithAndSegmentValues.contains(eachSegmentVal));
          vecArithAndSegmentValues.insert(eachSegmentVal);

          // also save its life time
          assert(!valToLifeTime.contains(eachSegmentVal));
          valToLifeTime[eachSegmentVal] = {0, 0};
        }

        assert(!vecArithAndSegmentValues.contains(vecVal));
        vecArithAndSegmentValues.insert(vecVal);

        assert(!vecArithResultToSegments.contains(vecVal));
        vecArithResultToSegments[vecVal] = segmentVals;
      }
    }
  }

  
  for (uint32_t levelId = 0; levelId < partData.mpartLevels.size(); levelId++) {
    for (const auto &mPart: partData.mpartLevels[levelId]) {
      for (const auto &eachInVal: mPart->inputValues) {
        // update val end time accordingly
        if (!isa<toucan::DefConstVectorOp>(eachInVal.getDefiningOp())) {
          // ignore const vec. They are placed in constVecPool and always alive
          assert(valToLifeTime.contains(eachInVal));
          auto oldValEndTime = valToLifeTime[eachInVal].end;
          assert(oldValEndTime <= levelId);
          valToLifeTime[eachInVal].end = levelId;
        }
      }
      for (const auto &eachOutVal: mPart->outputValueSet) {
        // update val start time accordingly
        // Cannot write to a const vec
        assert(!isa<toucan::DefConstVectorOp>(eachOutVal.getDefiningOp()));

          if (valToLifeTime.contains(eachOutVal)) {
            // not the first time appear
            // Note： It's OK if a vector is written by multiple mparts
            assert(isa<mlir::TypedValue<toucan::VecType>>(eachOutVal));
            auto oldStartTime = valToLifeTime[eachOutVal].start;
            auto oldEndTime = valToLifeTime[eachOutVal].end;

            // Note: A vector val being written many times, should be considered available only at last write.
            // But life starts at the first. Here we only track life time
            assert(oldStartTime <= levelId);
            assert(oldEndTime <= levelId);
            valToLifeTime[eachOutVal].end = levelId;
          } else {
            valToLifeTime[eachOutVal] = {levelId, levelId};
          }
          

          // Save vector size
          if (isa<mlir::TypedValue<toucan::VecType>>(eachOutVal)) {
            auto vecLength = cast<mlir::TypedValue<toucan::VecType>>(eachOutVal).getType().getLength();
            if (vecValToLength.contains(eachOutVal)) {
              assert(vecValToLength[eachOutVal] == vecLength);
            } else {
              vecValToLength[eachOutVal] = vecLength;
            }
          }


          if (auto vecArithOp = dyn_cast<toucan::VectorArithOp>(eachOutVal.getDefiningOp())) {
            mlir::SmallVector<mlir::Value> segmentVals;
            for (auto eachUser: eachOutVal.getUsers()) {
              if (auto segmentReadOp = dyn_cast<toucan::StaticVectorSegmentReadOp>(eachUser)) {
                auto segmentVal = segmentReadOp.getResult();
                segmentVals.push_back(segmentVal);
              }
            }

            if (segmentVals.size() != 0) {
              auto vecVal = vecArithOp.getResult();

              for (auto eachSegmentVal: segmentVals) {
                assert(!vecSegmentsToVecArith.contains(eachSegmentVal));
                vecSegmentsToVecArith[eachSegmentVal] = vecVal;

                assert(!vecArithAndSegmentValues.contains(eachSegmentVal));
                vecArithAndSegmentValues.insert(eachSegmentVal);

                // also save its life time
                assert(!valToLifeTime.contains(eachSegmentVal));
                valToLifeTime[eachSegmentVal] = {levelId, levelId};
              }

              assert(!vecArithAndSegmentValues.contains(vecVal));
              vecArithAndSegmentValues.insert(vecVal);

              assert(!vecArithResultToSegments.contains(vecVal));
              vecArithResultToSegments[vecVal] = segmentVals;
            }
          }
      }
    }
  }

  totalLevels = partData.mpartLevels.size();

  // regwrite, memwrite, stop, print survive to last level (totalLevels)
  for (auto &regWriteOp: partData.allRegWrites) {
    auto inputVal = regWriteOp.getData();

    assert(valToLifeTime.contains(inputVal));
    valToLifeTime[inputVal].end = totalLevels;
  }

  for (auto &exgWriteVal: partData.allExgWriteVals) {
    assert(valToLifeTime.contains(exgWriteVal));
    valToLifeTime[exgWriteVal].end = totalLevels;
  }
  
  for (auto &mwOp: partData.allMemWrites) {
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
  for (auto &stopOp: partData.allStops) {
    auto inputVal = stopOp.getEn();

    if (!isa<toucan::ConstantOp>(inputVal.getDefiningOp())) {
      checkValueExistance(inputVal);

      assert(valToLifeTime.contains(inputVal));
      valToLifeTime[inputVal].end = totalLevels;
    }
  }

  // print
  for (auto &printOp: partData.allPrints) {
    auto inputVal = printOp.getEn();

    if (!isa<toucan::ConstantOp>(inputVal.getDefiningOp())) {
      checkValueExistance(inputVal);

      assert(valToLifeTime.contains(inputVal));
      valToLifeTime[inputVal].end = totalLevels;
    }
  }

  // VecArith result and VecSegRead result should have same life time!
  for (const auto &[eachVecVal, segmentVals]: vecArithResultToSegments) {
    assert(valToLifeTime.contains(eachVecVal));
    auto min_start = valToLifeTime.at(eachVecVal).start;
    auto max_end = valToLifeTime.at(eachVecVal).end;
    for (const auto &segmentVal: segmentVals) {
      assert(valToLifeTime.contains(segmentVal));
      min_start = std::min(min_start, valToLifeTime.at(segmentVal).start);
      max_end = std::max(max_end, valToLifeTime.at(segmentVal).end);
    }

    valToLifeTime[eachVecVal] = {min_start, max_end};
    for (const auto &segmentVal: segmentVals) {
      valToLifeTime[segmentVal] = {min_start, max_end};
    }
  }

}
