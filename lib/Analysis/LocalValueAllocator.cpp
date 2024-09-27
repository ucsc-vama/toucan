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

#include "toucan/PartitioningGraph.h"
#include "toucan/ToucanAnalysis.h"
#include "toucan/ToucanAttributes.h"
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

void LocalValueAllocator::allocateLocalValues() {
  assert(valToLifeTime.size() != 0);

  mlir::DenseSet<mlir::Value> unPinnedVals;
  unPinnedVals.reserve(valToLifeTime.size());

  mlir::SmallVector<mlir::SmallVector<mlir::Value>> lifeTimeStartToVal, lifeTimeEndToVal;
  lifeTimeStartToVal.resize(totalLevels);
  lifeTimeEndToVal.resize(totalLevels);

  // Ignore pre-allocated const and io vals
  for (auto [eachVal, lifeTime]: valToLifeTime) {
    // save vals that needs allocation
    // if (!valToValId.contains(eachVal)) {
    //   unPinnedVals.insert(eachVal);
    // }

    bool isPinnedVal = pinnedInputVals.contains(eachVal) || pinnedOutputVals.contains(eachVal) || constVals.contains(eachVal);
    assert(isPinnedVal == valToValId.contains(eachVal));
    if (!isPinnedVal) {
      unPinnedVals.insert(eachVal);
    }

    // group vals by life time
    assert(lifeTime.start < totalLevels);
    assert(lifeTime.end < totalLevels);
    assert(lifeTime.start < lifeTime.end);

    lifeTimeStartToVal[lifeTime.start].push_back(eachVal);
    lifeTimeEndToVal[lifeTime.end].push_back(eachVal);
  }


  // Note: availableValIds must be ordered!
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

  for (auto [eachVal, eachValId]: valToValId) {
    // pinned vals
    auto valLifeTime = valToLifeTime[eachVal];

    assert(!pinnedOutputValIdToNextOccupy.contains(eachValId));

    assert(valLifeTime.end < totalLevels);
    if (pinnedOutputVals.contains(eachVal)) {
      // last level output vals
      pinnedOutputValIdToNextOccupy[eachValId] = valLifeTime.start;
    } else {
      // must be an input val
      assert(valLifeTime.start == 0);
    }
    nextAvailableValId = std::max(nextAvailableValId, static_cast<size_t>(eachValId));
  }
  assert(numOutputVals == pinnedOutputVals.size());
  assert(pinnedOutputValIdToNextOccupy.size() == numOutputVals);
  nextAvailableValId++;

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

          // temp
          continuousValSize = 0;
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
            // TODO: use part of existing vals
            uint32_t vecValIdStart = nextAvailableValId;
            valToValId[eachVal] = vecValIdStart;

for (uint32_t valId = vecValIdStart; valId < vecValIdStart + vecLength; valId++) {
  assert(!allocatedValIds.contains(valId));
    allocatedValIds.insert(valId);
}
            nextAvailableValId += vecLength;
#ifdef DEBUG_PRINT_LOCAL_VAL_ALLOC
          llvm::dbgs() << "Alloc " << vecLength << " vals for a vector\n";
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

          // temp
          foundFreeSlot = false;

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
assert(!allocatedValIds.contains(valId));
allocatedValIds.insert(valId);
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
        // pinnedCount++;
      }
    }

    // llvm::dbgs() << "When allocating, encouter " << pinnedCount << " pinned vals\n";


    // Release val ids that no longer used
    size_t numReleases = 0;
    for (auto eachVal: valsToRelease) {
      assert(valToValId.contains(eachVal));

      if (vecValToLength.contains(eachVal)) {
        // a vector
        auto vecLength = vecValToLength[eachVal];
        auto vecValId = valToValId[eachVal];
        for (uint32_t valId = vecValId; valId < (vecValId + vecLength); valId++) {
          availableValIds.insert(valId);
          numReleases++;
        }
      } else {
        auto valIdToRelease = valToValId[eachVal];
        availableValIds.insert(valIdToRelease);
        assert(valIdToRelease >= numConsts);
        numReleases++;
      }

    }
    if (numReleases != 0) {
      llvm::dbgs() << "Release " << numReleases << " vals\n";
    }
  }

  numTotalValSize = nextAvailableValId;
}

void LocalValueAllocator::populateInitialPinnedVals(PartitioningGraph &regionGraph, const mlir::SmallVector<CGValueMetaInfo> &valuePool, const mlir::SmallVector<mlir::SmallVector<uint32_t>> &partLevels, const mlir::SmallVector<CGExchangeValueMetaInfo> &exchangePool) {
  
  mlir::DenseMap<uint8_t, uint32_t> rawConstValToValId;
  // rawConstValToValId[0] = 0;
  compactConstValPool.push_back(0);  

  // assert(valuePool[0].value == 0 && valuePool[0].definingOp == nullptr);

  allocatedValIds.insert(0);

  for (size_t valIndex = 1; valIndex < valuePool.size(); valIndex++) {
    auto eachConstVal = valuePool[valIndex];
    assert(eachConstVal.isConst);

    assert(eachConstVal.definingOp != nullptr);
    auto val = cast<toucan::ConstantOp>(eachConstVal.definingOp).getResult();
    auto rawVal = eachConstVal.value;

    constVals.insert(val);

    // temp
    compactConstValPool.push_back(rawVal);
    valToValId[val] = valIndex;

    assert(!allocatedValIds.contains(valIndex));
    allocatedValIds.insert(valIndex);

    // if (!rawConstValToValId.contains(rawVal)) {
    //   // a new val
    //   auto newValId = compactConstValPool.size();
    //   rawConstValToValId[rawVal] = newValId;
    //   compactConstValPool.push_back(rawVal);
    //   valToValId[val] = newValId;
    // } else {
    //   // a existing val
    //   auto realValId = rawConstValToValId[rawVal];
    //   valToValId[val] = realValId;
    // }
  }

  numConsts = compactConstValPool.size();
  numOutputVals = 0;
  numInputVals = 0;

  uint32_t nextValId = numConsts;

  // allocate rw/ew
  const auto &lastLevelOps = partLevels.back();

  for (const auto eachVtx: lastLevelOps) {
    auto tOpName = regionGraph[eachVtx].toucanOpName;
    mlir::Value outputVal;
    bool hasOutputVal = true;

    if (tOpName == CGToucanOPName::ExchangeWrite) {
      auto exgValId = regionGraph[eachVtx].exchangeValId;
      assert(exchangePool.size() > exgValId);
      outputVal = exchangePool[exgValId].val;
    } else if (tOpName == CGToucanOPName::RegWrite) {
      assert(regionGraph[eachVtx].op != nullptr);
      auto regWriteOp = cast<toucan::RegWriteOp>(regionGraph[eachVtx].op);
      outputVal = regWriteOp.getData();
    } else {
      hasOutputVal = false;
    }

    if (hasOutputVal) {
      // a single val might be used as output val multiple times!!!
      if (!pinnedOutputVals.contains(outputVal)) {

        if (valToValId.contains(outputVal)) {
          // It's ok if this exgwrite reads from const pool
          assert(constVals.contains(outputVal));
        } else {
          pinnedOutputVals.insert(outputVal);
          // insert this val
          auto realValId = nextValId;
          numOutputVals++;
          // Vector val cannot be directly used as partition output val
          assert(!vecValToLength.contains(outputVal));

          assert(!valToValId.contains(outputVal));
          valToValId[outputVal] = realValId;


assert(!allocatedValIds.contains(realValId));
allocatedValIds.insert(realValId);

          nextValId++;
        }
      }
    }
  }
  assert(numOutputVals == pinnedOutputVals.size());


  // allocate rr/er
  const auto &firstLevelOps = partLevels[0];
  for (const auto eachVtx: firstLevelOps) {
    auto tOpName = regionGraph[eachVtx].toucanOpName;
    mlir::Value inputVal;
    bool hasInputVal = true;

    if (tOpName == CGToucanOPName::ExchangeRead) {
      auto exgValId = regionGraph[eachVtx].exchangeValId;
      assert(exchangePool.size() > exgValId);
      inputVal = exchangePool[exgValId].val;
    } else if (tOpName == CGToucanOPName::RegRead) {
      assert(regionGraph[eachVtx].op != nullptr);
      auto regReadOp = cast<toucan::RegReadOp>(regionGraph[eachVtx].op);
      inputVal = regReadOp.getResult();
    } else {
      hasInputVal = false;
      if (tOpName != CGToucanOPName::ConstDecl) {
        llvm::dbgs() << stringifyCGToucanOPName(tOpName) << "\n";
        llvm_unreachable("First level op should be ConstDecl, RegRead or ExchangeRead!");
      }
    }

    if (hasInputVal) {
      assert(!vecValToLength.contains(inputVal));
      assert(!pinnedOutputVals.contains(inputVal));
      assert(!constVals.contains(inputVal));
      pinnedInputVals.insert(inputVal);
      auto realValId = nextValId;
      numInputVals++;

      assert(!valToValId.contains(inputVal));
      valToValId[inputVal] = realValId;
assert(!allocatedValIds.contains(realValId));
allocatedValIds.insert(realValId);
      nextValId++;
    }
  }

  assert(numInputVals == pinnedInputVals.size());

}

void LocalValueAllocator::collectValueLifetime(PartitioningGraph &regionGraph, const mlir::SmallVector<mlir::SmallVector<uint32_t>> &partLevels, const mlir::SmallVector<CGExchangeValueMetaInfo> &exchangePool) {
  mlir::DenseMap<uint32_t, uint32_t> vtxIdToLevelId;

  uint32_t levelId = 0;
  for (const auto &eachLevel: partLevels) {
    for (const auto &eachVtx: eachLevel) {
      assert(!vtxIdToLevelId.contains(eachVtx));
      vtxIdToLevelId[eachVtx] = levelId;
    }
    levelId++;
  }

  totalLevels = levelId;

  auto getValEndOfLife = [&](const uint32_t vtxId) {
    auto userVtxRange = boost::out_edges(vtxId, regionGraph);
    uint32_t ret = 0;
    for (auto ei = userVtxRange.first; ei != userVtxRange.second; ei++) {
      auto userVtxId = boost::target(*ei, regionGraph);
      auto vtxUserLevel = vtxIdToLevelId[userVtxId];
      ret = std::max(ret, vtxUserLevel);
    }

    // temp
    // return ret;
    return totalLevels - 1;
  };

  levelId = 0;
  for (const auto &eachLevel: partLevels) {
    for (const auto &eachVtx: eachLevel) {
      auto tOpName = regionGraph[eachVtx].toucanOpName;
      auto op = regionGraph[eachVtx].op;

      mlir::Value resultVal;
      bool hasResultVal = true;

      switch (tOpName) {
        case CGToucanOPName::ConstDecl: {
          // resultVal = cast<toucan::ConstantOp>(op).getResult();
          hasResultVal = false;
          break;
        }
        case CGToucanOPName::LUT: {
          resultVal = cast<toucan::LUTOp>(op).getResult();
          break;
        }
        case CGToucanOPName::VecRead: {
          resultVal = cast<toucan::VectorReadOp>(op).getResult();
          break;
        }
        case CGToucanOPName::VecDecl: {
          if (auto vecDeclOp = dyn_cast<toucan::DefVectorOp>(op)) {
            resultVal = vecDeclOp.getHandle();
          } else {
            // a const vec, do not allocate storage in shared mem
            hasResultVal = false;
          }
          break;
        }
        case CGToucanOPName::RegRead: {
          resultVal = cast<toucan::RegReadOp>(op).getResult();
          break;
        }
        case CGToucanOPName::MemRead: {
          resultVal = cast<toucan::MemReadOp>(op).getResult();
          break;
        }
        case CGToucanOPName::ExchangeRead: {
          auto exchangeValId = regionGraph[eachVtx].exchangeValId;
          assert(exchangePool.size() > exchangeValId);
          resultVal = exchangePool[exchangeValId].val;
          break;
        }

        case CGToucanOPName::ExchangeWrite:
        case CGToucanOPName::MemWrite:
        case CGToucanOPName::RegWrite:
        case CGToucanOPName::Print:
        case CGToucanOPName::Stop: {
          hasResultVal = false;
          break;
        }

        default: {
          llvm_unreachable("Should not reach here");
        }
      }

      if (hasResultVal) {
        uint32_t valLifeStart = levelId;
        uint32_t valLifeEnd = getValEndOfLife(eachVtx);
        

        if (tOpName == CGToucanOPName::VecDecl) {
          // update value size
          assert(!isa<toucan::DefConstVectorOp>(op));

          auto vecLength = cast<mlir::TypedValue<toucan::VecType>>(resultVal).getType().getLength();
          assert(!vecValToLength.contains(resultVal));
          vecValToLength[resultVal] = vecLength;

          valToLifeTime[resultVal] = {valLifeStart, valLifeEnd};
        } else if (tOpName == CGToucanOPName::VecRead) {
          // For vec read, all readers of the same vector are merged together
          auto vecHandle = cast<toucan::VectorReadOp>(op).getHandle();

          for (auto eachVecUser: vecHandle.getUsers()) {
            assert(isa<toucan::VectorReadOp>(eachVecUser));
            auto resultVal = cast<toucan::VectorReadOp>(eachVecUser).getResult();

            valToLifeTime[resultVal] = {valLifeStart, valLifeEnd};
          }
        } else {
          // normal op with only 1 result value
          valToLifeTime[resultVal] = {valLifeStart, valLifeEnd};
        }
      }
    }
    levelId++;
  }


  // for all constants, they shall alive to the end
}