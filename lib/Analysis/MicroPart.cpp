#include "mlir/IR/Value.h"
#include "mlir/Support/LLVM.h"
#include "toucan/MicroPartitioner.h"
#include "toucan/PartitioningGraph.h"
#include "toucan/ToucanAttributes.h"
#include "toucan/ToucanOps.h"
#include "toucan/ToucanTypes.h"
#include "toucan/ToucanUtils.h"
#include "llvm/Support/Debug.h"
#include "llvm/Support/ErrorHandling.h"
#include "llvm/Support/raw_ostream.h"
#include <algorithm>
#include <cstddef>


using namespace toucan;

void MicroPart::clear() {
  nodes.clear();
  levels.clear();
  inputValues.clear();
  outputValues.clear();
  outputValueSet.clear();
  nodeToInputVals.clear();
  nodeToOutputVal.clear();
  specialOps.clear();
  valuesUsedByEachLevel.clear();
}

void MicroPart::updateNodeToLevel() {
  assert(!levels.empty());
  nodeToLevel.clear();

  for (size_t levelId = 0; levelId < levels.size(); levelId++) {
    for (auto &eachNode: levels[levelId]) {
      assert(!nodeToLevel.contains(eachNode));
      nodeToLevel[eachNode] = static_cast<uint32_t>(levelId);
    }
  }
}

void MicroPart::buildRegularLUTPart(const mlir::SmallVector<mlir::SmallVector<uint32_t>> &newNodesLevel) {
  assert(nodes.empty());
  assert(nodeToOpCount.empty());

  size_t totalNodeCount = 0;
  for (const auto &eachLevelNodes: newNodesLevel) {
    for (const auto &eachVtx: eachLevelNodes) {
      totalNodeCount ++;
      nodes.insert(eachVtx);
      // For LUT part, each node is just 1 op
      nodeToOpCount[eachVtx] = 1;
    }
  }
  levels = newNodesLevel;

  assert(nodes.size() == totalNodeCount);

  opType = CGToucanOPName::LUT;
  totalOpCount = totalNodeCount;
  updateNodeToLevel();
}


void MicroPart::buildSpecialPart(const CGToucanOPName vtxOpName, const mlir::SmallVector<mlir::Operation*> &rawOps) {
  assert(rawOps.size() <= 32);
  assert(nodes.empty());
  assert(nodeToOpCount.empty());
  totalOpCount = rawOps.size();
  levels.clear();

  for (auto rawOp: rawOps) {
    specialOps.push_back(rawOp);
  }

  opType = vtxOpName;
  assert(opType != CGToucanOPName::LUT);
  assert(totalOpCount > 0);
  assert(totalOpCount <= 32 && "Number of real ops in a special part cannot exceed hardware limit!");
}


bool MicroPart::checkAndCollectRegularPartIOValues(const PartitioningGraph &g, const mlir::DenseSet<uint32_t> &allNodes, const mlir::DenseMap<uint32_t, uint32_t> &newNodeIdToDepNodeId, const mlir::DenseMap<uint32_t, uint32_t> &newNodeIdToOriginalVecDeclId, const mlir::DenseMap<uint32_t, mlir::SmallVector<uint32_t>> outputVectorNopMap) {
  assert(opType == toucan::CGToucanOPName::LUT);
  assert(!levels.empty() && "Only check and collect IO values if it's loaded");

  inputValues.clear();
  outputValues.clear();
  outputValueSet.clear();
  nodeToInputVals.clear();
  nodeToOutputVal.clear();
  valuesUsedByEachLevel.clear();

  mlir::DenseSet<mlir::Value> allPreviousLevelResultValues;

  auto findDummyNopDepValue = [&](uint32_t dummyVtx) {
    auto vecDeclId = newNodeIdToOriginalVecDeclId.at(dummyVtx);
    auto vecOp = cast<toucan::DefVectorOp>(g[vecDeclId].op);

    uint32_t i = 0;
    const auto &thisVecNewIds = outputVectorNopMap.at(vecDeclId);
    for (;i < thisVecNewIds.size(); i++) {
      if (thisVecNewIds[i] == dummyVtx) break;
    }
    assert(i < thisVecNewIds.size());

    return vecOp.getInputs()[i];
  };



  // Check correctness and collect input value
  for (size_t levelId = 0; levelId < levels.size(); levelId++) {
    assert(!levels[levelId].empty());
    const auto &currentLevelNodes = levels[levelId];
    valuesUsedByEachLevel.emplace_back();

    for (auto eachVtx: currentLevelNodes) {
      assert(nodes.contains(eachVtx));
      auto vtxIsDummyNop = newNodeIdToDepNodeId.contains(eachVtx);
      if (vtxIsDummyNop) dummyNodes.insert(eachVtx);

      if (!vtxIsDummyNop) {
        auto rawOp = g[eachVtx].op;
        if (!isa<toucan::LUTOp>(rawOp)) {
          rawOp->print(llvm::errs());
          llvm::errs() << "\n";
          llvm::errs() << "Op type in graph is " << stringifyCGToucanOPName(g[eachVtx].toucanOpName) << "\n";
        }
        assert(isa<toucan::LUTOp>(rawOp));
      }

      // Must read something
      if (!vtxIsDummyNop) {
        auto inDegree = boost::in_degree(eachVtx, g);
        if (inDegree == 0) {
          llvm::errs() << "In micro part level " << levelId << ", vtx " << eachVtx << " has in degree == 0\n";
          auto vtxRawOp = g[eachVtx].op;
          vtxRawOp->print(llvm::errs());
          llvm::errs() << "\n";
          // llvm::errs() << stringifyCGToucanOPName(opType) << "\n";
          return false;
        }
      }



      // Check if is levelized correctly, also collect input val

      if (vtxIsDummyNop) {
        auto depVtx = newNodeIdToDepNodeId.at(eachVtx);
        auto depValue = findDummyNopDepValue(eachVtx);
        assert(!isa<mlir::TypedValue<toucan::VecType>>(depValue));

        if (!isa<toucan::ConstantOp>(depValue.getDefiningOp())) {
          // Ignore const
          if (nodeToInputVals.contains(eachVtx)) {
            nodeToInputVals[eachVtx].push_back(depValue);
          } else {
            nodeToInputVals[eachVtx] = {depValue};
          }

          if (!nodes.contains(depVtx)) {
            // a vecDecl reads element from outside
            inputValues.insert(depValue);
          }

          valuesUsedByEachLevel.back().insert(depValue);
        }

        if (nodes.contains(depVtx)) {
          // internal node read and write to vector. Just check correctness
          auto depVtxLevel = nodeToLevel[depVtx];
          assert(depVtxLevel + 1 == levelId);
        }
      } else {
        // regular node
        auto rawOp = g[eachVtx].op;
        for (auto eachOperand: rawOp->getOperands()) {
          assert(!isa<mlir::TypedValue<toucan::VecType>>(eachOperand));
          assert(!isa<mlir::TypedValue<toucan::RegType>>(eachOperand));
          assert(!isa<mlir::TypedValue<toucan::MemType>>(eachOperand));

          if (!isa<toucan::ConstantOp>(eachOperand.getDefiningOp())) {
            if (nodeToInputVals.contains(eachVtx)) {
              nodeToInputVals[eachVtx].push_back(eachOperand);
            } else {
              nodeToInputVals[eachVtx] = {eachOperand};
            }
            if (!allPreviousLevelResultValues.contains(eachOperand)) {
              inputValues.insert(eachOperand);
            }
            valuesUsedByEachLevel.back().insert(eachOperand);
          }
        }

        allPreviousLevelResultValues.insert(getOpResultValue(rawOp));
        
      }

      // Collect output vals
      if (vtxIsDummyNop) {
        // vec decl. Must be used by outside
        auto originalVecVtx = newNodeIdToOriginalVecDeclId.at(eachVtx);
        auto rawDeclOp = g[originalVecVtx].op;
        auto vecDeclOp = cast<toucan::DefVectorOp>(rawDeclOp);
        auto resultVecVal = vecDeclOp.getResult();

        // assert(!outputValues.contains(resultVecVal));
        outputValues.push_back(resultVecVal);
        outputValueSet.insert(resultVecVal);
        nodeToOutputVal[eachVtx] = resultVecVal;
      } else {
        // regular lut
        assert(g[eachVtx].toucanOpName == CGToucanOPName::LUT);
        auto rawOp = g[eachVtx].op;
        assert(rawOp != nullptr);
        auto lutOp = cast<toucan::LUTOp>(rawOp);
        auto resultVal = lutOp.getResult();

        assert(!nodeToOutputVal.contains(eachVtx));
        nodeToOutputVal[eachVtx] = resultVal;


        bool outputValueUsedOutsideThisPartition = false;

        mlir::SmallVector<uint32_t> allUserVtxes;
        for (auto eachUserEdge: boost::make_iterator_range(boost::out_edges(eachVtx, g))) {
          auto eachUserVtx = boost::target(eachUserEdge, g);
          allUserVtxes.push_back(eachUserVtx);
        }

        for (auto edgeTarget: allUserVtxes) {
          // assert(!allPreviousLevelLUTNodes.contains(edgeTarget) && "Should follow topo order");

          // result used by outside of this partition
          if (!nodes.contains(edgeTarget)) {
            // result is not used by dummy vecdecl in same partition
            bool targetIsVecDecl = isa<toucan::DefVectorOp>(g[edgeTarget].op);
            bool targetInCurrentRepCutPartition = allNodes.contains(edgeTarget);
            if (targetIsVecDecl) {
              if (!targetInCurrentRepCutPartition) continue;
              assert(outputVectorNopMap.contains(edgeTarget));

              bool allUserDummyNopInThisMPart = true;
              for (auto eachUserNop: outputVectorNopMap.at(edgeTarget)) {
                auto thisNopUseThisValue = newNodeIdToDepNodeId.at(eachUserNop) == eachVtx;
                if (thisNopUseThisValue && !nodes.contains(eachUserNop)) {
                  allUserDummyNopInThisMPart = false;
                  break;
                }
              }
              if (targetInCurrentRepCutPartition && !allUserDummyNopInThisMPart) {
                outputValueUsedOutsideThisPartition = true;
                break;
              }
            } else {
              if (targetInCurrentRepCutPartition) {
                outputValueUsedOutsideThisPartition = true;
                break;
              }
            }
            // if ((!targetIsVecDecl) && targetInCurrentRepCutPartition) {
            //   outputValueUsedOutsideThisPartition = true;
            //   break;
            // }
          }
        }
        // Save output Value
        if (outputValueUsedOutsideThisPartition) {
          assert(!vtxIsDummyNop);

          outputValues.push_back(resultVal);
          assert(!outputValueSet.contains(resultVal));
          outputValueSet.insert(resultVal);
        } else {
          assert(g[eachVtx].op != nullptr);
          assert(!isa<toucan::DefVectorOp>(g[eachVtx].op));
        }
      }

    }

  }

mlir::DenseSet<mlir::Value> visitedInputVals;
for (const auto &[_, vs]: nodeToInputVals) {
  for (const auto &v: vs) {
    visitedInputVals.insert(v);
  }
}
for (const auto &val: inputValues) {
  if (!visitedInputVals.contains(val)) {
    llvm::dbgs() << "Value missing\n";
    val.print(llvm::dbgs());
    llvm::dbgs() << "\n";
  }
  assert(visitedInputVals.contains(val));
}
  // if (allInputVals.size() != inputValues.size()) {
  //   llvm::dbgs() << "Incorrect input values\nallInputVals from nodeToInputVals:\n";
  //   for (const auto &v: allInputVals) {
  //     v.print(llvm::dbgs());
  //     llvm::dbgs() << "\n";
  //   }
  //   llvm::dbgs() << "inputValues:\n";
  //   for (const auto &v: inputValues) {
  //     v.print(llvm::dbgs());
  //     llvm::dbgs() << "\n";
  //   }
  // }

  return true;
}

bool MicroPart::checkAndCollectSpecialPartIOValues(const PartitioningGraph &g, const mlir::DenseMap<uint32_t, uint32_t> &newNodeIdToDepNodeId, const mlir::DenseMap<uint32_t, uint32_t> &newNodeIdToOriginalVecDeclId) {
  assert(opType != toucan::CGToucanOPName::LUT);
  assert(levels.empty() && "Special part should keep levels empty");
  assert(levels.size() == 0);

  inputValues.clear();
  outputValues.clear();
  outputValueSet.clear();
  nodeToInputVals.clear();
  nodeToOutputVal.clear();

  // Check correctness and collect input value
  for (auto rawOp: specialOps) {
    // collect input values
    for (auto eachOperand: rawOp->getOperands()) {
      if (!(
        isa<toucan::DefMemOp>(eachOperand.getDefiningOp()) ||
        isa<toucan::ConstantOp>(eachOperand.getDefiningOp()) || 
        isa<toucan::DefConstVectorOp>(eachOperand.getDefiningOp())
      )) {
        inputValues.insert(eachOperand);
      }
    }

    // Note: Result value of special part must be used (only 1 level), no need to check
    auto resultVal = getOpResultValue(rawOp);

    mlir::DenseSet<mlir::Value> dedupOutputValues;
    for (auto v: outputValues) dedupOutputValues.insert(v);
    assert(dedupOutputValues.size() == outputValues.size());

    outputValues.push_back(resultVal);
    assert(!outputValueSet.contains(resultVal));
    outputValueSet.insert(resultVal);
  }

  return true;
}



bool MicroPart::checkAndCollectIOValues(const PartitioningGraph &g, const mlir::DenseSet<uint32_t> &allNodes, const mlir::DenseMap<uint32_t, uint32_t> &newNodeIdToDepNodeId, const mlir::DenseMap<uint32_t, uint32_t> &newNodeIdToOriginalVecDeclId, const mlir::DenseMap<uint32_t, mlir::SmallVector<uint32_t>> outputVectorNopMap) {
  bool ret;
  bool isRegularPart = opType == CGToucanOPName::LUT;
  if (isRegularPart) {
    ret = checkAndCollectRegularPartIOValues(g, allNodes, newNodeIdToDepNodeId, newNodeIdToOriginalVecDeclId, outputVectorNopMap);
  } else {
    ret = checkAndCollectSpecialPartIOValues(g, newNodeIdToDepNodeId, newNodeIdToOriginalVecDeclId);
  }

  // double check. can be removed
  uint32_t numOpsAtFirstLevel = 0;
  if (isRegularPart) {
    for (auto eachVtx: levels[0]) {
      auto vtxIsDummyNop = newNodeIdToDepNodeId.contains(eachVtx);
      if (vtxIsDummyNop) {
        numOpsAtFirstLevel += 1;
      } else {
        auto opCount = g[eachVtx].opCount;
        numOpsAtFirstLevel += opCount;
      }
    }
  } else {
    numOpsAtFirstLevel = specialOps.size();
  }

  assert(outputValues.size() <= 32);
  assert(numOpsAtFirstLevel <= 32);

  return ret;
}

void MicroPart::print() const {
  bool isRegularPart = opType == CGToucanOPName::LUT;

  llvm::dbgs() << "============ Part print ===========\n";
  llvm::dbgs() << "Part line no " << lineno + 1 << "\n";

  if (isRegularPart) {
    size_t level_id = 0;
    for (const auto &eachLevel: levels) {
      llvm::dbgs() << "Level " << level_id << ":\n";
      for (const auto &eachVtx: eachLevel) {
        if (dummyNodes.contains(eachVtx)) {
          llvm::dbgs() << "  Dummy NOP, read from: ";
          assert(nodeToInputVals.contains(eachVtx));
          assert(nodeToInputVals.at(eachVtx).size() == 1);
          nodeToInputVals.at(eachVtx).front().print(llvm::dbgs());
          llvm::dbgs() << "\n";
        } else {
          llvm::dbgs() << "  Regular node: ";
          assert(nodeToOutputVal.contains(eachVtx));
          nodeToOutputVal.at(eachVtx).print(llvm::dbgs());
          llvm::dbgs() << "\n";
        }
      }
      level_id++;
    }
  } else {
    llvm::dbgs() << "Special part has " << nodes.size() << " nodes and " << specialOps.size() << " ops\n";
    for (const auto eachOp: specialOps) {
      eachOp->print(llvm::dbgs());
      llvm::dbgs() << "\n";
    }
  }

  llvm::dbgs() << "----\n";
}