#include "mlir/Support/LLVM.h"
#include "toucan/MicroPartitioner.h"
#include "toucan/PartitioningGraph.h"
#include "toucan/ToucanAttributes.h"
#include "toucan/ToucanOps.h"
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


bool MicroPart::checkAndCollectRegularPartIOValues(const PartitioningGraph &g, const mlir::DenseSet<uint32_t> &allNodes, const mlir::DenseMap<uint32_t, uint32_t> &newNodeIdToDepNodeId, const mlir::DenseMap<uint32_t, uint32_t> &newNodeIdToOriginalVecDeclId) {
  assert(opType == toucan::CGToucanOPName::LUT);
  assert(!levels.empty() && "Only check and collect IO values if it's loaded");

  inputValues.clear();
  outputValues.clear();
  outputValueSet.clear();
  nodeToInputVals.clear();
  nodeToOutputVal.clear();

  mlir::DenseSet<uint32_t> allPreviousLevelLUTNodes;
  allPreviousLevelLUTNodes.reserve(nodes.size());
  allPreviousLevelLUTNodes.insert(levels[0].begin(), levels[0].end());
  assert(!allPreviousLevelLUTNodes.empty());



  // Check correctness and collect input value
  for (size_t levelId = 0; levelId < levels.size(); levelId++) {
    assert(!levels[levelId].empty());
    const auto &currentLevelNodes = levels[levelId];
    for (auto eachVtx: currentLevelNodes) {
      assert(nodes.contains(eachVtx));
      auto vtxIsDummyNop = newNodeIdToDepNodeId.contains(eachVtx);

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
      auto inDegree = boost::in_degree(eachVtx, g);
      if (inDegree == 0 && !vtxIsDummyNop) {
        llvm::errs() << "In micro part level " << levelId << ", vtx " << eachVtx << " has in degree == 0\n";
        auto vtxRawOp = g[eachVtx].op;
        vtxRawOp->print(llvm::errs());
        llvm::errs() << "\n";
        // llvm::errs() << stringifyCGToucanOPName(opType) << "\n";
        return false;
      }


      // Check if is levelized correctly, also collect input val

      if (vtxIsDummyNop) {
        auto depVtx = newNodeIdToDepNodeId.at(eachVtx);
        auto depOp = g[depVtx].op;

        if (nodes.contains(depVtx)) {
          // internal node read and write to vector. Just check correctness
          auto depVtxLevel = nodeToLevel[depVtx];

          if (depVtxLevel + 1 != levelId) {
            llvm::errs() << "A dummy node " << eachVtx << " in a micro partition does NOT depends on input from previous level!\n";
            depOp->print(llvm::errs());
            llvm::errs() << "\n";
            auto originalVecDeclId = newNodeIdToOriginalVecDeclId.at(eachVtx);
            assert(g[originalVecDeclId].toucanOpName == CGToucanOPName::VecDecl);
            auto rawOp = g[originalVecDeclId].op;
            rawOp->print(llvm::errs());
            llvm::errs() << "\n";

            return false;
          }
        } else {
          // a vecDecl reads element from outside
          if (isa<toucan::DefVectorOp>(depOp)) {
            llvm::dbgs() << "Op\n";
            auto rawOp = g[newNodeIdToOriginalVecDeclId.at(eachVtx)].op;
            rawOp->print(llvm::dbgs());
            llvm::dbgs() << "\nDepends on external vector:\n";
            depOp->print(llvm::dbgs());
            llvm::dbgs() << "\n";
          }
          assert(!isa<toucan::DefVectorOp>(depOp));
          auto depValue = getOpResultValue(depOp);
          if (!isa<toucan::ConstantOp>(depValue.getDefiningOp())) {
            // Ignore const
            inputValues.insert(depValue);
            if (nodeToInputVals.contains(eachVtx)) {
              nodeToInputVals[eachVtx].push_back(depValue);
            } else {
              nodeToInputVals[eachVtx] = {depValue};
            }
          } else {
            // This input value of a vector is constant.
            // Do nothing and it should not be considered as input (since we have a const pool)
          }
        }
      } else {
        // regular node
        bool dependsOnPreviousLevel = false;
        auto in_edge_range = boost::in_edges(eachVtx, g);
        for (auto ei = in_edge_range.first; ei != in_edge_range.second; ei++) {
          auto depVtx = boost::source(*ei, g);
          auto depOp = g[depVtx].op;
  
          // ignore constant values. Accessing them don't need touch shared mem.
          if (isa<toucan::ConstantOp>(depOp)) {
            continue;
          }
  
          // Outside the current partition
          if (!allPreviousLevelLUTNodes.contains(depVtx)) {
            // Depends on a value outside of this partition. This is a new input value
            assert(!nodes.contains(depVtx) && "Part should be levelized (topo order)");
            if (isa<toucan::DefVectorOp>(depOp)) {
              llvm::dbgs() << "Op\n";
              auto rawOp = g[eachVtx].op;
              rawOp->print(llvm::dbgs());
              llvm::dbgs() << "\nDepends on external vector:\n";
              depOp->print(llvm::dbgs());
              llvm::dbgs() << "\n";
            }
            assert(!isa<toucan::DefVectorOp>(depOp));
            auto depValue = getOpResultValue(depOp);
            if (!isa<toucan::ConstantOp>(depValue.getDefiningOp())) {
              // Ignore const
              inputValues.insert(depValue);
              if (nodeToInputVals.contains(eachVtx)) {
                nodeToInputVals[eachVtx].push_back(depValue);
              } else {
                nodeToInputVals[eachVtx] = {depValue};
              }
            } else {
              // This input value of a vector is constant.
              // Do nothing and it should not be considered as input (since we have a const pool)
            }
          }
  
          if (nodeToLevel.contains(depVtx)) {
            if (nodeToLevel[depVtx] + 1 == levelId) {
              dependsOnPreviousLevel = true;
            }
          } else {
            // Should be an input edge from outside the part
            assert(!nodes.contains(depVtx));
          }
        }
  
        // Not depends on previous level and not the first level
        if (!dependsOnPreviousLevel && levelId != 0) {
          llvm::errs() << "Node " << eachVtx << " in a micro partition does NOT depends on input from previous level!\n";
          llvm::errs() << "Node " << eachVtx << " at level " << levelId << " (total " << levels.size() << "), out degree " << boost::out_degree(eachVtx, g) << ", in degree " << boost::in_degree(eachVtx, g) << "\n";

          g[eachVtx].op->print(llvm::errs());

          llvm::errs() << "\n";
  
          auto in_edge_range = boost::in_edges(eachVtx, g);
          for (auto ei = in_edge_range.first; ei != in_edge_range.second; ei++) {
            auto edgeSource = boost::source(*ei, g);
            llvm::errs() << "Dep node " << edgeSource << " at level " << nodeToLevel[edgeSource] << "\n";
          }
          return false;
        }
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
        bool outputValueUsedOutsideThisPartition = false;
        auto out_edge_range = boost::out_edges(eachVtx, g);
        for (auto ei = out_edge_range.first; ei != out_edge_range.second; ei++) {
          auto edgeTarget = boost::target(*ei, g);
          assert(!allPreviousLevelLUTNodes.contains(edgeTarget) && "Should follow topo order");

          // result used by outside of this partition
          if (!nodes.contains(edgeTarget)) {
            // result is not used by dummy vecdecl in same partition
            bool targetIsVecDecl = isa<toucan::DefVectorOp>(g[edgeTarget].op);
            bool targetInCurrentRepCutPartition = allNodes.contains(edgeTarget);
            if ((!targetIsVecDecl) && targetInCurrentRepCutPartition) {
              outputValueUsedOutsideThisPartition = true;
            }
          }
        }
        // Save output Value
        if (outputValueUsedOutsideThisPartition) {
          assert(!vtxIsDummyNop);
          if (!newNodeIdToOriginalVecDeclId.contains(eachVtx)) {
            // regular node
            assert(g[eachVtx].toucanOpName == CGToucanOPName::LUT);
            auto rawOp = g[eachVtx].op;
            assert(rawOp != nullptr);
            auto lutOp = cast<toucan::LUTOp>(rawOp);
            auto resultVal = lutOp.getResult();
            outputValues.push_back(resultVal);
            assert(!outputValueSet.contains(resultVal));
            outputValueSet.insert(resultVal);
            nodeToOutputVal[eachVtx] = resultVal;
          } else {
            // VecDecl. In fact bunch of NOPs
            assert(false && "Should not reach here");
            // auto originalVecDeclId = newNodeIdToOriginalVecDeclId.at(eachVtx);
            // assert(g[originalVecDeclId].toucanOpName == CGToucanOPName::VecDecl);
            // auto rawOp = g[originalVecDeclId].op;
            // assert(rawOp != nullptr);
            // auto vecDeclOp = cast<toucan::DefVectorOp>(rawOp);
            // auto resultVal = vecDeclOp.getResult();
            // assert(!outputValues.contains(resultVal));
            // outputValues.insert(resultVal);
            // assert(!nodeToOutputVal.contains(eachVtx));
            // nodeToOutputVal[eachVtx] = resultVal;
          }
        } else {
          assert(g[eachVtx].op != nullptr);
          assert(!isa<toucan::DefVectorOp>(g[eachVtx].op));
        }
      }

    }

    allPreviousLevelLUTNodes.insert(currentLevelNodes.begin(), currentLevelNodes.end());
  }

  // collect output values

  /*
  for (auto vtx: levels.back()) {
    // Each node at last level should only be used outside of the part
    auto out_edge_range = boost::out_edges(vtx, g);
    for (auto ei = out_edge_range.first; ei != out_edge_range.second; ei++) {
      auto edgeTarget = boost::target(*ei, g);
      assert(!nodes.contains(edgeTarget) && "No nodes should have edge pointing outside of current partition unless last level");
    }

    if (!newNodeIdToOriginalVecDeclId.contains(vtx)) {
      // regular node
      assert(g[vtx].toucanOpName == CGToucanOPName::LUT);
      auto rawOp = g[vtx].op;
      assert(rawOp != nullptr);
      auto lutOp = cast<toucan::LUTOp>(rawOp);
      auto resultVal = lutOp.getResult();
      outputValues.insert(resultVal);
    } else {
      // VecDecl. In fact bunch of NOPs
      auto originalVecDeclId = newNodeIdToOriginalVecDeclId.at(vtx);
      assert(g[originalVecDeclId].toucanOpName == CGToucanOPName::VecDecl);
      auto rawOp = g[originalVecDeclId].op;
      assert(rawOp != nullptr);
      auto vecDeclOp = cast<toucan::DefVectorOp>(rawOp);
      auto resultVal = vecDeclOp.getResult();
      outputValues.insert(resultVal);
    }
  }
    */
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



bool MicroPart::checkAndCollectIOValues(const PartitioningGraph &g, const mlir::DenseSet<uint32_t> &allNodes, const mlir::DenseMap<uint32_t, uint32_t> &newNodeIdToDepNodeId, const mlir::DenseMap<uint32_t, uint32_t> &newNodeIdToOriginalVecDeclId) {
  bool ret;
  bool isRegularPart = opType == CGToucanOPName::LUT;
  if (isRegularPart) {
    ret = checkAndCollectRegularPartIOValues(g, allNodes, newNodeIdToDepNodeId, newNodeIdToOriginalVecDeclId);
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