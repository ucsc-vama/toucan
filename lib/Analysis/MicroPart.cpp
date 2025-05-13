#include "mlir/Support/LLVM.h"
#include "toucan/MicroPartitioner.h"
#include "toucan/PartitioningGraph.h"
#include "toucan/ToucanAttributes.h"
#include "toucan/ToucanOps.h"
#include "toucan/ToucanUtils.h"
#include "llvm/Support/Debug.h"
#include "llvm/Support/ErrorHandling.h"
#include "llvm/Support/raw_ostream.h"


using namespace toucan;

void MicroPart::clear() {
  nodes.clear();
  levels.clear();
  inputValues.clear();
  outputValues.clear();
  nodeToInputVals.clear();
  nodeToOutputVal.clear();
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


void MicroPart::buildSpecialPart(const CGToucanOPName vtxOpName, const mlir::SmallVector<NodeIdAndOpCount> &nodeAllocation) {
  assert(nodes.empty());
  assert(nodeToOpCount.empty());
  totalOpCount = 0;
  levels.clear();
  levels.emplace_back();

  for (auto [nodeId, opCount]: nodeAllocation) {
    assert(!nodes.contains(nodeId));

    nodes.insert(nodeId);
    nodeToOpCount[nodeId] = opCount;
    totalOpCount += opCount;
    levels[0].push_back(nodeId);
  }

  opType = vtxOpName;
  assert(opType != CGToucanOPName::LUT);
  assert(totalOpCount > 0);
  assert(totalOpCount <= 32 && "Number of real ops in a special part cannot exceed hardware limit!");
  updateNodeToLevel();
}

// Check if op level acquired from MicroPartitioner is correct
bool MicroPart::checkAndCollectIOValues(const PartitioningGraph &g, const mlir::DenseMap<uint32_t, uint32_t> &newNodeIdToDepNodeId, const mlir::DenseMap<uint32_t, uint32_t> &newNodeIdToOriginalVecDeclId) {
  assert(!levels.empty() && "Only check and collect IO values if it's loaded");

  inputValues.clear();
  outputValues.clear();
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

      if (opType == CGToucanOPName::LUT) {
        // regular part
        if (!vtxIsDummyNop) {
          auto rawOp = g[eachVtx].op;
          if (!isa<toucan::LUTOp>(rawOp)) {
            rawOp->print(llvm::errs());
            llvm::errs() << "\n";
            llvm::errs() << "Op type in graph is " << stringifyCGToucanOPName(g[eachVtx].toucanOpName) << "\n";
          }
          assert(isa<toucan::LUTOp>(rawOp));
        }
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
          // internal node read
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
          // a vecDecl read from outside
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
      bool outputValueUsedOutsideThisPartition = false;
      auto out_edge_range = boost::out_edges(eachVtx, g);
      for (auto ei = out_edge_range.first; ei != out_edge_range.second; ei++) {
        auto edgeTarget = boost::target(*ei, g);
        assert(!allPreviousLevelLUTNodes.contains(edgeTarget) && "Should follow topo order");

        if (!nodes.contains(edgeTarget)) {
          outputValueUsedOutsideThisPartition = true;
        }
      }

      // Save output Value
      if (outputValueUsedOutsideThisPartition) {
        if (!newNodeIdToOriginalVecDeclId.contains(eachVtx)) {
          // regular node
          assert(g[eachVtx].toucanOpName == CGToucanOPName::LUT);
          auto rawOp = g[eachVtx].op;
          assert(rawOp != nullptr);
          auto lutOp = cast<toucan::LUTOp>(rawOp);
          auto resultVal = lutOp.getResult();
          outputValues.insert(resultVal);
        } else {
          // VecDecl. In fact bunch of NOPs
          auto originalVecDeclId = newNodeIdToOriginalVecDeclId.at(eachVtx);
          assert(g[originalVecDeclId].toucanOpName == CGToucanOPName::VecDecl);
          auto rawOp = g[originalVecDeclId].op;
          assert(rawOp != nullptr);
          auto vecDeclOp = cast<toucan::DefVectorOp>(rawOp);
          auto resultVal = vecDeclOp.getResult();
          outputValues.insert(resultVal);
          assert(!nodeToOutputVal.contains(eachVtx));
          nodeToOutputVal[eachVtx] = resultVal;
        }
      }
    }

    allPreviousLevelLUTNodes.insert(currentLevelNodes.begin(), currentLevelNodes.end());
  }

  // collect output values

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
  return true;
}

