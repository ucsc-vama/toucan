#include "mlir/Support/LLVM.h"
#include "toucan/MicroPartitioner.h"
#include "toucan/PartitioningGraph.h"


using namespace toucan;

void MicroPart::clear() {
  nodes.clear();
  levels.clear();
  inputValues.clear();
  outputValues.clear();
}

void MicroPart::buildRegularLUTPart(const mlir::SmallVector<uint32_t> &newNodes) {
  assert(nodes.empty());
  assert(nodeToOpCount.empty());

  mlir::DenseSet<uint32_t> nodeSet;
  for (auto n: newNodes) {
    nodeSet.insert(n);
  }
  assert(nodeSet.size() == newNodes.size());

  // Save new nodes
  nodes.assign(newNodes);

  // For LUT part, each node is just 1 op
  for (auto n: newNodes) {
    nodeToOpCount[n] = 1;
  }

  opType = CGToucanOPName::LUT;
  totalOpCount = newNodes.size();
}


void MicroPart::buildSpecialPart(const CGToucanOPName vtxOpName, const mlir::SmallVector<NodeIdAndOpCount> &nodeAllocation) {
  assert(nodes.empty());
  assert(nodeToOpCount.empty());
  totalOpCount = 0;

  mlir::DenseSet<uint32_t> nodeSet;
  for (auto [nodeId, opCount]: nodeAllocation) {
    assert(!nodeSet.contains(nodeId));
    nodeSet.insert(nodeId);

    nodes.push_back(nodeId);
    nodeToOpCount[nodeId] = opCount;
    totalOpCount += opCount;
  }

  opType = vtxOpName;
  assert(totalOpCount > 0);
  assert(totalOpCount <= 32 && "Number of real ops in a special part cannot exceed hardware limit!");
}

