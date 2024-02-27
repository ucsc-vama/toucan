#include "circt/Dialect/HW/HWTypes.h"
#include "circt/Support/LLVM.h"
#include "circt/Dialect/Comb/CombDialect.h"
#include "circt/Dialect/Comb/CombOps.h"
#include "circt/Dialect/Seq/SeqOps.h"

#include "mlir/Pass/AnalysisManager.h"
#include "toucan/ToucanAnalysis.h"

#include "llvm/Support/Casting.h"
#include "llvm/Support/Debug.h"
#include <cstdint>

using namespace toucan;

using namespace mlir;
using namespace llvm;
using namespace circt;


DesignGraph::DesignGraph(Operation *op, AnalysisManager &am) {
  // build graph
  auto isToucan4B = am.getAnalysis<IsLegalToucan4B>();
  assert(isToucan4B.isToucanOnly && "Must be flatten toucan design");

  auto modOp = cast<ModuleOp>(op);
  auto ops = modOp.getOps();
  auto numOps = std::distance(ops.begin(), ops.end());
  // assert(numOps < UINT32_MAX && "Too many nodes!");

  PartitioningGraph rawGraph;

  mlir::DenseMap<Operation*, uint64_t> rawOpToId;
  rawOpToId.reserve(numOps);

  uint32_t vertexIdCounter = 0;
  // add all vertecies
  for (auto &stmt: modOp.getOps()) {
    // a new node

    PartitioningGraphNodeProperty vp;
    vp.op = &stmt;
    vp.weight = 1;
    auto newVertex = boost::add_vertex(vp, rawGraph);

    rawOpToId[&stmt] = newVertex;

    // Ensure node id is incremental
    assert(newVertex == vertexIdCounter);
    vertexIdCounter++;
  }

  // add all edges
  for (auto &stmt: modOp.getOps()) {
    auto vtxId = rawOpToId[&stmt];
    for (auto user: stmt.getUsers()) {
      auto userVtxId = rawOpToId[user];
      boost::add_edge(vtxId, userVtxId, rawGraph);
    }
  }

  // TODO: Remove all defmem, defreg, const, const_vector

  // TODO: Merge vec def (toucan.vector) and its users

  // TODO: move rawGraph to g, ensure nodeid is incremental (no 'holes' between ids)

  llvm::dbgs() << "Graph has " << boost::num_vertices(g) << " vertices and " << boost::num_edges(g) << " edges\n";
}