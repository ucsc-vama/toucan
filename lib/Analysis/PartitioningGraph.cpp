#include "toucan/PartitioningGraph.h"
#include <boost/graph/depth_first_search.hpp>

using namespace toucan;
using namespace llvm;

void toucan::PartitioningGraph::mergeVerticies(uint32_t dst, const mlir::SmallVector<uint32_t> &toMerge, bool increseOpCount) {
  // update edge

  // avoid duplicated edges
  mlir::DenseSet<uint32_t> inVtxes;
  mlir::DenseSet<uint32_t> outVtxes;

  uint32_t mergedOpCount = 0;
  auto &&g = *this;

  // Note: it's possible if parallel edge exists
  // For example, a vector using multiple slots of same value
  auto dst_out_edges = boost::out_edges(dst, g);
  for (auto ei = dst_out_edges.first; ei != dst_out_edges.second; ++ei) {
    auto target = boost::target(*ei, g);
    // assert(!outVtxes.contains(target));
    outVtxes.insert(target);
  }
  auto dst_in_edges = boost::in_edges(dst, g);
  for (auto ei = dst_in_edges.first; ei != dst_in_edges.second; ++ei) {
    auto source = boost::source(*ei, g);
    // assert(!inVtxes.contains(source));
    inVtxes.insert(source);
  }

  mlir::DenseSet<uint32_t> edgesToRemove;

  for (auto vtxToMerge: toMerge) {
    edgesToRemove.clear();
    mergedOpCount += g[vtxToMerge].opCount;
    auto out_edges = boost::out_edges(vtxToMerge, g);
    for(auto ei = out_edges.first; ei != out_edges.second; ++ei) {
      auto target = boost::target(*ei, g);
      edgesToRemove.insert(target);
      if ((target != dst)) {
        boost::add_edge(dst, target, g);
        outVtxes.insert(target);
      }
    }
    for (const auto &target: edgesToRemove) {
      boost::remove_edge(vtxToMerge, target, g);
    }
    edgesToRemove.clear();

    auto in_edges = boost::in_edges(vtxToMerge, g);
    for (auto ei = in_edges.first; ei != in_edges.second; ++ei) {
      auto source = boost::source(*ei, g);
      edgesToRemove.insert(source);
      if ((source != dst)) {
        boost::add_edge(source, dst, g);
        inVtxes.insert(source);
      }
    }
    for (const auto &source: edgesToRemove) {
      boost::remove_edge(source, vtxToMerge, g);
    }

    // Remove all edges, but don't remove nodes since it's expensive
    assert(boost::in_degree(vtxToMerge, g) == 0);
    assert(boost::out_degree(vtxToMerge, g) == 0);
  }

  // update op count
  if (increseOpCount) g[dst].opCount += mergedOpCount;
}



// Custom visitor to detect cycles
struct cycle_detector : public boost::dfs_visitor<> {
    bool& has_cycle;

    cycle_detector(bool& cycle) : has_cycle(cycle) {}

    template <typename Edge, typename Graph>
    void back_edge(Edge, const Graph&) {
        has_cycle = true;  // Cycle detected
    }
};

bool toucan::PartitioningGraph::hasCycle() const {
  bool has_cycle = false;
  cycle_detector vis(has_cycle);

  auto &&graph = *this;
  boost::depth_first_search(graph, visitor(vis));

  return has_cycle;
}
