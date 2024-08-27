#include "toucan/PartitioningGraph.h"
#include <boost/graph/depth_first_search.hpp>

using namespace toucan;

void toucan::mergeVerticies(uint32_t dst, const mlir::SmallVector<uint32_t> &toMerge, PartitioningGraph &g) {
  // Merge!
  // update edge

  // avoid duplicated edges
  mlir::DenseSet<uint32_t> inVtxes;
  mlir::DenseSet<uint32_t> outVtxes;

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

  for (auto vtxToMerge: toMerge) {
    auto out_edges = boost::out_edges(vtxToMerge, g);
    for(auto ei = out_edges.first; ei != out_edges.second; ++ei) {
      auto target = boost::target(*ei, g);
      if ((target != dst) && (!outVtxes.contains(target))) {
        boost::add_edge(dst, target, g);
        outVtxes.insert(target);
      }
    }
    auto in_edges = boost::in_edges(vtxToMerge, g);
    for (auto ei = in_edges.first; ei != in_edges.second; ++ei) {
      auto source = boost::source(*ei, g);
      if ((source != dst) && (!inVtxes.contains(source))) {
        boost::add_edge(source, dst, g);
        inVtxes.insert(source);
      }
    }
  }
  // update weight
  g[dst].weight += toMerge.size();
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

bool toucan::partitioningGraphHasCycle(const PartitioningGraph &graph) {
  bool has_cycle = false;
  cycle_detector vis(has_cycle);
  boost::depth_first_search(graph, visitor(vis));

  return has_cycle;
}
