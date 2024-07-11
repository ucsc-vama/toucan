#include "toucan/PartitioningGraph.h"

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
