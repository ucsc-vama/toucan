
#pragma once

#include "mlir/IR/BuiltinTypes.h"
#include "mlir/IR/Dialect.h"
#include "mlir/IR/Operation.h"

#include "mlir/IR/Value.h"

#include "mlir/Support/LLVM.h"
#include "llvm/ADT/APInt.h"
#include "llvm/ADT/APSInt.h"
#include "llvm/Support/Debug.h"


#include <cstdint>
#include <memory>
#include <optional>

#include "toucan/ToucanAttributes.h"
#include "toucan/ToucanDialect.h"
#include "toucan/ToucanOps.h"
#include "toucan/ToucanTypes.h"
#include "toucan/CGToucanOpName.h"
// #include "toucan/MicroPart.h"

#include <boost/graph/adjacency_list.hpp>
#include <boost/graph/copy.hpp>

#include <unordered_map>
#include <filesystem>




namespace toucan {
  // Forward declaration
  class MicroPart;

  struct PartitioningGraphNodeProperty {
    public:
    // LUTOpName opName;
    mlir::Operation *op;
    // weight is actually number of ops in this node
    uint32_t weight;
    uint32_t opCount;
    uint32_t exchangeValId;
    std::shared_ptr<MicroPart> mp;
    
    CGToucanOPName toucanOpName;
  };


  class PartitioningGraph : public boost::adjacency_list<boost::listS, boost::vecS, boost::bidirectionalS, PartitioningGraphNodeProperty, boost::no_property, boost::no_property, boost::listS> {

    private:

    struct VertexPropertyCopier {
      const PartitioningGraph& src_graph;
      PartitioningGraph& dst_graph;

      VertexPropertyCopier(const PartitioningGraph& src, PartitioningGraph& dst)
        : src_graph(src), dst_graph(dst) {}

      template <typename VertexSrc, typename VertexDst>
      void operator()(const VertexSrc& v_src, VertexDst& v_dst) const {
        dst_graph[v_dst] = src_graph[v_src];
      }
    };

    public:



    void reserve_vertices(std::size_t n) { this->m_vertices.reserve(n); }

    void mergeVerticies(uint32_t dst, const mlir::SmallVector<uint32_t> &toMerge, bool increaseOpCount);

    bool hasCycle() const;

    void copy_from(const PartitioningGraph &g_in) {
      auto &g_out = *this;
      VertexPropertyCopier copier(g_in, g_out);
      boost::copy_graph(g_in, g_out, boost::vertex_copy(copier));
    }
  };



}
