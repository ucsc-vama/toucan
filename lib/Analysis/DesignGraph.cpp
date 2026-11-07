#include "circt/Dialect/Comb/CombDialect.h"
#include "circt/Dialect/Comb/CombOps.h"
#include "circt/Dialect/HW/HWTypes.h"
#include "circt/Dialect/Seq/SeqOps.h"
#include "circt/Support/LLVM.h"

#include "mlir/IR/Value.h"
#include "mlir/Pass/AnalysisManager.h"
#include "toucan/CGToucanOpName.h"
#include "toucan/PartitioningGraph.h"
#include "toucan/ToucanAnalysis.h"
#include "toucan/ToucanOps.h"

#include "llvm/ADT/STLExtras.h"
#include "llvm/Support/Casting.h"
#include "llvm/Support/Debug.h"
#include "llvm/Support/ErrorHandling.h"
#include <cstdint>
#include <functional>
#include <utility>

#include "toucan/ToucanConfigs.h"

using namespace toucan;

using namespace mlir;
using namespace llvm;
using namespace circt;

bool DesignGraph::opShouldRemoveInGraph(mlir::Operation *op) {
  // vp.opName ==
  // Dummy_DefReg,
  // Dummy_DefMem,
  if (isa<toucan::DefRegOp>(op) || isa<toucan::DefMemOp>(op)) {
    return true;
  }
  return false;
}

static uint32_t getOpWeight(Operation *op) {
  if (isa<toucan::LUTOp>(op))
    return 10;
  if (auto vecDeclOp = dyn_cast<toucan::DefVectorOp>(op)) {
    auto vecSize = vecDeclOp.getHandle().getType().getLength();
    return vecSize * 7;
  }
  if (isa<toucan::RegReadOp>(op))
    return 10;

  // terminal ops: their input values are alive to the last. Don't produce any value
  if (isa<toucan::RegWriteOp>(op))
    return 5;
  if (isa<toucan::MemWriteOp>(op))
    return 200;
  if (isa<toucan::PrintOp>(op))
    return 5;
  if (isa<toucan::StopOp>(op))
    return 5;
  // Consts and const vecs don't have place in smem
  if (isa<toucan::ConstantOp>(op))
    return 0;
  if (isa<toucan::DefConstVectorOp>(op))
    return 0;
  // Static segment read ops are just pointer to certain place of vector value
  if (isa<toucan::StaticVectorSegmentReadOp>(op))
    return 0;
  if (isa<toucan::MemReadOp>(op))
    return 200;
  // if (isa<toucan::VectorArithOp>(op)) return 10;
  if (auto vecLogicOp = dyn_cast<toucan::VectorLogicOp>(op)) {
    auto vecLength = vecLogicOp.getV1().getType().getLength();
    return vecLength * 10;
  }
  if (auto vecArithOp = dyn_cast<toucan::VectorArithOp>(op)) {
    auto vecLength = vecArithOp.getResult().getType().getLength();
    return vecLength * 10;
  }
  // if (isa<toucan::VectorLogicOp>(op)) return 10;
  if (isa<toucan::VectorReadOp>(op))
    return 10;

  return 10;
}

static uint32_t getOpCount(Operation *op) {
  if (auto vecDeclOp = dyn_cast<toucan::DefVectorOp>(op)) {
    auto vecSize = vecDeclOp.getHandle().getType().getLength();
    return vecSize;
  }
  return 1;
}

static CGToucanOPName getOpName(Operation *op) {
  if (isa<toucan::ConstantOp>(op) || isa<toucan::DefConstVectorOp>(op))
    return CGToucanOPName::ConstDecl;
  if (isa<toucan::LUTOp>(op))
    return CGToucanOPName::LUT;
  if (isa<toucan::DefVectorOp>(op))
    return CGToucanOPName::VecDecl;
  if (isa<toucan::VectorReadOp>(op))
    return CGToucanOPName::VecRead;
  if (isa<toucan::PrintOp>(op))
    return CGToucanOPName::Print;
  if (isa<toucan::StopOp>(op))
    return CGToucanOPName::Stop;
  if (isa<toucan::RegReadOp>(op))
    return CGToucanOPName::RegRead;
  if (isa<toucan::RegWriteOp>(op))
    return CGToucanOPName::RegWrite;
  if (isa<toucan::MemReadOp>(op))
    return CGToucanOPName::MemRead;
  if (isa<toucan::MemWriteOp>(op))
    return CGToucanOPName::MemWrite;
  if (isa<toucan::VectorLogicOp>(op))
    return CGToucanOPName::VecLogic;
  if (isa<toucan::VectorArithOp>(op))
    return CGToucanOPName::VecArith;
  if (isa<toucan::StaticVectorSegmentReadOp>(op))
    return CGToucanOPName::VecStaticRead;

  if (isa<toucan::DefRegOp>(op))
    return CGToucanOPName::Dummy_DefReg;
  if (isa<toucan::DefMemOp>(op))
    return CGToucanOPName::Dummy_DefMem;

  op->print(llvm::dbgs());
  llvm_unreachable("What's this op?");
  return CGToucanOPName::ShouldNotAppear;
}

DesignGraph::DesignGraph(Operation *op, AnalysisManager &am) {
  const auto &isToucan4B = am.getAnalysis<IsLegalToucan4B>();
  assert(isToucan4B.isToucanOnly && "Must be flatten toucan design");

  auto modOp = cast<ModuleOp>(op);
  auto numOps = std::distance(modOp.getOps().begin(), modOp.getOps().end());
  assert(numOps < UINT32_MAX && "Too many nodes!");

  SmallVector<Operation *> ops;
  mlir::DenseMap<Operation *, uint32_t> opIds;
  SmallVector<uint32_t> representatives;
  SmallVector<uint32_t> opCounts;
  ops.reserve(numOps);
  opIds.reserve(numOps);
  representatives.reserve(numOps);
  opCounts.reserve(numOps);
  for (auto &stmt : modOp.getOps()) {
    auto id = ops.size();
    opIds[&stmt] = id;
    ops.push_back(&stmt);
    representatives.push_back(id);
    opCounts.push_back(getOpCount(&stmt));
  }

  // The merge groups are disjoint: vector readers, an arithmetic producer with
  // its static segment readers, or writers of one memory. Record their final
  // representatives before allocating any edges. Repeatedly removing edges
  // from a raw graph would rescan high-fanout adjacency lists for every merge.
  auto merge = [&](uint32_t dst, ArrayRef<uint32_t> members, bool increaseOpCount) {
    assert(representatives[dst] == dst && "Merge groups must be disjoint");
    for (auto id : members) {
      assert(id != dst && representatives[id] == id && "Merge groups must be disjoint");
      representatives[id] = dst;
      if (increaseOpCount)
        opCounts[dst] += opCounts[id];
    }
  };

  SmallVector<uint32_t> members;
  mlir::DenseSet<Value> handledVectors;
  for (auto *rawOp : ops) {
    if (auto vecArithOp = dyn_cast<VectorArithOp>(rawOp)) {
      for (auto *user : vecArithOp->getUsers()) {
        assert(isa<StaticVectorSegmentReadOp>(user));
      }
    }

    if (auto vecReadOp = dyn_cast<VectorReadOp>(rawOp)) {
      auto handle = vecReadOp.getHandle();
      if (handledVectors.insert(handle).second) {
        members.clear();
        for (auto *user : handle.getUsers()) {
          assert(isa<VectorReadOp>(user));
          members.push_back(opIds.lookup(user));
        }
        // Preserve the original representative: the last reader in use order.
        auto dst = members.pop_back_val();
        merge(dst, members, true);
      }
    } else if (auto segmentOp = dyn_cast<StaticVectorSegmentReadOp>(rawOp)) {
      auto handle = segmentOp.getHandle();
      if (handledVectors.insert(handle).second) {
        assert(isa<VectorArithOp>(handle.getDefiningOp()));
        members.clear();
        for (auto *user : handle.getUsers()) {
          assert(isa<StaticVectorSegmentReadOp>(user));
          members.push_back(opIds.lookup(user));
        }
        // Segment reads are aliases of the producer's result, not extra work.
        merge(opIds.lookup(handle.getDefiningOp()), members, false);
      }
    } else if (auto memOp = dyn_cast<DefMemOp>(rawOp)) {
      members.clear();
      for (auto *user : memOp->getUsers()) {
        if (isa<MemWriteOp>(user))
          members.push_back(opIds.lookup(user));
      }
      if (!members.empty()) {
        auto dst = members.pop_back_val();
        merge(dst, members, true);
      }
    }

    if (auto regOp = dyn_cast<DefRegOp>(rawOp))
      regs.insert(regOp.getHandle());
  }

  // Retain the original operation order and representative metadata. Only the
  // operation count is accumulated by a merge; its weight remains unchanged.
  SmallVector<uint32_t> finalIds(ops.size(), UINT32_MAX);
  auto numVertices = llvm::count_if(llvm::seq<size_t>(0, ops.size()), [&](size_t id) {
    return representatives[id] == id && !opShouldRemoveInGraph(ops[id]);
  });
  g.reserve_vertices(numVertices);
  opToId.reserve(numVertices);
  for (uint32_t id = 0; id < ops.size(); ++id) {
    auto *stmt = ops[id];
    if (representatives[id] != id || opShouldRemoveInGraph(stmt))
      continue;

    PartitioningGraphNodeProperty vp;
    vp.op = stmt;
    vp.weight = getOpWeight(stmt);
    vp.toucanOpName = getOpName(stmt);
    vp.opCount = opCounts[id];
    vp.exchangeValId = 0;
    vp.mp = nullptr;
    auto vertex = boost::add_vertex(vp, g);
    finalIds[id] = vertex;
    opToId[stmt] = vertex;
  }
  for (uint32_t id = 0; id < ops.size(); ++id)
    finalIds[id] = finalIds[representatives[id]];

  // Emit the contracted dependency graph directly. Keep parallel edges: they
  // represent distinct uses and are significant to downstream partitioning.
  // Edges internal to a merge group disappear, as do declaration dependencies.
  for (uint32_t id = 0; id < ops.size(); ++id) {
    auto source = finalIds[id];
    if (source == UINT32_MAX)
      continue;
    for (auto *user : ops[id]->getUsers()) {
      auto userId = opIds.lookup(user);
      auto target = finalIds[userId];
      // Preserve a representative's existing self-edge. Only the internal
      // edges removed by contraction should disappear, not pre-existing cycles.
      bool retainedSelfEdge = id == userId && representatives[id] == id;
      if (target != UINT32_MAX && (source != target || retainedSelfEdge))
        boost::add_edge(source, target, g);
    }
  }

  llvm::outs() << "Design graph has " << boost::num_vertices(g) << " vertices and " << boost::num_edges(g)
               << " edges\n";

  for (auto vtx : boost::make_iterator_range(boost::vertices(g))) {
    if (g[vtx].toucanOpName == CGToucanOPName::RegRead) {
      assert(boost::out_degree(vtx, g) != 0 && "RegRead result must have a user!");
    }
  }
}
