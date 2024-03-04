#include "circt/Dialect/HW/HWTypes.h"
#include "circt/Support/LLVM.h"
#include "circt/Dialect/Comb/CombDialect.h"
#include "circt/Dialect/Comb/CombOps.h"
#include "circt/Dialect/Seq/SeqOps.h"

#include "mlir/Pass/AnalysisManager.h"
#include "toucan/ToucanAnalysis.h"
#include "toucan/ToucanOps.h"

#include "llvm/Support/Casting.h"
#include "llvm/Support/Debug.h"
#include "llvm/Support/ErrorHandling.h"
#include <cstdint>
#include <functional>
#include <utility>

using namespace toucan;

using namespace mlir;
using namespace llvm;
using namespace circt;


bool DesignGraph::opShouldRemoveInGraph(mlir::Operation *op) {
  if (isa<toucan::DefRegOp>(op)
  || isa<toucan::DefMemOp>(op)) {
      return true;
  }
  return false;
}

static void mergeVerticies(uint32_t dst, mlir::SmallVector<uint32_t> &toMerge, PartitioningGraph &g) {
  // Merge!
  // update edge
  for (auto vtxToMerge: toMerge) {
    auto out_edges = boost::out_edges(vtxToMerge, g);
    for(auto ei = out_edges.first; ei != out_edges.second; ++ei) {
      auto target = boost::target(*ei, g);
      assert(target != dst);
      boost::add_edge(dst, target, g);
    }
    auto in_edges = boost::in_edges(vtxToMerge, g);
    for (auto ei = in_edges.first; ei != in_edges.second; ++ei) {
      auto source = boost::source(*ei, g);
      if (source != dst) {
        boost::add_edge(source, dst, g);
      }
    }
  }
  // update weight
  g[dst].weight += toMerge.size();
}

static CGToucanOPName getOpName(Operation* op) {
  if (isa<toucan::ConstantOp>(op) || isa<toucan::DefConstVectorOp>(op)) return CGToucanOPName::ConstDecl;
  if (isa<toucan::LUTOp>(op)) return CGToucanOPName::LUT;
  if (isa<toucan::DefVectorOp>(op)) return CGToucanOPName::VecRead;
  if (isa<toucan::PrintOp>(op)) return CGToucanOPName::Print;
  if (isa<toucan::StopOp>(op)) return CGToucanOPName::Stop;
  if (isa<toucan::RegReadOp>(op)) return CGToucanOPName::RegRead;
  if (isa<toucan::RegWriteOp>(op)) return CGToucanOPName::RegWrite;
  if (isa<toucan::MemReadOp>(op)) return CGToucanOPName::MemRead;
  if (isa<toucan::MemWriteOp>(op)) return CGToucanOPName::MemWrite;
  llvm_unreachable("?");
}

DesignGraph::DesignGraph(Operation *op, AnalysisManager &am) {
  // build graph
  auto isToucan4B = am.getAnalysis<IsLegalToucan4B>();
  assert(isToucan4B.isToucanOnly && "Must be flatten toucan design");

  auto modOp = cast<ModuleOp>(op);
  auto ops = modOp.getOps();
  auto numOps = std::distance(ops.begin(), ops.end());
  // assert(numOps < UINT32_MAX && "Too many nodes!");

  PartitioningGraph rawGraph;

  mlir::DenseMap<Operation*, uint32_t> rawOpToId;
  rawOpToId.reserve(numOps);

  uint32_t vertexIdCounter = 0;
  // add all vertecies
  for (auto &stmt: modOp.getOps()) {
    // a new node

    PartitioningGraphNodeProperty vp;
    vp.op = &stmt;
    vp.weight = 1;
    vp.toucanOpName = getOpName(&stmt);
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

  mlir::DenseSet<uint32_t> vtxToRemove;



  mlir::SmallVector<uint32_t> vecReaders;
  mlir::SmallVector<uint32_t> memWriters;

  for (uint32_t vtxId = 0; vtxId < boost::num_vertices(rawGraph); vtxId++) {
    auto rawOp = rawGraph[vtxId].op;

    if (auto vecDeclOp = dyn_cast<toucan::DefVectorOp>(rawOp)) {
      // Merge all vector readers with vector decl. Const vecs are not affected
      for (auto userOp: vecDeclOp->getUsers()) {
        if (auto vecReadOp = dyn_cast<toucan::VectorReadOp>(userOp)) {
          auto vecReaderVtxId = rawOpToId[vecReadOp];
          vecReaders.push_back(vecReaderVtxId);
        } else {
          assert(false && "Vector used by node other than toucan::VectorReadOp");
        }
      }

      mergeVerticies(vtxId, vecReaders, rawGraph);
      vtxToRemove.insert(vecReaders.begin(), vecReaders.end());
      vecReaders.clear();
    } else if (auto memDeclOp = dyn_cast<toucan::DefMemOp>(rawOp)) {
      // Merge multiple writers of a same memory into 1
      for (auto userOp: memDeclOp->getUsers()) {
        if (auto memWriteOp = dyn_cast<toucan::MemWriteOp>(userOp)) {
          // a new write port
          auto writerVtxId = rawOpToId[memWriteOp];
          memWriters.push_back(writerVtxId);
        }
      }

      if (memWriters.size() > 1) {
        // A memory with multiple writer. Merge.
        auto topVtxId = memWriters.back();
        memWriters.pop_back();
        mergeVerticies(topVtxId, memWriters, rawGraph);
        vtxToRemove.insert(memWriters.begin(), memWriters.end());
      }

      memWriters.clear();
    } 
    // else if (auto constOp = dyn_cast<toucan::ConstantOp>(rawOp)) {
    //   // save
    //   constOps.push_back(&constOp);
    // } else if (auto constVecOp = dyn_cast<toucan::DefConstVectorOp>(rawOp)) {
    //   // save
    //   constVecOps.push_back(&constVecOp);
    // }
    
    if (opShouldRemoveInGraph(rawOp)) {
      // Remove all defmem, defreg, const, const vec
      vtxToRemove.insert(vtxId);
    }

  }


  // Build new graph without removed nodes, since removing them is expensive
  SmallVector<uint32_t> oldIdToNewId;
  oldIdToNewId.reserve(rawOpToId.size());
  for (uint32_t vtxId = 0; vtxId < boost::num_vertices(rawGraph); vtxId++) {
    if (vtxToRemove.contains(vtxId)) {
      oldIdToNewId.push_back(UINT32_MAX);
    } else {
      // This vtx should copy to new graph
      auto vp = rawGraph[vtxId];
      auto newVtxId = boost::add_vertex(vp, g);
      opToId[vp.op] = newVtxId;
      oldIdToNewId.push_back(newVtxId);
    }
  }
  assert(oldIdToNewId.size() == boost::num_vertices(rawGraph));

  // copy edges

  auto rawEdges = boost::edges(rawGraph);
  for (auto ei = rawEdges.first; ei != rawEdges.second; ++ei) {
    auto edgeSource = boost::source(*ei, rawGraph);
    auto edgeTarget = boost::target(*ei, rawGraph);

    auto newEdgeSource = oldIdToNewId[edgeSource];
    auto newEdgeTarget = oldIdToNewId[edgeTarget];

    if ((newEdgeSource != UINT32_MAX) && (newEdgeTarget != UINT32_MAX)) {
      // Valid
      boost::add_edge(newEdgeSource, newEdgeTarget, g);
    }
  }


  llvm::dbgs() << "Raw graph has " << boost::num_vertices(rawGraph) << " vertices and " << boost::num_edges(rawGraph) << " edges\n";

  llvm::dbgs() << "After removing and merging, graph has " << boost::num_vertices(g) << " vertices and " << boost::num_edges(g) << " edges\n";
}