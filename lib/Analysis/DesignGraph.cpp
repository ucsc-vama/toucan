#include "circt/Dialect/HW/HWTypes.h"
#include "circt/Support/LLVM.h"
#include "circt/Dialect/Comb/CombDialect.h"
#include "circt/Dialect/Comb/CombOps.h"
#include "circt/Dialect/Seq/SeqOps.h"

#include "mlir/IR/Value.h"
#include "mlir/Pass/AnalysisManager.h"
#include "toucan/PartitioningGraph.h"
#include "toucan/ToucanAnalysis.h"
#include "toucan/ToucanOps.h"

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
  if (isa<toucan::DefRegOp>(op)
  || isa<toucan::DefMemOp>(op)) {
      return true;
  }
  return false;
}

#ifdef REPCUT_WEIGHT_BALANCE_SMEM_USAGE
static uint32_t getOpWeight(Operation* op) {
  if (isa<toucan::RegReadOp>(op)) return 10;

  // terminal ops: their input values are alive to the last. Don't produce any value
  if (isa<toucan::RegWriteOp>(op)) return 0;
  if (isa<toucan::MemWriteOp>(op)) return 0;
  if (isa<toucan::PrintOp>(op)) return 0;
  if (isa<toucan::StopOp>(op)) return 0;
  // Consts and const vecs don't have place in smem
  if (isa<toucan::ConstantOp>(op)) return 0;
  if (isa<toucan::DefConstVectorOp>(op)) return 0;
  // Static segment read ops are just pointer to certain place of vector value
  if (isa<toucan::StaticVectorSegmentReadOp>(op)) return 0;

  // vectors should be placed in smem, with weight of vecSize (*10 to avoid floating point)
  if (auto vecDeclOp = dyn_cast<toucan::DefVectorOp>(op)) {
    auto vecSize = vecDeclOp.getHandle().getType().getLength();
    return vecSize * 10;
  }
  if (auto vecArithOp = dyn_cast<toucan::VectorArithOp>(op)) {
    auto resultVecSize = vecArithOp.getResult().getType().getLength();
    return resultVecSize * 10;
  }

  // regular values. weight 0.5
  return 1;
}
#endif

#ifdef REPCUT_WEIGHT_BALANCE_SIM_SPEED
static uint32_t getOpWeight(Operation* op) {
  if (isa<toucan::LUTOp>(op)) return 10;
  if (auto vecDeclOp = dyn_cast<toucan::DefVectorOp>(op)) {
    auto vecSize = vecDeclOp.getHandle().getType().getLength();
    return vecSize * 7;
  }
  if (isa<toucan::RegReadOp>(op)) return 10;

  // terminal ops: their input values are alive to the last. Don't produce any value
  if (isa<toucan::RegWriteOp>(op)) return 5;
  if (isa<toucan::MemWriteOp>(op)) return 200;
  if (isa<toucan::PrintOp>(op)) return 5;
  if (isa<toucan::StopOp>(op)) return 5;
  // Consts and const vecs don't have place in smem
  if (isa<toucan::ConstantOp>(op)) return 0;
  if (isa<toucan::DefConstVectorOp>(op)) return 0;
  // Static segment read ops are just pointer to certain place of vector value
  if (isa<toucan::StaticVectorSegmentReadOp>(op)) return 0;
  if (isa<toucan::MemReadOp>(op)) return 200;
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
  if (isa<toucan::VectorReadOp>(op)) return 10;

  return 10;
}
#endif

static uint32_t getOpCount(Operation* op) {
  if (auto vecDeclOp = dyn_cast<toucan::DefVectorOp>(op)) {
    auto vecSize = vecDeclOp.getHandle().getType().getLength();
    return vecSize;
  }
  return 1;
}

static CGToucanOPName getOpName(Operation* op) {
  if (isa<toucan::ConstantOp>(op) || isa<toucan::DefConstVectorOp>(op)) return CGToucanOPName::ConstDecl;
  if (isa<toucan::LUTOp>(op)) return CGToucanOPName::LUT;
  if (isa<toucan::DefVectorOp>(op)) return CGToucanOPName::VecDecl;
  if (isa<toucan::VectorReadOp>(op)) return CGToucanOPName::VecRead;
  if (isa<toucan::PrintOp>(op)) return CGToucanOPName::Print;
  if (isa<toucan::StopOp>(op)) return CGToucanOPName::Stop;
  if (isa<toucan::RegReadOp>(op)) return CGToucanOPName::RegRead;
  if (isa<toucan::RegWriteOp>(op)) return CGToucanOPName::RegWrite;
  if (isa<toucan::MemReadOp>(op)) return CGToucanOPName::MemRead;
  if (isa<toucan::MemWriteOp>(op)) return CGToucanOPName::MemWrite;
  if (isa<toucan::VectorLogicOp>(op)) return CGToucanOPName::VecLogic;
  if (isa<toucan::VectorArithOp>(op)) return CGToucanOPName::VecArith;
  if (isa<toucan::StaticVectorSegmentReadOp>(op)) return CGToucanOPName::VecStaticRead;

  if (isa<toucan::DefRegOp>(op)) return CGToucanOPName::Dummy_DefReg;
  if (isa<toucan::DefMemOp>(op)) return CGToucanOPName::Dummy_DefMem;

  op->print(llvm::dbgs());
  llvm_unreachable("What's this op?");
  return CGToucanOPName::ShouldNotAppear;
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
    vp.weight = getOpWeight(&stmt);
    vp.toucanOpName = getOpName(&stmt);
    assert(vp.toucanOpName != CGToucanOPName::ShouldNotAppear);
    vp.opCount = getOpCount(&stmt);
    vp.exchangeValId = 0;
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

  mlir::DenseSet<mlir::Value> vecHandled_VecRead;
  mlir::DenseSet<mlir::Value> vecHandled_VecStaticRead;

  // Note: Merge some nodes
  for (uint32_t vtxId = 0; vtxId < boost::num_vertices(rawGraph); vtxId++) {
    auto rawOp = rawGraph[vtxId].op;

    // Assertion: All VecArith should followed by StaticVectorSegmentReadOp
    if (auto vecArithOp = dyn_cast<toucan::VectorArithOp>(rawOp)) {
      auto vecUserEdges = boost::out_edges(vtxId, rawGraph);
      for (auto ei = vecUserEdges.first; ei != vecUserEdges.second; ei++) {
        auto userVtxId = boost::target(*ei, rawGraph);
        auto userOpName = rawGraph[userVtxId].toucanOpName;
        assert(userOpName == CGToucanOPName::VecStaticRead);
      }

      for (auto userOp: vecArithOp->getUsers()) {
        assert(isa<toucan::StaticVectorSegmentReadOp>(userOp));
      }
    }

    if (auto vecReadOp = dyn_cast<toucan::VectorReadOp>(rawOp)) {
      // vector read. merge all into 1
      auto vecHandle = vecReadOp.getHandle();
      if (!vecHandled_VecRead.contains(vecHandle)) {
        // a New vector
        vecReaders.clear();
        for (auto userOp: vecHandle.getUsers()) {
          assert(isa<toucan::VectorReadOp>(userOp));
          auto vecReaderVtxId = rawOpToId[userOp];
          vecReaders.push_back(vecReaderVtxId);
        }

        if (vecReaders.size() > 1) {
          // Merge readers
          auto topVtxId = vecReaders.back();
          vecReaders.pop_back();
          rawGraph.mergeVerticies(topVtxId, vecReaders, true);
          vtxToRemove.insert(vecReaders.begin(), vecReaders.end());
        }

        vecHandled_VecRead.insert(vecHandle);
      }
    } else if (auto vecSegReadOp = dyn_cast<toucan::StaticVectorSegmentReadOp>(rawOp)) {
      // Merge StaticVectorSegmentReadOp with its vector producing op (which should be a VectorArithOp)
      // After this, no StaticVectorSegmentReadOp should appear in the graph
      auto vecHandle = vecSegReadOp.getHandle();

      if (!vecHandled_VecStaticRead.contains(vecHandle)) {
        // Producing op should be a VectorArith
        assert(isa<toucan::VectorArithOp>(vecHandle.getDefiningOp()));

        // a new vec
        // collect all StaticVectorSegmentReadOp that share same vec
        vecReaders.clear();
        for (auto userOp: vecHandle.getUsers()) {
          assert(isa<toucan::StaticVectorSegmentReadOp>(userOp));
          auto vecReaderVtxId = rawOpToId[userOp];
          vecReaders.push_back(vecReaderVtxId);
        }
        assert(vecReaders.size() > 0);

        // Merge them with vector producing op, which is a VectorArithOp
        // Don't increase op count, as VecStaticRead is not a real op
        auto vecDeclNodeId = rawOpToId[vecHandle.getDefiningOp()];
        rawGraph.mergeVerticies(vecDeclNodeId, vecReaders, false);
        vtxToRemove.insert(vecReaders.begin(), vecReaders.end());

        vecHandled_VecStaticRead.insert(vecHandle);
      }
    } else if (auto memDeclOp = dyn_cast<toucan::DefMemOp>(rawOp)) {
      // Merge multiple writers of a same memory into 1
      memWriters.clear();
      for (auto userOp: memDeclOp->getUsers()) {
        if (auto memWriteOp = dyn_cast<toucan::MemWriteOp>(userOp)) {
          // a new write port
          auto writerVtxId = rawOpToId[memWriteOp];
          assert(isa<toucan::MemWriteOp>(rawGraph[writerVtxId].op));
          memWriters.push_back(writerVtxId);
        }
      }

      if (memWriters.size() > 1) {
        // A memory with multiple writer. Merge.
        auto topVtxId = memWriters.back();
        memWriters.pop_back();
        rawGraph.mergeVerticies(topVtxId, memWriters, true);
        vtxToRemove.insert(memWriters.begin(), memWriters.end());
      }
    } 
    // else if (auto constOp = dyn_cast<toucan::ConstantOp>(rawOp)) {
    //   // save
    //   constOps.push_back(&constOp);
    // } else if (auto constVecOp = dyn_cast<toucan::DefConstVectorOp>(rawOp)) {
    //   // save
    //   constVecOps.push_back(&constVecOp);
    // }
    if (auto regDeclOp = dyn_cast<toucan::DefRegOp>(rawOp)) {
      auto regVal = regDeclOp.getHandle();
      regs.insert(regVal);
    }
    
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


  llvm::outs() << "Raw graph has " << boost::num_vertices(rawGraph) << " vertices and " << boost::num_edges(rawGraph) << " edges\n";

  llvm::outs() << "After removing and merging, graph has " << boost::num_vertices(g) << " vertices and " << boost::num_edges(g) << " edges\n";
}