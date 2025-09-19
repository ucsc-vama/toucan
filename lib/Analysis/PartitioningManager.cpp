
#include "toucan/PartitioningManager.h"
#include "toucan/CGToucanOpName.h"
#include "toucan/RepCutPartitioner.h"
#include "mlir/IR/Operation.h"
#include "mlir/IR/PatternMatch.h"
#include "mlir/IR/Value.h"
#include "mlir/Support/LLVM.h"
#include "mlir/Support/LogicalResult.h"
#include "toucan/MicroPartitioner.h"
#include "toucan/PartitioningGraph.h"
#include "toucan/ToucanAnalysis.h"
#include "toucan/ToucanAttributes.h"
#include "toucan/ToucanOps.h"
#include "llvm/Support/Casting.h"
#include "llvm/Support/Debug.h"
#include "llvm/Support/ErrorHandling.h"
#include "llvm/Support/raw_ostream.h"

#include "mlir/IR/Builders.h"
#include "toucan/ToucanTypes.h"
#include "toucan/ToucanUtils.h"
#include <boost/graph/topological_sort.hpp>
#include <boost/range/iterator_range.hpp>

#include <cassert>
#include <cstdint>
#include <fstream>
#include <memory>
#include <utility>

using namespace toucan;
using namespace mlir;


static bool are_ids_consecutive(const PartitioningGraph& g) {
  auto [begin, end] = vertices(g);
  for (size_t i = 0; begin != end; ++begin, ++i) {
    if (*begin != i) return false;
  }
  return true;
}

static bool isDirectoryExists(const std::string &dir) {
  return std::filesystem::exists(dir);
}

static bool createDirectoryIfNotExists(const std::string dir) {
  if (!isDirectoryExists(dir)) {
    bool created = std::filesystem::create_directory(dir);
    return created;
  }
  return true;
}

template <typename T>
static mlir::DenseMap<uint32_t, mlir::SmallVector<mlir::Value>> collectGraphVtxToInputVals(PartitioningGraph &g, const boost::iterator_range<T>& rng) {
  mlir::DenseMap<uint32_t, mlir::SmallVector<mlir::Value>> inputValMap;
  for (const auto &vtx: rng) {
    auto tOpName = g[vtx].toucanOpName;

    switch (tOpName) {
      case CGToucanOPName::MPart_Regular:
      case CGToucanOPName::MPart_Special: {
        const auto mPart = g[vtx].mp;
        assert(mPart != nullptr);

        // inputValMap[vtx] = mPart->inputValues;
        inputValMap[vtx] = {};
        inputValMap[vtx].append(mPart->inputValues.begin(), mPart->inputValues.end());
        // vtxOutputVals_r1.insert(mPart->outputValueSet.begin(), mPart->outputValueSet.end());

        break;
      }
      case CGToucanOPName::Print:
      case CGToucanOPName::Stop:
      case CGToucanOPName::RegWrite:{
        const auto op = g[vtx].op;
        assert(op != nullptr);
        inputValMap[vtx] = {};
        for (const auto &eachVal: op->getOperands()) {
          auto valDefiningOp = eachVal.getDefiningOp();
          if (!(isa<toucan::ConstantOp>(valDefiningOp) || isa<toucan::DefRegOp>(valDefiningOp) || isa<toucan::DefMemOp>(valDefiningOp))) {
            // vtxInputVals_r1.insert(eachVal);
            inputValMap[vtx].push_back(eachVal);
          }
        }

        break;
      }
      case CGToucanOPName::MemWrite: {
        const auto op = g[vtx].op;
        assert(op != nullptr);
        assert(isa<toucan::MemWriteOp>(op));
        auto memVal = cast<toucan::MemWriteOp>(op).getMem();
        inputValMap[vtx] = {};

        for (auto eachUserOp: memVal.getUsers()) {
          if (auto mwOp = dyn_cast<toucan::MemWriteOp>(eachUserOp)) {
            for (const auto &eachVal: mwOp->getOperands()) {
              auto valDefiningOp = eachVal.getDefiningOp();
              if (!(isa<toucan::ConstantOp>(valDefiningOp) || isa<toucan::DefRegOp>(valDefiningOp) || isa<toucan::DefMemOp>(valDefiningOp))) {
                // vtxInputVals_r1.insert(eachVal);
                inputValMap[vtx].push_back(eachVal);
              }
            }
          }
        }

        break;
      }

      case CGToucanOPName::RegRead:
      case CGToucanOPName::ExchangeRead: {
        break;
      }

      default: {
        llvm_unreachable("Op that should not appear here");
      }
    }
  }
  return inputValMap;
}


// Note: This function doesn't collect output value for exchange read
template <typename T>
static mlir::DenseMap<uint32_t, mlir::SmallVector<mlir::Value>> collectGraphVtxToOutputVals_NoExchangeRead(PartitioningGraph &g, const boost::iterator_range<T>& rng) {
  mlir::DenseMap<uint32_t, mlir::SmallVector<mlir::Value>> outputValMap;
  for (const auto &vtx: rng) {
    auto tOpName = g[vtx].toucanOpName;

    switch (tOpName) {
      case CGToucanOPName::MPart_Regular:
      case CGToucanOPName::MPart_Special: {
        const auto mPart = g[vtx].mp;
        assert(mPart != nullptr);

        outputValMap[vtx] = {};
        outputValMap[vtx].append(mPart->outputValueSet.begin(), mPart->outputValueSet.end());
        // outputValMap[vtx] = mPart->outputValueSet;

        break;
      }
      case toucan::CGToucanOPName::RegRead: {
        outputValMap[vtx] = {cast<toucan::RegReadOp>(g[vtx].op).getResult()};
        break;
      }
      case CGToucanOPName::Print:
      case CGToucanOPName::Stop:
      case CGToucanOPName::RegWrite:
      case CGToucanOPName::MemWrite: {
        break;
      }

      case CGToucanOPName::ExchangeRead: {
        break;
      }



      default: {
        llvm_unreachable("Op that should not appear here");
      }
    }
  }
  return outputValMap;
}


template <typename T>
static mlir::DenseSet<mlir::Operation*> collectGraphRawOps(PartitioningGraph &g, const boost::iterator_range<T>& rng) {
  mlir::DenseSet<mlir::Operation*> rawOps;

  for (auto vtx: rng) {

    auto tOpName = g[vtx].toucanOpName;

    switch (tOpName) {
      case CGToucanOPName::MPart_Regular:{
        const auto mPart = g[vtx].mp;
        assert(mPart != nullptr);

        for (const auto &[_, outputVal]: mPart->nodeToOutputVal) {
          // For now, each value can have only 1 writer
          rawOps.insert(outputVal.getDefiningOp());
        }
        assert(mPart->nops.empty());

        break;
      }
      case CGToucanOPName::MPart_Special: {
        const auto mPart = g[vtx].mp;
        assert(mPart != nullptr);

        for (const auto &op: mPart->specialOps) {
          rawOps.insert(op);
        }
        break;
      }

      case CGToucanOPName::RegRead:
      case CGToucanOPName::Print:
      case CGToucanOPName::Stop:
      case CGToucanOPName::RegWrite: {
        const auto op = g[vtx].op;
        assert(op != nullptr);

        rawOps.insert(op);
        break;
      }
      case CGToucanOPName::MemWrite: {
        const auto op = g[vtx].op;
        assert(op != nullptr);

        assert(isa<toucan::MemWriteOp>(op));
        auto memVal = cast<toucan::MemWriteOp>(op).getMem();

        for (auto eachUserOp: memVal.getUsers()) {
          if (auto mwOp = dyn_cast<toucan::MemWriteOp>(eachUserOp)) {
            rawOps.insert(mwOp);
          }
        }
        break;
      }

      case toucan::CGToucanOPName::ExchangeWrite:
      case CGToucanOPName::ExchangeRead: {
        // produces value, but already in valToProducingVtxes
        break;
      }

      default: {
        llvm_unreachable("Op that should not appear here");
      }
    }
  }

  return rawOps;
}

mlir::LogicalResult PartitioningManager::init(std::filesystem::path outputDirectory, int numRegions) {
  assert(numRegions == 2 && "Only support 2 regions for now!");

  this->outputDirectory = outputDirectory;

  regionWorkDirectory.clear();
  for (int rid = 0; rid < numRegions; rid++) {
    std::ostringstream oss;
    oss << "s2_r" << rid;
    std::string dirName = oss.str();
    auto workDir = outputDirectory / dirName;
    regionWorkDirectory.push_back(workDir);
    
    auto created = createDirectoryIfNotExists(workDir);
    if (!created) {
      llvm::errs() << "Fail to create directory: " << workDir << "\n";
      return failure();
    }
  }

  microPartitionerWorkDir = outputDirectory / "s1_mp";
  auto created = createDirectoryIfNotExists(microPartitionerWorkDir);
  if (!created) {
    llvm::errs() << "Fail to create directory: " << microPartitionerWorkDir << "\n";
    return failure();
  }

  rawGraphPath = microPartitionerWorkDir / graphFileName;
  vectorElementsMapFilePath = microPartitionerWorkDir / graphVectorDeclInfoFileName;

  return success();
}


mlir::LogicalResult PartitioningManager::runStage1MicroPartitioner(const PartitioningGraph &rawGraph) {
  llvm::outs() << "=============== Micro partitioning ===============\n";

  mlir::SmallVector<uint32_t> allNodesInEntireGraph;
  allNodesInEntireGraph.reserve(boost::num_vertices(rawGraph));
  for (auto vtx: boost::make_iterator_range((boost::vertices(rawGraph)))) {
    allNodesInEntireGraph.push_back(vtx);
  }

  // collect and dump vector map
  rawGraphVectorElementsMap = collectGraphVectorDeclInfoToFile(rawGraph);
  auto dumpVecSucc = dumpGraphVectorDeclInfoToFile(vectorElementsMapFilePath, rawGraphVectorElementsMap);
  if (failed(dumpVecSucc)) {
    return failure();
  }

  // dump entire graph for mp
  auto dumpGraphSucc = dumpGraphToFileForMicroPartitioner(rawGraph, allNodesInEntireGraph, rawGraphPath);
  if (failed(dumpGraphSucc)) {
    return failure();
  }


  mp = std::make_unique<MicroPartitioner>();

  mp->init(allNodesInEntireGraph, microPartitionerWorkDir, 0, rawGraphVectorElementsMap);
  // Note: force set input/output file path
  mp->graphVectorInfoFile = vectorElementsMapFilePath;
  mp->inputGraphFile = rawGraphPath;

  auto ret = mp->partition();

  if (failed(ret)) {
    llvm::errs() << "Fail to partition\n";
    return ret;
  }

  // Note: 1 special op per part
  ret = mp->arrangeSpecialOps(rawGraph, 1);
  if (failed(ret)) {
    llvm::errs() << "Fail to place special ops\n";
    return ret;
  }

  mp->collectPartIOValues(context, rawGraph);

  return success();
}

static void updateValProducingVtxForVecSegmentRead(mlir::DenseMap<mlir::Value, mlir::SmallVector<uint32_t>> &valToProducingVtxes) {
  mlir::SmallVector<mlir::TypedValue<toucan::VecType>> allVecVals;

  for (const auto &[val, _]: valToProducingVtxes) {
    if (auto vecVal = dyn_cast<mlir::TypedValue<toucan::VecType>>(val)) {
      allVecVals.push_back(vecVal);
    }
  }

  for (auto &vecVal: allVecVals) {
    for (auto userOp: vecVal.getUsers()) {
      if (auto segReadOp = dyn_cast<toucan::StaticVectorSegmentReadOp>(userOp)) {
        assert(valToProducingVtxes.contains(vecVal));
        auto segmentVal = segReadOp.getResult();
        if (valToProducingVtxes.contains(segmentVal)) {
          assert(valToProducingVtxes[segmentVal] == valToProducingVtxes[vecVal]);
        }
        valToProducingVtxes[segmentVal] = valToProducingVtxes[vecVal];
      }
    }
  }
}

mlir::LogicalResult PartitioningManager::buildMicroPartGraph(const PartitioningGraph &rawGraph) {
  assert(!mp->partLevels.empty());
  microPartGraph = std::make_unique<PartitioningGraph>();
  mlir::DenseMap<mlir::Value, mlir::SmallVector<uint32_t>> mpGraphValueToProducingVtx;

  // Add all micro parts
  for (const auto &eachLevel: mp->partLevels) {
    for (auto &mPart: eachLevel) {
      PartitioningGraphNodeProperty vp;
      vp.op = nullptr;
      // Don't assign weight for now. Region 0 and region 1 has different weight
      vp.weight = 0;
      vp.toucanOpName = (mPart->isRegularPart()) ? CGToucanOPName::MPart_Regular : CGToucanOPName::MPart_Special;
      vp.opCount = 0;
      vp.exchangeValId = 0;
      vp.mp = mPart;
      auto newVertex = boost::add_vertex(vp, *microPartGraph);

      // track producing values
      for (const auto &eachVal: mPart->outputValueSet) {
        if (mpGraphValueToProducingVtx.contains(eachVal)) {
          // could be vector
          mpGraphValueToProducingVtx[eachVal].push_back(newVertex);
        } else {
          mpGraphValueToProducingVtx[eachVal] = {static_cast<uint32_t>(newVertex)};
        }
      }
    }


  }

  // Add all reg reads
  for (auto oldVtxId: mp->allRegReads) {
    auto newVertex = boost::add_vertex(rawGraph[oldVtxId], *microPartGraph);
    assert(!mpGraph_idToRawGraphId.contains(newVertex));
    mpGraph_idToRawGraphId[newVertex] = oldVtxId;

    auto vp = rawGraph[oldVtxId];
    assert(vp.toucanOpName == CGToucanOPName::RegRead);
    assert(vp.op != nullptr);

    // track producing values
    auto resultVal = cast<toucan::RegReadOp>(vp.op).getResult();
    assert(!mpGraphValueToProducingVtx.contains(resultVal));
    mpGraphValueToProducingVtx[resultVal] = {static_cast<uint32_t>(newVertex)};
  }


  updateValProducingVtxForVecSegmentRead(mpGraphValueToProducingVtx);
  // At this point we have all producer visited. Create edges for all MParts
  for (auto vtx: boost::make_iterator_range((boost::vertices(*microPartGraph)))) {
    auto tOpName = (*microPartGraph)[vtx].toucanOpName;
    if (!(tOpName == CGToucanOPName::MPart_Regular || tOpName == CGToucanOPName::MPart_Special)) {
      // Not a MPart
      continue;
    }

    auto mPart = (*microPartGraph)[vtx].mp;
    assert(mPart != nullptr);

    for (const auto &eachVal: mPart->inputValues) {
      assert(!isa<toucan::ConstantOp>(eachVal.getDefiningOp()));
      assert(mpGraphValueToProducingVtx.contains(eachVal));

      for (const auto &eachSrcVtx: mpGraphValueToProducingVtx[eachVal]) {
        auto dstVtx = vtx;
        boost::add_edge(eachSrcVtx, dstVtx, *microPartGraph);
      }
    }
  }

  // visit RW, MW, Stop, Print, add missing vtx and edges
  auto createVtxAndConnectEdge = [&](uint32_t oldVtxId) {
    auto vp = rawGraph[oldVtxId];
    auto newVertex = boost::add_vertex(vp, *microPartGraph);
    assert(!mpGraph_idToRawGraphId.contains(newVertex));
    mpGraph_idToRawGraphId[newVertex] = oldVtxId;

    assert(vp.op != nullptr);

    for (const auto &eachVal: vp.op->getOperands()) {
      auto valDefiningOp = eachVal.getDefiningOp();
      if (!(isa<toucan::ConstantOp>(valDefiningOp) || isa<toucan::DefRegOp>(valDefiningOp) || isa<toucan::DefMemOp>(valDefiningOp))) {
        assert(mpGraphValueToProducingVtx.contains(eachVal));
        for (const auto &eachSrcVtx: mpGraphValueToProducingVtx[eachVal]) {
          auto dstVtx = newVertex;
          boost::add_edge(eachSrcVtx, dstVtx, *microPartGraph);
        }
      }
    }
  };
  
  for (auto oldVtxId: mp->allRegWrites) {
    createVtxAndConnectEdge(oldVtxId);
  }
  for (auto oldVtxId: mp->allMemWrites) {
    createVtxAndConnectEdge(oldVtxId);
  }
  for (auto oldVtxId: mp->allPrints) {
    createVtxAndConnectEdge(oldVtxId);
  }
  for (auto oldVtxId: mp->allStops) {
    createVtxAndConnectEdge(oldVtxId);
  }

  // levelize
  levelizeGraph(*microPartGraph, microPartGraph_levels);

  return success();
}


int PartitioningManager::findCutPoint() {

  // count expected parts
  std::vector<uint32_t> estimateMPartsPerLevel;
  for (const auto &eachLevel: microPartGraph_levels) {
    uint32_t thisLevelMPartCntEst = 0;

    for (const auto &vtx: eachLevel) {
      auto tOpName = (*microPartGraph)[vtx].toucanOpName;
      auto mPart = (*microPartGraph)[vtx].mp;

      uint32_t numOpWidth = 0;

      switch (tOpName) {
        case CGToucanOPName::RegRead: {
          numOpWidth += 1;
          break;
        }
        case CGToucanOPName::MPart_Regular: {
          uint32_t maxParallelOps = 0;
          for (const auto &el: mPart->levels) {
            maxParallelOps = std::max(maxParallelOps, static_cast<uint32_t>(el.size()));
          }

          numOpWidth += maxParallelOps;
          break;
        }
        case CGToucanOPName::MPart_Special: {
          numOpWidth += mPart->specialOps.size();
          break;
        }
        case CGToucanOPName::LUT: {
          numOpWidth += 1;
          break;
        }

        case CGToucanOPName::Print:
        case CGToucanOPName::Stop:
        case CGToucanOPName::RegWrite:
        case CGToucanOPName::MemWrite: {
          numOpWidth += 1;
          break;
        }
        default: {
          llvm_unreachable("Unexpected node type");
        }
      }

      thisLevelMPartCntEst += (numOpWidth + 31) / 32;
    }
    estimateMPartsPerLevel.push_back(thisLevelMPartCntEst);
  }

  float region0VtxTarget = 0.2f;


  int totalNumVtxes = 0;
  int maxLevels = estimateMPartsPerLevel.size();

  for (const auto &eachLevelEst: estimateMPartsPerLevel) {
    totalNumVtxes += eachLevelEst;
  }

  int region0NumVtxes = 0;
  int cutPoint = 0;

  size_t level_id = 0;
  for (const auto &eachLevel: estimateMPartsPerLevel) {
    llvm::outs() << "Graph level " << level_id << " has " << eachLevel << " estimated MPart nodes\n";
    level_id++;
  }
  for (const auto &eachLevelEstMP: estimateMPartsPerLevel) {
    region0NumVtxes += eachLevelEstMP;
    cutPoint ++;
    if (region0NumVtxes >= (totalNumVtxes * region0VtxTarget)) {
      break;
    }
  }

  assert(cutPoint < maxLevels);

  return cutPoint;
}


void PartitioningManager::cutGraph(int cutPoint) {

  microPartGraph_r0 = std::make_shared<PartitioningGraph>();
  microPartGraph_r1 = std::make_shared<PartitioningGraph>();

  mlir::SmallVector<uint32_t> region0Vtxes, region1Vtxes;

  // Note: Move all terminus vtx to region 2
  auto isTerminusVtx = [](CGToucanOPName tOpName) {
    return (
      tOpName == CGToucanOPName::RegWrite
       || tOpName == CGToucanOPName::MemWrite
       || tOpName == CGToucanOPName::Stop
       || tOpName == CGToucanOPName::Print
    );
  };
  mlir::SmallVector<uint32_t> terminusVtxesBeforeCutPoint;

  for (int level_id = 0; level_id < static_cast<int>(microPartGraph_levels.size()); level_id++) {
    const auto &eachLevel = microPartGraph_levels[level_id];

    if (level_id <= cutPoint) {
      for (auto &eachVtx: eachLevel) {
        auto tOpName = (*microPartGraph)[eachVtx].toucanOpName;
        if (isTerminusVtx(tOpName)) {
          terminusVtxesBeforeCutPoint.push_back(eachVtx);
        } else {
          region0Vtxes.push_back(eachVtx);
        }
      }
    } else {
      region1Vtxes.append(eachLevel.begin(), eachLevel.end());
    }
  }

  llvm::outs() << terminusVtxesBeforeCutPoint.size() << " terminus vtxes moved from region 0 to region 1\n";
  region1Vtxes.append(terminusVtxesBeforeCutPoint.begin(), terminusVtxesBeforeCutPoint.end());


  
  llvm::outs() << "Cut micropart graph at level " << cutPoint << ", region 0 has " << region0Vtxes.size() << " vertices, region 1 has " << region1Vtxes.size() << "\n";


  mlir::DenseSet<mlir::Value> vtxInputVals_r0, vtxOutputVals_r0;
  mlir::DenseSet<mlir::Value> vtxInputVals_r1, vtxOutputVals_r1;
  {
    // Collect values that need to be passed from r0 to r1

    // 1. collect r0 io values
    auto vtxToInputVals_r0 = collectGraphVtxToInputVals(*microPartGraph, boost::make_iterator_range(region0Vtxes));
    auto vtxToOutputVals_r0 = collectGraphVtxToOutputVals_NoExchangeRead(*microPartGraph, boost::make_iterator_range(region0Vtxes));
    for (const auto &kv: vtxToInputVals_r0) {
      for (const auto &v: kv.second) {
        vtxInputVals_r0.insert(v);
      }
    }
    for (const auto &kv: vtxToOutputVals_r0) {
      for (const auto &v: kv.second) {
        vtxOutputVals_r0.insert(v);
      }
    }

    // 2. collect r1 io values
    auto vtxToInputVals_r1 = collectGraphVtxToInputVals(*microPartGraph, boost::make_iterator_range(region1Vtxes));
    auto vtxToOutputVals_r1 = collectGraphVtxToOutputVals_NoExchangeRead(*microPartGraph, boost::make_iterator_range(region1Vtxes));
    for (const auto &kv: vtxToInputVals_r1) {
      for (const auto &v: kv.second) {
        vtxInputVals_r1.insert(v);
      }
    }
    for (const auto &kv: vtxToOutputVals_r1) {
      for (const auto &v: kv.second) {
        vtxOutputVals_r1.insert(v);
      }
    }


    // 3. collect exchange values
    for (const auto &eachVal: vtxInputVals_r1) {

      bool isExchangeVal = false;
      mlir::Value exgVal = eachVal;

      if (auto segReadOp = dyn_cast<toucan::StaticVectorSegmentReadOp>(eachVal.getDefiningOp())) {
        // special handling for segment vals.
        auto vecVal = segReadOp.getHandle();
        assert(!exchangeValues.contains(eachVal));

        bool vecValInRegion0 = vtxOutputVals_r0.contains(vecVal);
        // bool vecValInRegion1 = vtxOutputVals_r1.contains(vecVal);
        bool vecValAlreadyInExchange = exchangeValues.contains(vecVal);
        if (vecValInRegion0 && (!vecValAlreadyInExchange)) {
          isExchangeVal = true;
          exgVal = vecVal;
        }
      } else {
        if (vtxOutputVals_r0.contains(eachVal)) isExchangeVal = true;
      }

      if (isExchangeVal) {
        // randomly assign an ID for now. Will sort later
        auto exgId = exchangeValPool.size();
        assert(!exchangeValues.contains(exgVal));
        exchangeValues[exgVal] = exgId;
        exchangeValPool.push_back(exgVal);
      }
    }

  }


  // build graph
  mlir::SmallVector<uint32_t> r0VtxToOldId, r1VtxToOldId;

  auto createNewSubGraph = [&](PartitioningGraph &newGraph, const PartitioningGraph &oldGraph, mlir::SmallVector<uint32_t> &newVtxToOldVtx, const mlir::SmallVector<uint32_t> oldVtxesInNewGraph, const mlir::DenseMap<mlir::Value, uint32_t> exgReadVals, const mlir::DenseMap<mlir::Value, uint32_t> exgWriteVals) {
    newVtxToOldVtx.clear();
    newVtxToOldVtx.reserve(oldVtxesInNewGraph.size() * 2);
    assert(boost::num_vertices(newGraph) == 0);

    // Add all normal nodes
    for (const auto eachOldVtx: oldVtxesInNewGraph) {
      auto newVtx = boost::add_vertex(oldGraph[eachOldVtx], newGraph);
      assert(newVtx == newVtxToOldVtx.size());
      newVtxToOldVtx.push_back(eachOldVtx);
    }


    mlir::DenseMap<mlir::Value, mlir::SmallVector<uint32_t>> valToProducingVtxes;

    // Add exchange read nodes, collect output values
    for (const auto &kv: exgReadVals) {
      const auto &eachVal = kv.getFirst();
      const auto &exgValId = kv.getSecond();

      // randomly assign an id
      PartitioningGraphNodeProperty vp;
      vp.op = nullptr;
      // Don't assign weight for now. Region 0 and region 1 has different weight
      vp.weight = 0;
      vp.toucanOpName = CGToucanOPName::ExchangeRead;
      vp.opCount = 1;
      vp.exchangeValId = exgValId;
      vp.mp = nullptr;
      auto newVertex = boost::add_vertex(vp, newGraph);

      assert(!valToProducingVtxes.contains(eachVal));
      valToProducingVtxes[eachVal] = {static_cast<uint32_t>(newVertex)};
    }

    // collect all IO values for all vtxes
    mlir::DenseMap<uint32_t, mlir::SmallVector<mlir::Value>> vtxToOutputVals, vtxToInputVals;
    {
      vtxToInputVals = collectGraphVtxToInputVals(newGraph, boost::make_iterator_range(boost::vertices(newGraph)));
      vtxToOutputVals = collectGraphVtxToOutputVals_NoExchangeRead(newGraph, boost::make_iterator_range(boost::vertices(newGraph)));
    }


    // update val producing map
    for (auto &[vtx, outputVals]: vtxToOutputVals) {
      for (auto &val: outputVals) {
        if (!valToProducingVtxes.contains(val)) {
          valToProducingVtxes[val] = {};
        }
        valToProducingVtxes[val].push_back(vtx);
      }
    }
    updateValProducingVtxForVecSegmentRead(valToProducingVtxes);


    for (auto vtx: boost::make_iterator_range((boost::vertices(newGraph)))) {
      auto &vtxInputVals = vtxToInputVals[vtx];

      for (const auto &eachVal: vtxInputVals) {
        assert(valToProducingVtxes.contains(eachVal) && "Input value not found. Is it missing in exchange?");
        for (const auto valProducingVtx: valToProducingVtxes[eachVal]) {
          boost::add_edge(valProducingVtx, vtx, newGraph);
        }
      }
    }

    // Add exchange write and connect edges
    for (const auto &kv: exgWriteVals) {
      const auto &eachVal = kv.getFirst();
      const auto &exgValId = kv.getSecond();

      // assign an id
      PartitioningGraphNodeProperty vp;
      vp.op = nullptr;
      // Don't assign weight for now. Region 0 and region 1 has different weight
      vp.weight = 0;
      vp.toucanOpName = CGToucanOPName::ExchangeWrite;
      vp.opCount = 1;
      vp.exchangeValId = exgValId;
      vp.mp = nullptr;
      auto newVertex = boost::add_vertex(vp, newGraph);

      assert(valToProducingVtxes.contains(eachVal));
      for (const auto valProducingVtx: valToProducingVtxes[eachVal]) {
        boost::add_edge(valProducingVtx, newVertex, newGraph);
      }
    }
  };


  // build r0
  llvm::outs() << "Build micro part graph for region 0\n";

  mlir::DenseMap<mlir::Value, uint32_t> emptyMap;

  mlir::SmallVector<uint32_t> newVtxToOldVtx_r0;

  // r0: has only exchange write but no read
  createNewSubGraph(
    *microPartGraph_r0, 
    *microPartGraph, 
    newVtxToOldVtx_r0, 
    region0Vtxes, 
    emptyMap, 
    exchangeValues);

  // build r1

  llvm::outs() << "Build micro part graph for region 1\n";
  mlir::SmallVector<uint32_t> newVtxToOldVtx_r1;

  // r1: has only exchange read but no write
  createNewSubGraph(
    *microPartGraph_r1, 
    *microPartGraph, 
    newVtxToOldVtx_r1, 
    region1Vtxes, 
    exchangeValues, 
    emptyMap);

}


void PartitioningManager::breakDirectIOConnection(const PartitioningGraph &rawGraph) {
  valToUserOpsDoNotReplace.clear();

  // collect raw ops in region 1
  auto rawOpsInRegion1 = collectGraphRawOps(*microPartGraph_r1, boost::make_iterator_range((boost::vertices(*microPartGraph_r1))));


  newNOPVtxes_r0 = breakDirectIOConnectionWorker(*microPartGraph_r0, rawOpsInRegion1);
  newNOPVtxes_r1 = breakDirectIOConnectionWorker(*microPartGraph_r1, {});

  levelizeGraph(*microPartGraph_r0, microPartGraph_r0_levels);
  levelizeGraph(*microPartGraph_r1, microPartGraph_r1_levels);

  llvm::outs() << "Update all mPart IO values\n";
  mp->collectPartIOValues(context, rawGraph);

  valToUserOpsDoNotReplace.shrink_and_clear();
}

mlir::SmallVector<uint32_t> PartitioningManager::breakDirectIOConnectionWorker(PartitioningGraph &g, const mlir::DenseSet<mlir::Operation*> &rawOpsInFollowingRegions) {
  mlir::SmallVector<uint32_t> newNOPVtxes;

  auto tOpIsInput = [](CGToucanOPName tOpName) {
    return (
      tOpName == CGToucanOPName::ExchangeRead 
      || tOpName == CGToucanOPName::RegRead
    );
  };

  auto tOpIsOutput = [](CGToucanOPName tOpName) {
    return (
      tOpName == CGToucanOPName::ExchangeWrite
      || tOpName == CGToucanOPName::RegWrite
    );
  };

  auto opIsOutput = [](mlir::Operation *op) {
    return (
      isa<toucan::RegWriteOp>(op)
    );
  };

  mlir::SmallVector<std::pair<uint32_t, uint32_t>> edgesToBreak;

  for (auto srcVtx : boost::make_iterator_range(boost::vertices(g))) {
    auto srcTOpName = g[srcVtx].toucanOpName;

    if (tOpIsInput(srcTOpName)) {
      // an input node
      assert(boost::in_degree(srcVtx, g) == 0);

      for (auto ei = boost::out_edges(srcVtx, g); ei.first != ei.second; ei.first++) {
        auto dstVtx = boost::target(*ei.first, g);
        auto dstTOpName = g[dstVtx].toucanOpName;

        if (tOpIsOutput(dstTOpName)) {
          assert(boost::out_degree(dstVtx, g) == 0);
          // this edge need break
          edgesToBreak.push_back({srcVtx, dstVtx});

          {
            bool srcVtxIsExgRead = (srcTOpName == CGToucanOPName::ExchangeRead);
            bool dstVtxIsExgWrite = (dstTOpName == CGToucanOPName::ExchangeWrite);
            // Not for correctness, but direct connection from exgread to exgWrite is unnecessary
            assert(!(srcVtxIsExgRead && dstVtxIsExgWrite) && "Remove this assertion should still work. However a direct edge from ExgRead to ExgWrite is unnecessary.");
          }
        }
      }
    }
  }

  // pick any operation to create IRRewriter
  mlir::Operation *anyOp = nullptr;
  for (auto srcVtx : boost::make_iterator_range(vertices(g))) {
    auto op = g[srcVtx].op;
    if (op != nullptr) {
      anyOp = op;
      break;
    }
  }

  auto loc = anyOp->getLoc();
  OpBuilder builder(anyOp);
  mlir::IRRewriter rewriter(builder);

  

  {
    // find user op that inside this graph/region. Do not replace them in later stage
    mlir::DenseMap<uint32_t, mlir::Value> allSrcVtxInEdgesToBreak;
    for (auto [srcVtx, dstVtx]: edgesToBreak) {
      auto dstTOpName = g[dstVtx].toucanOpName;
      bool dstVtxIsExgWrite = (dstTOpName == CGToucanOPName::ExchangeWrite);
      bool dstVtxIsRegWrite = (dstTOpName == CGToucanOPName::RegWrite);
      // bool dstVtxIsStop = (dstTOpName == CGToucanOPName::Stop);
      // bool dstVtxIsPrint = (dstTOpName == CGToucanOPName::Print);

      mlir::Value srcVal;
      if (dstVtxIsRegWrite) {
        auto dstOp = g[dstVtx].op;
        assert(isa<toucan::RegWriteOp>(dstOp));
        srcVal = cast<toucan::RegWriteOp>(dstOp).getData();
      } else if (dstVtxIsExgWrite) {
        auto exchangeValId = g[dstVtx].exchangeValId;
        assert(exchangeValPool.size() > exchangeValId);
        srcVal = exchangeValPool[exchangeValId];
      } else {
        llvm::errs() << stringifyCGToucanOPName(dstTOpName) << "\n";
        llvm_unreachable("Unexpected break edge end point");
      }

      if (!allSrcVtxInEdgesToBreak.contains(srcVtx)) {
        allSrcVtxInEdgesToBreak[srcVtx] = srcVal;
      } else {
        assert(allSrcVtxInEdgesToBreak[srcVtx] == srcVal);
      }

    }
    for (auto [srcVtx, srcVal]: allSrcVtxInEdgesToBreak) {
      if (!valToUserOpsDoNotReplace.contains(srcVal)) {
        valToUserOpsDoNotReplace[srcVal] = {};
      }
      for (auto ei = boost::out_edges(srcVtx, g); ei.first != ei.second; ++ei.first) {
        auto dstVtx = boost::target(*ei.first, g);

        auto dstVtxOpName = g[dstVtx].toucanOpName;
        auto dstOp = g[dstVtx].op;
        auto mp = g[dstVtx].mp;

        switch (dstVtxOpName) {
          case CGToucanOPName::MPart_Regular: {
            assert(mp != nullptr);
            assert(mp->inputValues.contains(srcVal));

            for (auto &[_, v]: mp->nodeToOutputVal) {
              auto op = v.getDefiningOp();
              // valToUserOpsDoNotReplace[srcVal].insert(op);
              for (const auto &operand: op->getOperands()) {
                if (operand == srcVal) {
                  valToUserOpsDoNotReplace[srcVal].insert(op);
                  break;
                }
              }
            }

            assert(mp->nops.empty());
            assert(valToUserOpsDoNotReplace[srcVal].size() != 0);

            break;
          }
          case CGToucanOPName::MPart_Special: {
            assert(mp != nullptr);
            assert(mp->inputValues.contains(srcVal));

            for (const auto &op: mp->specialOps) {
              // valToUserOpsDoNotReplace[srcVal].insert(op);
              for (const auto &operand: op->getOperands()) {
                if (operand == srcVal) {
                  valToUserOpsDoNotReplace[srcVal].insert(op);
                  break;
                }
              }
            }
            assert(valToUserOpsDoNotReplace[srcVal].size() != 0);

            break;
          }
          case CGToucanOPName::Print:
          case CGToucanOPName::Stop:
          case CGToucanOPName::RegWrite:
          case CGToucanOPName::MemWrite: {
            assert(dstOp != nullptr);
            valToUserOpsDoNotReplace[srcVal].insert(dstOp);

            break;
          }

          case CGToucanOPName::ExchangeWrite:
          case CGToucanOPName::ExchangeRead: {
            break;
          }

          default: {
            llvm::dbgs() << stringifyCGToucanOPName(dstVtxOpName) << "\n";
            llvm_unreachable("Op that should not appear here");
          }
        }
      }
    }
  }


  // mlir::DenseMap<uint32_t, uint32_t> regToNewReader;
  mlir::DenseMap<mlir::Value, uint32_t> valToNewReader;
  for (auto [srcVtx, dstVtx]: edgesToBreak) {
    auto dstTOpName = g[dstVtx].toucanOpName;
    bool dstVtxIsExgWrite = (dstTOpName == CGToucanOPName::ExchangeWrite);
    bool dstVtxIsRegWrite = (dstTOpName == CGToucanOPName::RegWrite);
    // bool dstVtxIsStop = (dstTOpName == CGToucanOPName::Stop);
    // bool dstVtxIsPrint = (dstTOpName == CGToucanOPName::Print);

    boost::remove_edge(srcVtx, dstVtx, g);

    mlir::Value srcVal;
    if (dstVtxIsRegWrite) {
      auto dstOp = g[dstVtx].op;
      assert(isa<toucan::RegWriteOp>(dstOp));
      srcVal = cast<toucan::RegWriteOp>(dstOp).getData();
    } else if (dstVtxIsExgWrite) {
      auto exchangeValId = g[dstVtx].exchangeValId;
      assert(exchangeValPool.size() > exchangeValId);
      srcVal = exchangeValPool[exchangeValId];
    } else {
      llvm::errs() << stringifyCGToucanOPName(dstTOpName) << "\n";
      llvm_unreachable("Unexpected break edge end point");
    }


    if (!valToNewReader.contains(srcVal)) {
      // create a nop
      auto newNop = rewriter.create<toucan::LUTOp>(loc, toucan::LUTOpName::LUT_Nop, srcVal);

      // Insert nop
      PartitioningGraphNodeProperty vp;
      vp.op = newNop;
      vp.weight = DESIGNGRAPH_BREAK_IO_NOP_WEIGHT;
      vp.opCount = 1;
      vp.exchangeValId = UINT32_MAX;
      vp.mp = nullptr;
      vp.toucanOpName = CGToucanOPName::LUT;

      auto nopVtxId = boost::add_vertex(vp, g);
      newNOPVtxes.push_back(nopVtxId);


      srcVal.replaceUsesWithIf(newNop.getResult(), [&](const OpOperand &operand) {
        auto userOp = operand.getOwner();
        auto operandVal = operand.get();

        assert(valToUserOpsDoNotReplace.contains(operandVal));
        auto thisUseIsInCurrentRegion = valToUserOpsDoNotReplace[operandVal].contains(userOp);

        bool usedInFollowingRegion = rawOpsInFollowingRegions.contains(userOp);
        bool isOutput = opIsOutput(userOp);

        return (usedInFollowingRegion && (!thisUseIsInCurrentRegion)) || isOutput;
      });

      // use this nop vtx to avoid direct contact
      boost::add_edge(srcVtx, nopVtxId, g);
      assert(!valToNewReader.contains(srcVal));
      valToNewReader[srcVal] = nopVtxId;

    }

    // add edge that use the nop
    auto nopVtxId = valToNewReader[srcVal];
    boost::add_edge(nopVtxId, dstVtx, g);

    // update use
    auto nopVal = cast<toucan::LUTOp>(g[nopVtxId].op).getResult();
    auto dstVtxOpName = g[dstVtx].toucanOpName;
    auto dstOp = g[dstVtx].op;

    if (dstVtxOpName == CGToucanOPName::ExchangeWrite) {
      // update exchangeVal
      assert(dstOp == nullptr);
      auto exchangeValId = g[dstVtx].exchangeValId;
      assert(exchangeValPool.size() > exchangeValId);
      auto oldExchangeVal = exchangeValPool[exchangeValId];
      assert(exchangeValues.at(oldExchangeVal) == exchangeValId);

      exchangeValPool[exchangeValId] = nopVal;
      exchangeValues.erase(oldExchangeVal);
      exchangeValues[nopVal] = exchangeValId;
    }
  }

  llvm::outs() << "Insert " << valToNewReader.size() << " extra NOP verticies to break " << edgesToBreak.size() << " direct IO edges\n";

  return newNOPVtxes;
}


void PartitioningManager::updateGraphWeight_r0() {
  // region 0: optimize for throughput
  auto r0Graph = *microPartGraph_r0;
  assert(!microPartGraph_r0_levels.empty());

  for (auto vtx: boost::make_iterator_range((boost::vertices(r0Graph)))) {
    auto tOpName = r0Graph[vtx].toucanOpName;
    auto mPart = r0Graph[vtx].mp;
    auto exchangeValId = r0Graph[vtx].exchangeValId;
    auto rawOp = r0Graph[vtx].op;

    uint32_t weight = 0;

    switch (tOpName) {
      case CGToucanOPName::RegRead: {
        weight = 2;
        break;
      }
      case CGToucanOPName::MPart_Regular: {
        auto mpart_levels = mPart->levels.size();

        size_t max_ops = 0;
        for (const auto &eachLevel: mPart->levels) {
          max_ops = std::max(max_ops, eachLevel.size());
        }
        assert(mpart_levels > 0);

        weight = max_ops * 10;

        break;
      }
      case CGToucanOPName::MPart_Special: {
        auto mpart_numOps = mPart->specialOps.size();
        // For now should just be 1
        assert(mpart_numOps == 1);

        weight = mpart_numOps * 10;
        break;
      }
      case CGToucanOPName::LUT: {
        assert(rawOp != nullptr);
        assert(isa<toucan::LUTOp>(rawOp));
        assert(cast<toucan::LUTOp>(rawOp).getOpName() == LUTOpName::LUT_Nop);

        weight = 10;
        break;
      }
      case CGToucanOPName::ExchangeWrite: {
        auto exgVal = exchangeValPool[exchangeValId];
        if (auto vecVal = dyn_cast<mlir::TypedValue<toucan::VecType>>(exgVal)) {
          // a vector
          auto vecLength = vecVal.getType().getLength();
          weight = vecLength;
        } else {
          weight = 1;
        }
        break;
      }

      case CGToucanOPName::Print:
      case CGToucanOPName::Stop:
      case CGToucanOPName::RegWrite:
      case CGToucanOPName::MemWrite:
      default: {
        llvm_unreachable("Unexpected node type");
      }
    }

    r0Graph[vtx].weight = weight;
  }
}
void PartitioningManager::updateGraphWeight_r1() {
  // region 1: optimize for latency
  auto r1Graph = *microPartGraph_r1;
  assert(!microPartGraph_r1_levels.empty());

  for (uint32_t level_id = 0; level_id < microPartGraph_r1_levels.size(); level_id++) {
    for (const auto eachVtx: microPartGraph_r1_levels[level_id]) {
      uint32_t weight = 0;
      auto tOpName = r1Graph[eachVtx].toucanOpName;
      auto exgValId = r1Graph[eachVtx].exchangeValId;

      weight = level_id * 10 + 1;

      if (tOpName == CGToucanOPName::ExchangeRead) {
        auto val = exchangeValPool[exgValId];
        uint32_t readBytes = 1;
        if (auto vecVal = llvm::dyn_cast<mlir::TypedValue<toucan::VecType>>(val)) {
          readBytes = vecVal.getType().getLength();
        }

        weight = readBytes * 2;
      }

      r1Graph[eachVtx].weight = weight;
    }
  }
}


mlir::LogicalResult PartitioningManager::runStage2RepCutPartitioner(float partSizeRatio, float ibFactor) {
  assert(regionWorkDirectory.size() == 2);
  // r0: optimize for throughput


  llvm::outs() << "=============== RepCut partitioning for region 0 ===============\n";

  auto p_r0 = RepCutPartitioner(regionWorkDirectory[0], microPartGraph_r0);
  p_r0.PARTITION_MAX_WEIGHT = 60000;
  p_r0.PARTITION_PREFERRED_WEIGHT = 40000;
  p_r0.targetIb = ibFactor;

  p_r0.setPartitionTarget(partSizeRatio);
  auto ret_r0 = p_r0._partition(context);
  if (failed(ret_r0)) return ret_r0;

  // r1 optimize for latency

  llvm::outs() << "=============== RepCut partitioning for region 1 ===============\n";
  auto p_r1 = RepCutPartitioner(regionWorkDirectory[1], microPartGraph_r1);

  p_r1.PARTITION_MAX_WEIGHT = 50000;
  p_r1.PARTITION_PREFERRED_WEIGHT = 30000;
  p_r1.targetIb = ibFactor;

  p_r1.setPartitionTarget(partSizeRatio);

  auto ret_r1 = p_r1._partition(context);
  if (failed(ret_r1)) return ret_r1;

  assert(p_r0.repcutPartitions.size() != 0);
  assert(p_r1.repcutPartitions.size() != 0);
  std::swap(p_r0.repcutPartitions, repcutPartitions_r0);
  std::swap(p_r1.repcutPartitions, repcutPartitions_r1);

  return success();
}


void PartitioningManager::collectRepCutPartitionCodeGenData() {


  auto getCodeGenData = [&](
    RepCutPartitionCodeGenData &info, 
    PartitioningGraph &partGraph, 
    const mlir::SmallVector<uint32_t> &repcutNodes, 
    const mlir::SmallVector<mlir::SmallVector<uint32_t>>& graphLevels, 
    const mlir::SmallVector<mlir::Value> &exchangeValPool
  ) {
    // copy existing MP
    // merge special MP
    // build NOP MP
    mlir::DenseSet<uint32_t> repcutNodeSet;
    repcutNodeSet.insert(repcutNodes.begin(), repcutNodes.end());

    assert(info.mpartLevels.empty());
    mlir::DenseSet<mlir::Value> activeVals;
    auto assertValExist = [&activeVals](const mlir::Value &val) {
      auto v = val;
      if (auto segReadOp = dyn_cast<toucan::StaticVectorSegmentReadOp>(val.getDefiningOp())) {
        auto vecVal = segReadOp.getHandle();
        v = vecVal;
      }

      if (!isa<toucan::ConstantOp>(v.getDefiningOp())) {
        if (!activeVals.contains(v)) {
          v.print(llvm::dbgs());
          llvm::dbgs() << "\n";
        }
        assert(activeVals.contains(v));
      }
    };
    auto insertActiveVal = [&activeVals](const mlir::Value &val) {
      activeVals.insert(val);
    };

    mlir::SmallVector<mlir::SmallVector<uint32_t>> graphLevelOfThisPart;
    for (uint32_t level_id = 0; level_id < graphLevels.size(); level_id++) {
      graphLevelOfThisPart.emplace_back();

      for (const auto vtx: graphLevels[level_id]) {
        if (repcutNodeSet.contains(vtx)) {
          graphLevelOfThisPart.back().push_back(vtx);
        };
      }
    }


    mlir::SmallVector<std::shared_ptr<MicroPart>> specialMPartsInThisLevel;
    mlir::SmallVector<toucan::LUTOp> nopsInThisLevel;


    info.mpartLevels.reserve(graphLevelOfThisPart.size());
    for (uint32_t level_id = 0; level_id < graphLevelOfThisPart.size(); level_id++) {
      if (graphLevelOfThisPart[level_id].empty()) continue;

      // llvm::dbgs() << "Scanning at level " << level_id << "\n";

      mlir::SmallVector<std::shared_ptr<MicroPart>> mpsThisLevel;
      mpsThisLevel.reserve(graphLevelOfThisPart[level_id].size());

      specialMPartsInThisLevel.clear();
      nopsInThisLevel.clear();
      for (const auto vtx: graphLevelOfThisPart[level_id]) {
        assert(repcutNodeSet.contains(vtx));

        auto tOpName = partGraph[vtx].toucanOpName;
        auto op = partGraph[vtx].op;
        auto &mPart = partGraph[vtx].mp;
        auto opCount = partGraph[vtx].opCount;

        switch (tOpName) {
          case CGToucanOPName::MPart_Regular: {
            // regular mPart. Just keep it
            assert(mPart != nullptr);
            assert(mPart->partIsValid);
            mpsThisLevel.push_back(mPart);

            for (const auto &eachVal: mPart->inputValues) {
              // assertValExist(eachVal);
              auto v = eachVal;
              if (auto segReadOp = dyn_cast<toucan::StaticVectorSegmentReadOp>(eachVal.getDefiningOp())) {
                auto vecVal = segReadOp.getHandle();
                v = vecVal;
              }
              assert(activeVals.contains(v));
            }
            for (const auto &eachVal: mPart->outputValueSet) {
              insertActiveVal(eachVal);
            }
            break;
          }
          case CGToucanOPName::MPart_Special: {
            assert(mPart != nullptr);
            assert(mPart->partIsValid);
            assert(!mPart->isRegularPart());
            assert(mPart->specialOps.size() == 1);
            specialMPartsInThisLevel.push_back(mPart);

            for (const auto &eachVal: mPart->inputValues) {
              assertValExist(eachVal);
            }
            for (const auto &eachVal: mPart->outputValueSet) {
              insertActiveVal(eachVal);
            }
            break;
          }
          case CGToucanOPName::LUT: {
            // should be a nop
            assert(op != nullptr);
            assert(mPart == nullptr);
            assert(opCount == 1);
            assert(cast<toucan::LUTOp>(op).getOpName() == LUTOpName::LUT_Nop);
            nopsInThisLevel.push_back(cast<toucan::LUTOp>(op));

            auto lutOp = cast<toucan::LUTOp>(op);
            auto inputVal = lutOp.getInputs().back();
            auto outputVal = lutOp.getResult();
            assertValExist(inputVal);
            insertActiveVal(outputVal);

            break;
          }

          case CGToucanOPName::Print: {
            assert(op != nullptr);
            assert(opCount == 1);
            auto printOp = cast<toucan::PrintOp>(op);
            info.allPrints.push_back(printOp);

            assertValExist(printOp.getEn());
            break;
          }
          case CGToucanOPName::Stop: {
            assert(op != nullptr);
            assert(opCount == 1);
            auto stopOp = cast<toucan::StopOp>(op);
            info.allStops.push_back(stopOp);

            assertValExist(stopOp.getEn());
            break;
          }
          case CGToucanOPName::RegRead: {
            assert(op != nullptr);
            assert(opCount == 1);
            auto regReadOp = cast<toucan::RegReadOp>(op);
            info.allRegReads.push_back(regReadOp);

            insertActiveVal(regReadOp.getResult());
            break;
          }
          case CGToucanOPName::RegWrite: {
            assert(op != nullptr);
            assert(opCount == 1);
            auto regWriteOp = cast<toucan::RegWriteOp>(op);
            info.allRegWrites.push_back(regWriteOp);

            assertValExist(regWriteOp.getData());
            break;
          }
          case CGToucanOPName::MemWrite: {
            assert(op != nullptr);
            assert(isa<toucan::MemWriteOp>(op));

            // mem write could be merged nodes
            // collect all of them

            uint32_t realOpCount = 0;
            auto oneMWOp = cast<toucan::MemWriteOp>(op);
            auto memVal = oneMWOp.getMem();

            for (auto eachUserOp: memVal.getUsers()) {
              if (auto mwOp = dyn_cast<toucan::MemWriteOp>(eachUserOp)) {
                realOpCount += 1;
                info.allMemWrites.push_back(mwOp);

                assertValExist(mwOp.getEn());
                assertValExist(mwOp.getAddrVec());
                assertValExist(mwOp.getData());
              }
            }
            assert(realOpCount == opCount);

            break;
          }

          case CGToucanOPName::ExchangeRead: {
            assert(opCount == 1);
            auto exgValId = partGraph[vtx].exchangeValId;
            assert(exgValId != UINT32_MAX);
            auto exgVal = exchangeValPool[exgValId];
            info.allExgReadVals.push_back(exgVal);

            insertActiveVal(exgVal);
            break;
          }
          case CGToucanOPName::ExchangeWrite: {
            assert(opCount == 1);
            auto exgValId = partGraph[vtx].exchangeValId;
            assert(exgValId != UINT32_MAX);
            auto exgVal = exchangeValPool[exgValId];
            info.allExgWriteVals.push_back(exgVal);

            assertValExist(exgVal);
            break;
          }

          default: {
            llvm_unreachable("Unexpected node type");
          }
        }
      }

      // Merge inserted NOPs into mParts
      if (!nopsInThisLevel.empty()) {
        assert(level_id == 1);
        
        mlir::SmallVector<toucan::LUTOp> nopsInThisPart;

        for (size_t i = 0; i < nopsInThisLevel.size(); i++) {
          nopsInThisPart.push_back(nopsInThisLevel[i]);

          if (nopsInThisPart.size() >= 32 || (i+1) == nopsInThisLevel.size()) {
            auto newMP = std::make_shared<MicroPart>();
            newMP->buildNOPRegularLUTPart(nopsInThisPart);
            mpsThisLevel.push_back(newMP);
            nopsInThisPart.clear();
          }
        }
        assert(nopsInThisPart.empty());
      }

      // Merge special mParts
      if (!specialMPartsInThisLevel.empty()) {
        // pack special mparts
        mlir::SmallVector<mlir::SmallVector<std::shared_ptr<MicroPart>>> mpartGroupByType;
        mlir::SmallVector<std::shared_ptr<MicroPart>> mPartToMerge;

        mpartGroupByType.resize(getMaxEnumValForCGToucanOPName());

        for (auto &eachMp: specialMPartsInThisLevel) {
          auto opType = eachMp->opType;
          auto opTypeInInt = static_cast<int>(opType);

          mpartGroupByType[opTypeInInt].push_back(eachMp);
        }
        
        for (uint32_t i = 0; i < mpartGroupByType.size(); i++) {
          // auto opType = static_cast<CGToucanOPName>(i);
          auto mParts = mpartGroupByType[i];
          if (mParts.empty()) continue;

          mPartToMerge.clear();

          for (size_t i = 0; i < mParts.size(); i++) {
            mPartToMerge.push_back(mParts[i]);

            if (mPartToMerge.size() >= 32 || (i+1) == mParts.size()) {
              auto newMP = std::make_shared<MicroPart>();
              newMP->mergeSpecialPartFromOtherParts(mPartToMerge);
              mpsThisLevel.push_back(newMP);
              mPartToMerge.clear();
            }
          }
          assert(mPartToMerge.empty());
        }
      }

      if (!mpsThisLevel.empty()) {
        info.mpartLevels.emplace_back(std::move(mpsThisLevel));
      }
    }

    for (auto &regReadOp: info.allRegReads) {
      info.readRegs.push_back(regReadOp.getReg());
    }

    for (auto &regWriteOp: info.allRegWrites) {
      info.writeRegs.push_back(regWriteOp.getReg());
    }

    // check: cannot have duplicated reg/exg read/writes
    mlir::DenseSet<mlir::Value> valSet;
    valSet.insert(info.allExgReadVals.begin(), info.allExgReadVals.end());
    assert(valSet.size() == info.allExgReadVals.size());

    valSet.clear();
    valSet.insert(info.allExgWriteVals.begin(), info.allExgWriteVals.end());
    assert(valSet.size() == info.allExgWriteVals.size());

    valSet.clear();
    valSet.insert(info.readRegs.begin(), info.readRegs.end());
    assert(valSet.size() == info.readRegs.size());

    valSet.clear();
    valSet.insert(info.writeRegs.begin(), info.writeRegs.end());
    assert(valSet.size() == info.writeRegs.size());
  };


  partCodeGenData_r0.clear();
  partCodeGenData_r0.resize(repcutPartitions_r0.size());
  for (size_t i = 0; i < repcutPartitions_r0.size(); i++) {
    // llvm::dbgs() << "Working on region 0 part " << i << "\n";
    // parallel?
    getCodeGenData(
      partCodeGenData_r0[i], 
      *microPartGraph_r0, 
      repcutPartitions_r0[i], 
      microPartGraph_r0_levels, 
      exchangeValPool
    );
  }

  partCodeGenData_r1.clear();
  partCodeGenData_r1.resize(repcutPartitions_r1.size());
  for (size_t i = 0; i < repcutPartitions_r1.size(); i++) {
    // llvm::dbgs() << "Working on region 1 part " << i << "\n";
    // parallel?
    getCodeGenData(
      partCodeGenData_r1[i], 
      *microPartGraph_r1, 
      repcutPartitions_r1[i], 
      microPartGraph_r1_levels, 
      exchangeValPool
    );
  }
}



mlir::DenseMap<uint32_t, mlir::SmallVector<uint32_t>> PartitioningManager::collectGraphVectorDeclInfoToFile(const PartitioningGraph &rawGraph) {

  mlir::DenseMap<uint32_t, mlir::SmallVector<uint32_t>> originalVectorElementsMap;

  for (auto vtx: boost::make_iterator_range((boost::vertices(rawGraph)))) {
    auto vtxOpName = rawGraph[vtx].toucanOpName;

    if (vtxOpName == CGToucanOPName::VecDecl) {
      // A vector decl
      auto vecDeclOp = dyn_cast<toucan::DefVectorOp>(rawGraph[vtx].op);
      assert(vecDeclOp != nullptr);

      mlir::DenseMap<mlir::Value, uint32_t> vecInputValToOpId;
      // check input edges (that ultimately forms the vector)
      for (auto ei = boost::in_edges(vtx, rawGraph); ei.first != ei.second; ++ei.first) {
        auto inputNodeId = boost::source(*ei.first, rawGraph);
        auto inputNodeOp = rawGraph[inputNodeId].op;
        auto inputNodeOpName = rawGraph[inputNodeId].toucanOpName;

        assert(inputNodeOpName != CGToucanOPName::VecDecl);

        if (inputNodeOpName == CGToucanOPName::VecRead) {
          // May be a bunch of VecRead
          auto inputVector = cast<toucan::VectorReadOp>(inputNodeOp).getHandle();
          for (auto userOp: inputVector.getUsers()) {
            auto mergedVecReadOp = cast<toucan::VectorReadOp>(userOp);
            auto resultVal = mergedVecReadOp.getResult();

            // It's OK to see same value many times, as we allow parallel edge
            if (vecInputValToOpId.contains(resultVal)) {
              assert(vecInputValToOpId[resultVal] == inputNodeId);
            }

            vecInputValToOpId[resultVal] = inputNodeId;
          }
        } else if (inputNodeOpName == CGToucanOPName::VecArith) {
          for (auto userOp: inputNodeOp->getUsers()) {
            // Each user should be StaticVectorSegmentReadOp
            auto segReadOp = cast<toucan::StaticVectorSegmentReadOp>(userOp);
            auto resultVal = segReadOp.getResult();

            // It's OK to see same value many times, as we allow parallel edge
            if (vecInputValToOpId.contains(resultVal)) {
              assert(vecInputValToOpId[resultVal] == inputNodeId);
            }

            vecInputValToOpId[resultVal] = inputNodeId;
          }
        } else {
          // any better way?
          assert(inputNodeOp != nullptr);
          auto resultVal = inputNodeOp->getUses().begin()->get();

          if (vecInputValToOpId.contains(resultVal)) {
            // Tolerate identical input edges
            assert(vecInputValToOpId[resultVal] == inputNodeId);
          }

          vecInputValToOpId[resultVal] = inputNodeId;
        }
      }

      assert(!originalVectorElementsMap.contains(vtx));
      originalVectorElementsMap[vtx] = {};
      for (auto inputVal: vecDeclOp.getInputs()) {
        assert(vecInputValToOpId.contains(inputVal));
        auto sourceVtx = vecInputValToOpId[inputVal];
        originalVectorElementsMap[vtx].push_back(sourceVtx);
      }
    }
  }

  return originalVectorElementsMap;
}




LogicalResult PartitioningManager::dumpGraphVectorDeclInfoToFile(std::string fileName, const mlir::DenseMap<uint32_t, mlir::SmallVector<uint32_t>> &originalVectorElementsMap) {

  auto ofs = std::ofstream(fileName, std::ios::out | std::ios::trunc);

  if (!ofs.is_open()) {
    llvm::errs() << "Cannot open file for write: " << fileName << "\n";
    return failure();
  }

  ofs << "Format:\nVecDecl_node_id Vector_element_id_0 Vector_element_id_1 ...\n";

  for (auto &kv: originalVectorElementsMap) {
    auto vtx = kv.first;
    auto elems = kv.second;
    assert(elems.size() != 0);

    ofs << vtx;
    for (auto sourceVtx: kv.second) {
      ofs << " " << sourceVtx;
    }
    ofs << "\n";
  }

  ofs.close();
  return success();
}




// Dump graph for MicroPartitioner use
// Note: Different from dumpGraphToFile!!!
// Partially allow parallel edge: parallel edge from VecRead/VecArith implies
//    multiple value read from a merged node
LogicalResult PartitioningManager::dumpGraphToFileForMicroPartitioner(const PartitioningGraph &g, mlir::SmallVector<uint32_t> partNodes, std::string fileName) {
  mlir::DenseSet<uint32_t> partNodeSet;
  for (auto n: partNodes) {
    partNodeSet.insert(n);
  }

  
  std::ostringstream oss;
  size_t numValidVtxes = 0, numValidEdges = 0;


  for (auto vtx: boost::make_iterator_range((boost::vertices(g)))) {
    if (partNodeSet.contains(vtx)) {
      // valid node
      numValidVtxes++;
      auto tOpName = g[vtx].toucanOpName;

      mlir::SmallVector<uint32_t> outNeighs;

      if (tOpName == CGToucanOPName::VecArith || tOpName == CGToucanOPName::VecRead) {
        // Use parallel edge to represent multiple reads from same merged node (but different ops)
        auto rawOp = g[vtx].op;
        assert(rawOp != nullptr);
        mlir::DenseSet<mlir::Value> allResultValueOfThisNode;

        // Collect output values for the same merged node
        if (auto vecReadOp = dyn_cast<toucan::VectorReadOp>(rawOp)) {
          assert(tOpName == CGToucanOPName::VecRead);
          auto vecVal = vecReadOp.getHandle();

          for (auto userOp: vecVal.getUsers()) {
            if (auto vecReadOp = dyn_cast<toucan::VectorReadOp>(userOp)) {
              allResultValueOfThisNode.insert(vecReadOp.getResult());
            }
          }
        } else {
          assert(tOpName == CGToucanOPName::VecArith);
          auto vecArithOp = cast<toucan::VectorArithOp>(rawOp);
          auto vecVal = vecArithOp.getResult();
          for (auto userOp: vecVal.getUsers()) {
            if (auto staticReadOp = dyn_cast<toucan::StaticVectorSegmentReadOp>(userOp)) {
              allResultValueOfThisNode.insert(staticReadOp.getResult());
            }
          }
        }

        mlir::DenseSet<uint32_t> visitedEdges;
        for (auto ei = boost::out_edges(vtx, g); ei.first != ei.second; ++ei.first) {
          auto v = boost::target(*ei.first, g);

          if (visitedEdges.contains(v)) {
            // Dont visit parallel edge, they are recalculated later
            continue;
          }
          visitedEdges.insert(v);

          // Use parallel edge to represent multiple reads from a same merged node
          // This allows MicroPartitioner correctly track read count
          uint32_t readCount = 0;

          mlir::SmallVector<mlir::Operation*> targetRawOps;
          auto _targetRawOp = g[v].op;
          switch (g[v].toucanOpName) {
            case toucan::CGToucanOPName::VecRead: {
              uint32_t thisNodeOpCount = 0;
              auto vecReadOp = cast<toucan::VectorReadOp>(_targetRawOp);
              auto vecVal = vecReadOp.getHandle();

              for (auto userOp: vecVal.getUsers()) {
                if (isa<toucan::VectorReadOp>(userOp)) {
                  targetRawOps.push_back(userOp);
                  thisNodeOpCount++;
                }
              }

              assert(thisNodeOpCount == g[v].opCount);
              break;
            }

            case toucan::CGToucanOPName::MemWrite: {
              uint32_t thisNodeOpCount = 0;
              assert(isa<toucan::MemWriteOp>(_targetRawOp));
              auto mwOp = cast<toucan::MemWriteOp>(_targetRawOp);
              auto memVal = mwOp.getMem();

              for (auto userOp: memVal.getUsers()) {
                if (isa<toucan::MemWriteOp>(userOp)) {
                  targetRawOps.push_back(userOp);
                  thisNodeOpCount++;
                }
              }

              assert(thisNodeOpCount == g[v].opCount);
              break;
            }
            default: {
              // Note: do nothing for VecArith following VecStaticSegRead, as they only read from producing result of VecArith
              targetRawOps.push_back(_targetRawOp);
            }
          }

          for (auto targetRawOp: targetRawOps) {
            if (targetRawOp == nullptr) {
              readCount = 1;
            } else {
              for (const auto eachOperand: targetRawOp->getOperands()) {
                if (allResultValueOfThisNode.contains(eachOperand)) {
                  readCount += 1;
                }
              }
            }
          }

          assert(readCount >= 1);
          for (size_t i = 0; i < readCount; i++) {
            outNeighs.push_back(v);
          }
        }
      } else {
        // Otherwise, dedup same edge
        mlir::DenseSet<uint32_t> outNeighsSet;
        for (auto ei = boost::out_edges(vtx, g); ei.first != ei.second; ++ei.first) {
          auto v = boost::target(*ei.first, g);
          outNeighsSet.insert(v);
        }
        outNeighs.assign(outNeighsSet.begin(), outNeighsSet.end());
      }


      mlir::SmallVector<uint32_t> validOutNeighs;
      for (auto &target: outNeighs) {
        if (partNodeSet.contains(target)) {
          validOutNeighs.push_back(target);
        }
      }

      numValidEdges += validOutNeighs.size();

      oss << vtx << ' ';
      oss << stringifyCGToucanOPName(tOpName);
      oss << ' ' << g[vtx].weight;
      for (const auto v: validOutNeighs) {
        oss << ' ' << v;
      }
      oss << "\n";
    }
  }

  assert(numValidVtxes == partNodeSet.size());


  auto ofs = std::ofstream(fileName, std::ios::out | std::ios::trunc);
  if (!ofs.is_open()) {
    llvm::errs() << "Cannot open file for write: " << fileName << "\n";
    return failure();
  }
  ofs << numValidEdges << ' ' << numValidVtxes << "\n";
  ofs << oss.str();

  ofs.close();

  return success();
}



static void getVtxToLevel(const PartitioningGraph &g, mlir::SmallVector<uint32_t> &levels, uint32_t maxVtxId) {
  std::vector<uint32_t> topo_order;
  mlir::SmallVector<uint32_t> sinkVtxes;

  topo_order.reserve(boost::num_vertices(g));
  boost::topological_sort(g, std::back_inserter(topo_order));
  std::reverse(topo_order.begin(), topo_order.end());

  // Initialize levels
  levels.clear();
  levels.resize(maxVtxId, 0);
  uint32_t maxLevel = 0;

  // Assign levels based on dependencies. Ignore sink vtxes
  for (auto v : topo_order) {
    if (boost::out_degree(v, g) == 0) {
      // a sink node
      sinkVtxes.push_back(v);
    } else if (boost::in_degree(v, g) != 0) {
      uint32_t max_pred_level = 0;
      for (auto ei = boost::in_edges(v, g); ei.first != ei.second; ++ei.first) {
        auto u = boost::source(*ei.first, g);
        max_pred_level = std::max(max_pred_level, levels[u]);
      }
      uint32_t v_level = max_pred_level + 1;
      levels[v] = v_level;
      maxLevel = std::max(maxLevel, v_level);
      assert(v_level < UINT32_MAX);
    }
  }

  assert(!sinkVtxes.empty());

  // move all sinkVtx to last level
  for (auto v: sinkVtxes) {
    levels[v] = maxLevel + 1;
  }
}

void PartitioningManager::levelizeGraph(const PartitioningGraph &g, mlir::SmallVector<mlir::SmallVector<uint32_t>> &graphLevels) const {

  mlir::SmallVector<uint32_t> levels;
  getVtxToLevel(g, levels, boost::num_vertices(g));


  for (uint32_t vtx = 0; vtx < levels.size(); vtx++) {
    uint32_t vtxLevel = levels[vtx];
    assert(vtxLevel != UINT32_MAX);
    while (graphLevels.size() <= vtxLevel) {
      graphLevels.emplace_back();
    }
    graphLevels[vtxLevel].push_back(vtx);
  }

  // Every level should have at least 1 nodes
  for (const auto &eachLevel: graphLevels) {
    assert(!eachLevel.empty());
  }

}
