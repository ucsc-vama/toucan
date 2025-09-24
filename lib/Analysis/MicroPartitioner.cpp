#include "toucan/MicroPartitioner.h"

#include "mlir/IR/Diagnostics.h"
#include "mlir/IR/MLIRContext.h"
#include "mlir/IR/Threading.h"
#include "mlir/IR/Operation.h"
#include "mlir/IR/Value.h"
#include "mlir/Support/LLVM.h"
#include "mlir/Support/LogicalResult.h"
#include "toucan/PartitioningGraph.h"
#include "toucan/ToucanOps.h"
#include "toucan/ToucanTypes.h"
#include "llvm/Support/Debug.h"
#include "llvm/Support/ErrorHandling.h"
#include "llvm/Support/Program.h"
#include "llvm/Support/raw_ostream.h"

#include <boost/algorithm/string.hpp>
#include <cstddef>
#include <fstream>
#include <memory>
#include <sstream>
#include <string>

#include "toucan/ToucanConfigs.h"

using namespace toucan;
using namespace mlir;
using namespace llvm;


void MicroPartitioner::mergeSpecialMParts(const size_t maxOpsPerMPart) {
  llvm_unreachable("Under construction");
}


LogicalResult MicroPartitioner::arrangeSpecialOps(const PartitioningGraph &g, const size_t maxOpsPerMPart) {

  assert(maxOpsPerMPart <= 32);
  assert(partLevels.size() == excludeNodeLevels.size());

  auto numLevels = partLevels.size();
  auto maxOpToInt = getMaxEnumValForCGToucanOPName();

  mlir::SmallVector<mlir::SmallVector<uint32_t>> excludeNodeGroupedByOp;
  excludeNodeGroupedByOp.resize(maxOpToInt + 1);

  mlir::SmallVector<mlir::Operation*> specialOps;



  for (size_t levelId = 0; levelId < numLevels; levelId++) {
    for (int i = 0; i < maxOpToInt; i++) {
      excludeNodeGroupedByOp[i].clear();
    }

    for (auto eachVtx: excludeNodeLevels[levelId]) {
      auto vtxOpName = g[eachVtx].toucanOpName;
      auto vtxOpName_int = static_cast<int>(vtxOpName);

      assert(vtxOpName_int >= 0);
      assert(vtxOpName_int <= maxOpToInt);

      excludeNodeGroupedByOp[vtxOpName_int].push_back(eachVtx);
    }

    for (int vtxOpName_int = 0; vtxOpName_int <= maxOpToInt; vtxOpName_int++) {
      if (excludeNodeGroupedByOp[vtxOpName_int].empty()) continue;

      auto vtxOpName = static_cast<CGToucanOPName>(vtxOpName_int);
      auto &vtxes = excludeNodeGroupedByOp[vtxOpName_int];


      switch (vtxOpName) {

        case CGToucanOPName::VecDecl:
        case CGToucanOPName::LUT: {
          // should not appear
          llvm::errs() << "Op type " << stringifyCGToucanOPName(vtxOpName) << " should not appear as excluded\n";
          return failure();
        }

        case CGToucanOPName::VecStaticRead:
        case CGToucanOPName::Dummy_DefReg:
        case CGToucanOPName::Dummy_DefMem:{
          // should not appear
          llvm::errs() << "Op type " << stringifyCGToucanOPName(vtxOpName) << " should not appear in graph\n";
          return failure();
        }

        case CGToucanOPName::ConstDecl: {
          // Do nothing for const decl. They don't execute
          break;
        }

        case CGToucanOPName::VecRead:
        case CGToucanOPName::VecLogic:
        case CGToucanOPName::VecArith:
        case CGToucanOPName::MemRead: {
          // Note: MemWrite, VecRead nodes could be merged
          // For those 4 ops, each part can only have 1 level
          mlir::SmallVector<NodeIdAndOpCount> idAndOpCount;

          for (auto eachVtx: vtxes) {
            auto opCount = g[eachVtx].opCount;
            idAndOpCount.push_back({eachVtx, opCount});
          }
          std::sort(idAndOpCount.begin(), idAndOpCount.end(), [](const NodeIdAndOpCount&a, const NodeIdAndOpCount&b) {return a.opCount > b.opCount; });


          specialOps.clear();
          for (auto [eachVtx, opCount]: idAndOpCount) {
            auto rawOp = g[eachVtx].op;
            assert(rawOp != nullptr);
            if (opCount == 1) {
              specialOps.push_back(rawOp);
            } else {
              // multiple
              uint32_t thisNodeOpCount = 0;
              if (auto vecReadOp = dyn_cast<toucan::VectorReadOp>(rawOp)) {
                auto vecVal = vecReadOp.getHandle();

                for (auto userOp: vecVal.getUsers()) {
                  if (isa<toucan::VectorReadOp>(userOp)) {
                    specialOps.push_back(userOp);
                    thisNodeOpCount++;
                  }
                }
              } else {
                llvm_unreachable("Unexpected node with opCount != 1");
              }
              assert(thisNodeOpCount == opCount);
            }
          }

          // for every maxOpsPerMPart
          mlir::SmallVector<mlir::Operation*> thisPartOps;
          for (size_t i = 0; i < specialOps.size(); i++) {
            thisPartOps.push_back(specialOps[i]);

            if (thisPartOps.size() == maxOpsPerMPart || i+1 == specialOps.size()) {
              auto newPart = std::make_shared<MicroPart>();
              partLevels.back().emplace_back(newPart);
              newPart->buildSpecialPart(vtxOpName, thisPartOps);
              newPart->lineno = UINT32_MAX;
              newPart->partId = static_cast<uint32_t>(partId);
              thisPartOps.clear();
            }
          }
          assert(thisPartOps.size() == 0);

          break;
        }

        case CGToucanOPName::RegRead: {
          // Reg read should only appear in first level
          assert(levelId == 0);
          allRegReads.append(vtxes);
          break;
        }
        case CGToucanOPName::Print: {
          // move to later (last level)
          allPrints.append(vtxes);
          break;
        }
        case CGToucanOPName::Stop: {
          // move to later (last level)
          allStops.append(vtxes);
          break;
        }
        case CGToucanOPName::RegWrite: {
          // move to later (last level)
          allRegWrites.append(vtxes);
          break;
        }
        case CGToucanOPName::MemWrite: {
          // move to later (last level)
          allMemWrites.append(vtxes);
          break;
        }
        
        case CGToucanOPName::ExchangeRead:
        case CGToucanOPName::ExchangeWrite: {
          llvm::errs() << "ExchangeRead and ExchangeWrite is not supported yet!\n";
          return failure();
        }

        default: {
          // should not appear
          llvm::errs() << "Unknow op\n";
          return failure();
        }
      }
    }
  }

  return success();
}

mlir::LogicalResult MicroPartitioner::partition() {
  auto ret = callExternalPartitioner();
  if (failed(ret)) return ret;

  ret = loadMicroParts();
  if (failed(ret)) return ret;

  ret = loadVectorNopMap();
  if (failed(ret)) return ret;

  uint32_t totalNumParts = 0;
  for (const auto &level: partLevels) {
    totalNumParts += level.size();
  }

  std::ostringstream oss;
  oss << "Has " << totalNumParts << " micro parts, " << partLevels.size() << " micro part levels\n";

  llvm::outs() << oss.str();

  return mlir::success();
};


mlir::LogicalResult MicroPartitioner::callExternalPartitioner() {
  if (!std::filesystem::exists(inputGraphFile)) {
    llvm::errs() << "File " << inputGraphFile << " does not exist!\n";
    llvm_unreachable("Micro partitioner input file doesn't exists! This should not happen");
  }
  if (!std::filesystem::exists(graphVectorInfoFile)) {
    llvm::errs() << "File " << graphVectorInfoFile << " does not exist!\n";
    llvm_unreachable("Graph vector info file doesn't exists! This should not happen");
  }

  std::string mPartMaxSizeString = std::to_string(MICRO_PARTITIONER_MAX_PART_SIZE);

  llvm::StringRef args[] = {
    microPartitionerBin,
    "--graph",
    inputGraphFile,
    "--vector",
    graphVectorInfoFile,
    "--output",
    outputFile,
    "--vecmap",
    outputVectorMapFile,
    "--max-part-size",
    mPartMaxSizeString
  };

  std::optional<llvm::StringRef> redirects[] = {
    std::nullopt,
    consoleLogFile,
    consoleLogFile
  };


  auto mpartExe = llvm::sys::findProgramByName(microPartitionerBin);
  if (!mpartExe) {
    llvm::errs() << "Cannot find " << microPartitionerBin << ". Please ensure it's in your PATH!\n";
    return failure();
  }

  int result = llvm::sys::ExecuteAndWait(*mpartExe, args, std::nullopt, redirects);

  if (result != 0) {
    llvm::errs() << "MicroPart partitioner returns non-zero code: " << result << "\n";
    llvm::errs() << "Partitioner at: " << mpartExe.get() << "\n";
    for (const auto &eachArg: args) {
      llvm::errs() << eachArg << " ";
    }
    llvm::errs() << "\n";

    return failure();
  }

  return success();
}


static bool testIsInteger(const std::string& s) {
  if (s.empty()) return false;
  size_t start = 0;
  if (s[0] == '-' || s[0] == '+') { // Handle signs
      start = 1;
      if (s.size() == 1) return false; // Only a sign is invalid
  }
  for (size_t i = start; i < s.size(); i++) {
      if (!isdigit(s[i])) return false;
  }
  return true;
}

static bool checkIsValidIntegerTokenAndReport(uint32_t lineno, std::string& s) {
  if (!testIsInteger(s)) {
    errs() << "Error on stoi at line " << lineno << ": [" << s << "]\n";
    return false;
  }
  return true;
}

mlir::LogicalResult MicroPartitioner::loadMicroParts() {
  if (!std::filesystem::exists(outputFile)) {
    llvm_unreachable("Micro partitioner output file doesn't exists! This should not happen");
  }

  std::ifstream file(outputFile);
  std::string line;
  uint32_t lineno = 0;
  int levelId = -1;
  assert(partLevels.empty());
  partLevels.clear();
  excludeNodeLevels.clear();

  std::vector<std::string> split_line;
  mlir::SmallVector<mlir::SmallVector<uint32_t>> newPartNodesLevel;

  mlir::DenseSet<uint32_t> allNodesInMP, allNodesExclude;

  while (std::getline(file, line)) {
    split_line.clear();
    boost::split(split_line, line, boost::is_any_of(" "));

    if (split_line[0].size() != 1) return failure();

    if (split_line[0] == "L") {
      // a new level
      if (split_line.size() != 2) return failure();

      checkIsValidIntegerTokenAndReport(lineno, split_line[1]);
      levelId = std::stoi(split_line[1]);
      if (levelId != static_cast<int>(partLevels.size())) return failure();

      partLevels.emplace_back();
      // avoid frequent reallocation.
      partLevels.back().reserve(512);
      excludeNodeLevels.emplace_back();
    } else if (split_line[0] == "e") {
      // Exclude nodes
      if (split_line.size() < 2) return failure();

      mlir::SmallVector<uint32_t> thisLevelExcludeNodes;
      for (size_t i = 1; i < split_line.size(); i++) {
        if (!checkIsValidIntegerTokenAndReport(lineno, split_line[i])) {
          // something is wrong!
          errs() << "Error on exclude part parsing\n";
          for (auto &t: split_line) {
            errs() << "[" << t << "] ";
          }
          errs() << "\n";
          errs() << outputFile << "\n";
          return failure();
        }
        assert(!split_line[i].empty());
        auto v = std::stoi(split_line[i]);
        thisLevelExcludeNodes.push_back(v);

        assert(!allNodesExclude.contains(v) && "Duplicated node");
        allNodesExclude.insert(v);
      }

      assert(levelId + 1 == static_cast<int>(excludeNodeLevels.size()));
      excludeNodeLevels[levelId].swap(thisLevelExcludeNodes);
    } else if (split_line[0] == "n") {
      // Normal partition
      assert(levelId + 1 == static_cast<int>(partLevels.size()));

      newPartNodesLevel.clear();
      for (size_t i = 1; i < split_line.size(); i++) {
        if (split_line[i] == "l") {
          // a new level
          newPartNodesLevel.emplace_back();
        } else {
          // a node
          assert(!newPartNodesLevel.empty());
          if (!checkIsValidIntegerTokenAndReport(lineno, split_line[i])) {
            // something is wrong!
            errs() << "Error on normal part parsing\n";
            for (auto &t: split_line) {
              errs() << "[" << t << "] ";
            }
            errs() << "\n";
            errs() << outputFile << "\n";
            return failure();
          }
          assert(!split_line[i].empty());
          auto v = std::stoi(split_line[i]);
          newPartNodesLevel.back().push_back(v);

          assert(!allNodesInMP.contains(v) && "Duplicated node!");
          allNodesInMP.insert(v);
        }
      }

      auto newPart = std::make_shared<MicroPart>();
      partLevels[levelId].push_back(newPart);
      newPart->buildRegularLUTPart(newPartNodesLevel);
      newPart->lineno = lineno;
      newPart->partId = static_cast<uint32_t>(partId);
    } else {
      llvm::errs() << "Cannot parse line\n" << line;
      return failure();
    }
    lineno ++;
  }

  // Note: It's possible to collect more nodes than repcut nodes, since some VecDecl nodes are converted to multiple NOPs by MicroPart partitioner

  return success();
}

mlir::LogicalResult MicroPartitioner::loadVectorNopMap() {
  if (!std::filesystem::exists(outputVectorMapFile)) {
    llvm_unreachable("Micro partitioner vector NOP map output file doesn't exists! This should not happen");
  }

  std::ifstream file(outputVectorMapFile);
  std::string line;
  uint32_t lineno = 0;
  outputVectorNopMap.clear();

  std::vector<std::string> split_line;
  std::vector<uint32_t> lineValues;

  while (std::getline(file, line)) {
    if (line.empty()) return failure();

    split_line.clear();
    boost::split(split_line, line, boost::is_any_of(" "));
    if (split_line.size() <= 1) return failure();

    lineValues.clear();
    for (auto eachToken: split_line) {
      if (!checkIsValidIntegerTokenAndReport(lineno, eachToken)) {
        // something is wrong!
        errs() << "Error on parsing\nTokens:";
        for (auto &t: split_line) {
          errs() << "[" << t << "] ";
        }
        errs() << "\n";
        return failure();
      }
      assert(!eachToken.empty());
      lineValues.push_back(std::stoi(eachToken));
    }
    assert(lineValues.size() > 1);

    auto vecDeclNodeId = lineValues[0];
    outputVectorNopMap[vecDeclNodeId] = {};
    for (size_t i = 1; i < lineValues.size(); i++) {
      outputVectorNopMap[vecDeclNodeId].push_back(lineValues[i]);
    }

    lineno++;
  }

  return success();
}

void MicroPartitioner::collectPartIOValues(mlir::MLIRContext *context, const PartitioningGraph &g) {
  // 1. find map from new node id to original vecDecl Id

  for (const auto &[vecDeclId, vecNewNodes]: outputVectorNopMap) {
    assert(originalVectorElementsMap.contains(vecDeclId));

    for (const auto &eachNewNode: vecNewNodes) {
      // Note: it's possible that eachNewNode exists in g.g due to name conflict: 
      // each partition only have partial view of the original graph
      newNodeIdToOriginalVecDeclId[eachNewNode] = vecDeclId;
    }

    auto numVecElements = vecNewNodes.size();
    assert(originalVectorElementsMap[vecDeclId].size() == numVecElements);
    for (size_t i = 0; i < numVecElements; i++) {
      auto newNodeId = vecNewNodes[i];
      auto depNodeId = originalVectorElementsMap[vecDeclId][i];

      newNodeIdToDepNodeId[newNodeId] = depNodeId;
    }
  }

  // 2. collect for all parts
  // uint32_t levelId = 0;
  // consider parallel
  for (auto &eachPartLevel: partLevels) {
    size_t numMPartThisLevel = eachPartLevel.size();

    auto allPartGood = mlir::failableParallelForEachN(context, 0, numMPartThisLevel, [&](size_t partIndexInThisLevel) {
      auto eachPart = eachPartLevel[partIndexInThisLevel];

      auto partGood = eachPart->checkAndCollectIOValues(g, allNodes, newNodeIdToDepNodeId, newNodeIdToOriginalVecDeclId, outputVectorNopMap);

      if (!partGood) {
        llvm::errs() << "Error when checking micro parts\n";
        return failure();
      }
      return success();
    });

    if (failed(allPartGood)) {
      // fail!!
      llvm_unreachable("Should not happen");
    }
  }
}

