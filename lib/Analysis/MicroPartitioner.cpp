#include "toucan/MicroPartitioner.h"

#include "mlir/IR/Diagnostics.h"
#include "mlir/Support/LLVM.h"
#include "mlir/Support/LogicalResult.h"
#include "toucan/PartitioningGraph.h"
#include "llvm/Support/ErrorHandling.h"
#include "llvm/Support/Program.h"
#include "llvm/Support/raw_ostream.h"

#include <boost/algorithm/string.hpp>
#include <cstddef>
#include <fstream>
#include <string>

using namespace toucan;
using namespace mlir;
using namespace llvm;







static void packNodes(const mlir::SmallVector<NodeIdAndOpCount>& nodesSorted,
 mlir::SmallVector<mlir::SmallVector<NodeIdAndOpCount>>& resultPacks) {
  // Validate input sorting in debug builds
#ifndef NDEBUG
  for (size_t i = 1; i < nodesSorted.size(); ++i) {
    assert(nodesSorted[i-1].opCount >= nodesSorted[i].opCount &&
           "Input must be sorted in descending opCount order");
  }
#endif

  resultPacks.clear();
  mlir::SmallVector<uint32_t> packRemaining;

  for (const auto& node : nodesSorted) {
    uint32_t remaining = node.opCount;
    
    // First try to place full node in existing packs
    for (size_t i = 0; i < packRemaining.size() && remaining > 0; ++i) {
      if (packRemaining[i] >= remaining) {
        resultPacks[i].push_back({node.node, remaining});
        packRemaining[i] -= remaining;
        remaining = 0;
        break;
      }
    }

    // Split remaining across packs if needed
    while (remaining > 0) {
      uint32_t allocated = 0;
      
      // Try to fill existing packs
      for (size_t i = 0; i < packRemaining.size() && remaining > 0; ++i) {
        const uint32_t chunk = std::min(remaining, packRemaining[i]);
        if (chunk > 0) {
          resultPacks[i].push_back({node.node, chunk});
          packRemaining[i] -= chunk;
          remaining -= chunk;
          allocated += chunk;
        }
      }

      // Create new packs for remaining
      if (remaining > 0) {
        const uint32_t chunk = std::min(remaining, 32u);
        resultPacks.push_back({{node.node, chunk}});
        packRemaining.push_back(32 - chunk);
        remaining -= chunk;
        allocated += chunk;
      }

      if (allocated == 0) break; // Prevent infinite loop
    }
  }
}

static void printPackingResult(const mlir::SmallVector<mlir::SmallVector<NodeIdAndOpCount>>& resultPacks) {
  using namespace llvm;
  
  outs() << "Packing result (" << resultPacks.size() << " packs):\n";
  
  // Track node distribution
  std::map<uint32_t, SmallVector<uint32_t>> nodeParts;

  for (size_t i = 0; i < resultPacks.size(); ++i) {
    outs() << "Pack " << i << ":\n";
    for (const auto& item : resultPacks[i]) {
      outs() << "  Node " << item.node << ": " << item.opCount << " ops\n";
      nodeParts[item.node].push_back(item.opCount);
    }
  }

  // Print node splits
  outs() << "\nNode operations:\n";
  for (const auto& [node, parts] : nodeParts) {
    outs() << "Node " << node << ": ";
    uint32_t total = 0;
    for (size_t i = 0; i < parts.size(); ++i) {
      if (i > 0) outs() << " + ";
      outs() << parts[i];
      total += parts[i];
    }
    outs() << " = " << total << "\n";
  }
}

LogicalResult MicroPartitioner::arrangeSpecialOps(PartitioningGraph &g) {


  assert(partLevels.size() == excludeNodeLevels.size());

  auto numLevels = partLevels.size();
  auto maxOpToInt = getMaxEnumValForCGToucanOPName();

  mlir::SmallVector<mlir::SmallVector<uint32_t>> excludeNodeGroupedByOp;
  excludeNodeGroupedByOp.resize(maxOpToInt + 1);



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

          // TODO: Am I right? Ensure descending
          for (size_t i = 0; i < idAndOpCount.size() - 1; i++) {
            assert(idAndOpCount[i].opCount >= idAndOpCount[i+1].opCount);
          }

          mlir::SmallVector<mlir::SmallVector<NodeIdAndOpCount>> resultPacks;
          packNodes(idAndOpCount, resultPacks);
          outs() << stringifyCGToucanOPName(vtxOpName) << ":\n";
          printPackingResult(resultPacks);


          // create new partitions
          for (auto &eachPartNodes: resultPacks) {
            MicroPart newPart;
            newPart.buildSpecialPart(vtxOpName, eachPartNodes);

            partLevels[levelId].push_back(newPart);
          }

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

  return mlir::success();
};


mlir::LogicalResult MicroPartitioner::callExternalPartitioner() {
  if (!std::filesystem::exists(microPartitionerPythonPath)) {
    llvm_unreachable("Micro partitioner script doesn't exists! This should not happen");
  }
  if (!std::filesystem::exists(inputGraphFile)) {
    llvm_unreachable("Micro partitioner input file doesn't exists! This should not happen");
  }
  if (!std::filesystem::exists(graphVectorInfoFile)) {
    llvm_unreachable("Graph vector info file doesn't exists! This should not happen");
  }

  llvm::StringRef args[] = {
    pythonName,
    microPartitionerPythonPath,
    "--graph",
    inputGraphFile,
    "--vector",
    graphVectorInfoFile,
    "--output",
    outputFile,
    "--vecmap",
    outputVectorMapFile
  };

  std::optional<llvm::StringRef> redirects[] = {
    std::nullopt,
    consoleLogFilePrefix,
    consoleLogFilePrefix
  };


  auto pythonExe = llvm::sys::findProgramByName(pythonName);
  if (!pythonExe) {
    llvm::errs() << "Cannot find " << pythonName << ". Please ensure it's in your PATH!\n";
    return failure();
  }

  int result = llvm::sys::ExecuteAndWait(*pythonExe, args, std::nullopt, redirects);

  if (result != 0) {
    llvm::errs() << "MicroPart partitioner returns non-zero code: " << result << "\n";
    llvm::errs() << pythonExe.get() << " ";
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
  partLevels.clear();
  excludeNodeLevels.clear();

  std::vector<std::string> split_line;
  mlir::SmallVector<mlir::SmallVector<uint32_t>> newPartNodesLevel;

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
        thisLevelExcludeNodes.push_back(std::stoi(split_line[i]));
      }

      assert(levelId + 1 == static_cast<int>(excludeNodeLevels.size()));
      excludeNodeLevels[levelId].swap(thisLevelExcludeNodes);
    } else if (split_line[0] == "n") {
      // Normal partition
      assert(levelId + 1 == static_cast<int>(partLevels.size()));
      partLevels.back().emplace_back();

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
        }
      }

      auto &newPart = partLevels.back().back();
      newPart.buildRegularLUTPart(newPartNodesLevel);
    } else {
      llvm::errs() << "Cannot parse line\n" << line;
      return failure();
    }
    lineno ++;
  }

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

void MicroPartitioner::collectPartIOValues(const PartitioningGraph &g) {
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
  for (auto &eachPartLevel: partLevels) {
    for (auto &eachPart: eachPartLevel) {
      auto partGood = eachPart.checkAndCollectIOValues(g, newNodeIdToDepNodeId, newNodeIdToOriginalVecDeclId);

      if (!partGood) {
        llvm::errs() << "Error when checking micro parts\n";
        llvm_unreachable("Not suppose to happen");
      }
    }
  }
}

