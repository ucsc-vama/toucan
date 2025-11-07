#include "circt/Dialect/HW/HWTypes.h"
#include "circt/Support/LLVM.h"
#include "circt/Dialect/Comb/CombDialect.h"
#include "circt/Dialect/Comb/CombOps.h"
#include "circt/Dialect/Seq/SeqOps.h"


#include "mlir/IR/Operation.h"
#include "mlir/IR/Builders.h"

#include "mlir/IR/Threading.h"
#include "mlir/IR/Value.h"
#include "mlir/Pass/AnalysisManager.h"
#include "mlir/Support/LLVM.h"
#include "mlir/Support/LogicalResult.h"

#include "llvm/ADT/BitVector.h"
#include "llvm/ADT/STLExtras.h"
#include "llvm/ADT/SmallVector.h"
#include "llvm/Support/Casting.h"
#include "llvm/Support/Debug.h"
#include "llvm/Support/ErrorHandling.h"
#include "llvm/Support/raw_ostream.h"

#include "toucan/MultiRegionMicroPartScheduler.h"
#include "toucan/CGToucanOpName.h"
#include "toucan/MicroPartLocalValueAllocator.h"
#include "toucan/MicroPartitioner.h"
#include "toucan/PartitioningManager.h"
#include "toucan/ToucanAnalysis.h"
#include "toucan/ToucanAttributes.h"
#include "toucan/ToucanCodeGenInfo.h"
#include "toucan/ToucanOps.h"
#include "toucan/ToucanTypes.h"
#include "toucan/ToucanUtils.h"
#include "toucan/PartitioningGraph.h"

#include "toucan/ToucanConfigs.h"

#include <cassert>
#include <cstddef>
#include <cstdint>

#include <boost/graph/topological_sort.hpp>
#include <cstring>
#include <iterator>
#include <array>
#include <mutex>
#include <optional>
#include <tuple>
#include <unordered_map>
#include <utility>
#include <vector>
#include <algorithm>

using namespace toucan;

using namespace mlir;
using namespace llvm;
using namespace circt;

// #define DEBUG_PRINT_CONST_VEC_DEDUP_COUNT

void MultiRegionMicroPartScheduler::sortRegistersForLocality(mlir::SmallVector<mlir::SmallVector<mlir::TypedValue<toucan::RegType>>> &regOrdered) {
  auto numRegions = regionPartData.size();
  assert(numRegions > 0);

  // Assertion: only region 0 can read registers, only last region can write to registers
  for (size_t i = 0; i < numRegions - 1; i++) {
    for (const auto &eachPart: regionPartData[i]) {
      assert(eachPart.allRegWrites.empty());
    }
  }
  for (size_t i = 1; i < numRegions; i++) {
    for (const auto &eachPart: regionPartData[i]) {
      assert(eachPart.allRegReads.empty());
    }
  }

  auto &firstRegion = regionPartData[0];
  auto &lastRegion = regionPartData.back();
  // Note: there should be no duplicate regwrite. 


  // Here we assume replication rate is relatively small
  mlir::DenseMap<mlir::TypedValue<toucan::RegType>, mlir::SmallVector<uint32_t>> regValToReaderPartIds;
  mlir::DenseMap<mlir::TypedValue<toucan::RegType>, uint32_t> regValToWriterPartId;
  mlir::DenseSet<mlir::TypedValue<toucan::RegType>> regValRead, regValWrite;

  // Collect reg read info
  for (size_t partId = 0; partId < firstRegion.size(); partId++) {
    // only have reg read
    for (auto &regVal: firstRegion[partId].readRegs) {
      regValRead.insert(regVal);
      if (!regValToReaderPartIds.contains(regVal)) {
        regValToReaderPartIds[regVal] = {};
      }
      regValToReaderPartIds[regVal].push_back(partId);
    }
  }
  
  // Collect reg write info
  for (size_t partId = 0; partId < lastRegion.size(); partId++) {
    // only have reg read
    for (auto &regVal: lastRegion[partId].writeRegs) {
      regValWrite.insert(regVal);
      assert(!regValToWriterPartId.contains(regVal) && "Each register can have only 1 writer");
      regValToWriterPartId[regVal] = partId;
    }
  }


#ifdef DEBUG_PRINT_REG_LAYOUT
  llvm::dbgs() << "In total, there are " << regValsWithMultipleReads.size() << " shared reads\n";
#endif

  // reg vals that read by multiple partitions (multiple reader)
  mlir::SmallVector<mlir::SmallVector<mlir::TypedValue<toucan::RegType>>> groupedSharedReadVals;
  // reg vals that has only 1 reader
  mlir::SmallVector<mlir::SmallVector<mlir::TypedValue<toucan::RegType>>> groupedReadOnceVals;
  // reg vals with no reader
  mlir::SmallVector<mlir::SmallVector<mlir::TypedValue<toucan::RegType>>> groupedWriteOnlyVals;
  // reg vals with no writer
  mlir::SmallVector<mlir::TypedValue<toucan::RegType>> sortedReadOnlyVals;

  auto numReadParts = firstRegion.size();
  auto numWriteParts = lastRegion.size();

  groupedSharedReadVals.resize(numWriteParts);
  groupedReadOnceVals.resize(numWriteParts);
  groupedWriteOnlyVals.resize(numWriteParts);


  // Segment by writer, then reader
  // Sort by reader: shared read, read by p0, p1, p2, ...

  // First, group by writer
  for (size_t writerPartId = 0; writerPartId < numWriteParts; writerPartId++) {
    // segment by reader
    mlir::SmallVector<mlir::SmallVector<mlir::TypedValue<toucan::RegType>>> currentSectionReadOnceValSegments;
    currentSectionReadOnceValSegments.resize(numReadParts);


    for (auto regVal: lastRegion[writerPartId].writeRegs) {
      // for each writer section
      if (regValToReaderPartIds.contains(regVal)) {
        const auto &readers = regValToReaderPartIds[regVal];
        assert(!readers.empty());
        if (readers.size() > 1) {
          // has multiple reader
          groupedSharedReadVals[writerPartId].push_back(regVal);
        } else {
          // has 1 reader
          auto readerId = readers[0];
          currentSectionReadOnceValSegments[readerId].push_back(regVal);
        }
      } else {
        // no reader, write only
        groupedWriteOnlyVals[writerPartId].push_back(regVal);
      }
    }

    // merge all regs with 1 reader
    for (auto & eachReaderSegment: currentSectionReadOnceValSegments) {
      std::copy(eachReaderSegment.begin(), eachReaderSegment.end(), 
        std::back_inserter(groupedReadOnceVals[writerPartId]));
    }

    // sort sharedVals by number of readers
    std::sort(
      groupedSharedReadVals[writerPartId].begin(), 
      groupedSharedReadVals[writerPartId].end(), 
      [&regValToReaderPartIds](const auto &a, const auto &b) {
        return regValToReaderPartIds.at(a).size() > regValToReaderPartIds.at(b).size();
    });

  }


  // Special handling for read-only vals
  mlir::SmallVector<mlir::TypedValue<toucan::RegType>> readOnlySharedVals, readOnlyOnceVals;

  for (auto &eachReadVal: regValRead) {
    if (!regValToWriterPartId.contains(eachReadVal)) {
      // read only, no writer
      assert(regValToReaderPartIds.contains(eachReadVal));
      auto readers = regValToReaderPartIds[eachReadVal];
      assert(!readers.empty());

      if (readers.size() > 1) {
        readOnlySharedVals.push_back(eachReadVal);
      } else {
        readOnlyOnceVals.push_back(eachReadVal);
      }
    }
  }
  
  std::sort(readOnlyOnceVals.begin(), readOnlyOnceVals.end(), 
  [&] (const mlir::TypedValue<toucan::RegType>& a, const mlir::TypedValue<toucan::RegType>& b) {
    auto readerPartId_a = regValToReaderPartIds[a][0];
    auto readerPartId_b = regValToReaderPartIds[b][0];
    return readerPartId_a < readerPartId_b;
  });

  std::sort(
    readOnlySharedVals.begin(), 
    readOnlySharedVals.end(), 
    [&regValToReaderPartIds](const auto &a, const auto &b) {
      return regValToReaderPartIds.at(a).size() > regValToReaderPartIds.at(b).size();
  });

  // Merge all together
#ifdef DEBUG_PRINT_REG_LAYOUT
    llvm::dbgs() << readOnlySharedVals.size() << " shared read only vals, and " << readOnlyOnceVals.size() << " read only once vals\n";
#endif
  std::copy(readOnlySharedVals.begin(), readOnlySharedVals.end(), std::back_inserter(sortedReadOnlyVals));
  std::copy(readOnlyOnceVals.begin(), readOnlyOnceVals.end(), std::back_inserter(sortedReadOnlyVals));



  // schedule
  // put read only vals to a separate section (to satisfy alignment requirement)
  if (!sortedReadOnlyVals.empty()) {
    regOrdered.reserve(numWriteParts + 1);
#ifdef DEBUG_PRINT_REG_LAYOUT
    llvm::dbgs() << "  Standalone section for " << sortedReadOnlyVals.size() << " read only regs\n";
#endif
    regOrdered.emplace_back();
    std::copy(sortedReadOnlyVals.begin(), sortedReadOnlyVals.end(), std::back_inserter(regOrdered.back()));
  } else {
    regOrdered.reserve(numWriteParts);
  }

  for (size_t partId = 0; partId < numWriteParts; partId++) {
#ifdef DEBUG_PRINT_REG_LAYOUT
    llvm::dbgs() << "Schedule for writer part " << partId << "\n";
#endif
    regOrdered.emplace_back();

#ifdef DEBUG_PRINT_REG_LAYOUT
    llvm::dbgs() << "  Append " << groupedSharedReadVals[partId].size() << " share regs\n";
    llvm::dbgs() << "  Append " << groupedReadOnceVals[partId].size() << " read once regs\n";
    llvm::dbgs() << "  Append " << groupedWriteOnlyVals[partId].size() << " write only regs\n";
#endif

    std::copy(groupedSharedReadVals[partId].begin(), groupedSharedReadVals[partId].end(), std::back_inserter(regOrdered.back()));
    std::copy(groupedReadOnceVals[partId].begin(), groupedReadOnceVals[partId].end(), std::back_inserter(regOrdered.back()));
    std::copy(groupedWriteOnlyVals[partId].begin(), groupedWriteOnlyVals[partId].end(), std::back_inserter(regOrdered.back()));
  }

  return;
}


void MultiRegionMicroPartScheduler::sortRegistersForLocality_2(mlir::SmallVector<mlir::SmallVector<mlir::TypedValue<toucan::RegType>>> &regOrdered) {
  auto numRegions = regionPartData.size();
  assert(numRegions > 0);

  // Assertion: only region 0 can read registers, only last region can write to registers
  for (size_t i = 0; i < numRegions - 1; i++) {
    for (const auto &eachPart: regionPartData[i]) {
      assert(eachPart.allRegWrites.empty());
    }
  }
  for (size_t i = 1; i < numRegions; i++) {
    for (const auto &eachPart: regionPartData[i]) {
      assert(eachPart.allRegReads.empty());
    }
  }

  auto &firstRegion = regionPartData[0];
  auto &lastRegion = regionPartData.back();
  // Note: there should be no duplicate regwrite. 


  // Here we assume replication rate is relatively small
  mlir::DenseMap<mlir::TypedValue<toucan::RegType>, mlir::SmallVector<uint32_t>> regValToReaderPartIds;
  mlir::DenseMap<mlir::TypedValue<toucan::RegType>, uint32_t> regValToWriterPartId;
  mlir::DenseSet<mlir::TypedValue<toucan::RegType>> regValRead, regValWrite;

  // Collect reg read info
  for (size_t partId = 0; partId < firstRegion.size(); partId++) {
    // only have reg read
    for (auto &regVal: firstRegion[partId].readRegs) {
      regValRead.insert(regVal);
      if (!regValToReaderPartIds.contains(regVal)) {
        regValToReaderPartIds[regVal] = {};
      }
      regValToReaderPartIds[regVal].push_back(partId);
    }
  }
  
  // Collect reg write info
  for (size_t partId = 0; partId < lastRegion.size(); partId++) {
    // only have reg read
    for (auto &regVal: lastRegion[partId].writeRegs) {
      regValWrite.insert(regVal);
      assert(!regValToWriterPartId.contains(regVal) && "Each register can have only 1 writer");
      regValToWriterPartId[regVal] = partId;
    }
  }


  // reg vals with no writer
  mlir::SmallVector<mlir::TypedValue<toucan::RegType>> sortedReadOnlyVals;

  auto numReadParts = firstRegion.size();
  auto numWriteParts = lastRegion.size();


  // Segment by writer, then reader
  // Sort by reader: shared read, read by p0, p1, p2, ...

  // First, group by writer
  regOrdered.clear();
  for (size_t writerPartId = 0; writerPartId < numWriteParts; writerPartId++) {
    // segment by reader
    regOrdered.emplace_back(lastRegion[writerPartId].writeRegs);

    mlir::DenseSet<mlir::Value> valsReadByCurrentPart;
    for (size_t readerPartId = 0; readerPartId < numReadParts; readerPartId++) {
      valsReadByCurrentPart.clear();

      for (auto &eachVal: firstRegion[readerPartId].readRegs) valsReadByCurrentPart.insert(eachVal);

      std::stable_sort(regOrdered.back().begin(), regOrdered.back().end(), [&](const auto &a, const auto &b) {
        auto a_read_by_current_part = valsReadByCurrentPart.contains(a);
        auto b_read_by_current_part = valsReadByCurrentPart.contains(b);
        return a_read_by_current_part < b_read_by_current_part;
      });
    }
  }


  // Special handling for read-only vals
  mlir::SmallVector<mlir::TypedValue<toucan::RegType>> readOnlySharedVals, readOnlyOnceVals;

  for (auto &eachReadVal: regValRead) {
    if (!regValToWriterPartId.contains(eachReadVal)) {
      // read only, no writer
      assert(regValToReaderPartIds.contains(eachReadVal));
      auto readers = regValToReaderPartIds[eachReadVal];
      assert(!readers.empty());

      if (readers.size() > 1) {
        readOnlySharedVals.push_back(eachReadVal);
      } else {
        readOnlyOnceVals.push_back(eachReadVal);
      }
    }
  }
  
  std::sort(readOnlyOnceVals.begin(), readOnlyOnceVals.end(), 
  [&] (const mlir::TypedValue<toucan::RegType>& a, const mlir::TypedValue<toucan::RegType>& b) {
    auto readerPartId_a = regValToReaderPartIds[a][0];
    auto readerPartId_b = regValToReaderPartIds[b][0];
    return readerPartId_a < readerPartId_b;
  });

  std::sort(
    readOnlySharedVals.begin(), 
    readOnlySharedVals.end(), 
    [&regValToReaderPartIds](const auto &a, const auto &b) {
      return regValToReaderPartIds.at(a).size() > regValToReaderPartIds.at(b).size();
  });

  // Merge all together
#ifdef DEBUG_PRINT_REG_LAYOUT
    llvm::dbgs() << readOnlySharedVals.size() << " shared read only vals, and " << readOnlyOnceVals.size() << " read only once vals\n";
#endif
  std::copy(readOnlySharedVals.begin(), readOnlySharedVals.end(), std::back_inserter(sortedReadOnlyVals));
  std::copy(readOnlyOnceVals.begin(), readOnlyOnceVals.end(), std::back_inserter(sortedReadOnlyVals));



  // schedule
  // put read only vals to a separate section (to satisfy alignment requirement)
  if (!sortedReadOnlyVals.empty()) {
#ifdef DEBUG_PRINT_REG_LAYOUT
    llvm::dbgs() << "  Standalone section for " << sortedReadOnlyVals.size() << " read only regs\n";
#endif
    regOrdered.emplace_back();
    std::copy(sortedReadOnlyVals.begin(), sortedReadOnlyVals.end(), std::back_inserter(regOrdered.back()));
  }

  return;
}



// Update allRegWrites in each MicroPartitioner
void MultiRegionMicroPartScheduler::sortRegWriteOps(RepCutPartitionCodeGenData &partMeta) const {
  std::sort(
    partMeta.allRegWrites.begin(),
    partMeta.allRegWrites.end(),
    [&](RegWriteOp &a, RegWriteOp &b) {
      auto reg_a = a.getReg();
      auto reg_b = b.getReg();
      auto order_a = codeGenInfo.toucanRegToId.at(reg_a);
      auto order_b = codeGenInfo.toucanRegToId.at(reg_b);
      return order_a < order_b;
    }
  );

  mlir::DenseSet<mlir::TypedValue<toucan::RegType>> writeRegSetBefore, writeRegSetAfter;
  writeRegSetBefore.insert(partMeta.writeRegs.begin(), partMeta.writeRegs.end());

  partMeta.writeRegs.clear();
  for (auto op: partMeta.allRegWrites) {
    partMeta.writeRegs.push_back(op.getReg());
  }

  writeRegSetAfter.insert(partMeta.writeRegs.begin(), partMeta.writeRegs.end());
  assert(writeRegSetBefore == writeRegSetAfter);
}

void MultiRegionMicroPartScheduler::sortRegReadOps(RepCutPartitionCodeGenData &partMeta) const {
  std::sort(
    partMeta.allRegReads.begin(),
    partMeta.allRegReads.end(),
    [&](RegReadOp &a, RegReadOp &b) {
      auto reg_a = a.getReg();
      auto reg_b = b.getReg();
      auto order_a = codeGenInfo.toucanRegToId.at(reg_a);
      auto order_b = codeGenInfo.toucanRegToId.at(reg_b);
      return order_a < order_b;
    }
  );

  mlir::DenseSet<mlir::TypedValue<toucan::RegType>> readRegSetBefore, readRegSetAfter;
  readRegSetBefore.insert(partMeta.readRegs.begin(), partMeta.readRegs.end());

  partMeta.readRegs.clear();
  for (auto op: partMeta.allRegReads) {
    partMeta.readRegs.push_back(op.getReg());
  }

  readRegSetAfter.insert(partMeta.readRegs.begin(), partMeta.readRegs.end());
  assert(readRegSetBefore == readRegSetAfter);
}


void MultiRegionMicroPartScheduler::fillRegPool(mlir::SmallVector<mlir::SmallVector<mlir::TypedValue<toucan::RegType>>> regPoolOrdered) {
  // 2. Allocate storage for all registers
  size_t sortedRegPoolSize = 0;
  for (auto &eachSection: regPoolOrdered) {
    for (auto &regVal: eachSection) {
      // allocate storate for every register
      auto regDefiningOp = regVal.getDefiningOp();

      CGRegMetaInfo regMeta;

      regMeta.namehint = getSVNameHintAttr(regDefiningOp);
      auto fragmentIdAttr = getSignalFragmentIDAttr(regDefiningOp);
      if (fragmentIdAttr) {
        regMeta.fragment_id = fragmentIdAttr->getInt();
      } else {
        regMeta.fragment_id = UINT32_MAX;
      }
      regMeta.bitWidth = regVal.getType().getElementWidth();
      regMeta.isPadding = false;
      regMeta.isIO = hasIOSignalMarker(regDefiningOp);
      regMeta.val = regVal;

      auto regId = codeGenInfo.regPool.size();
      assert(!codeGenInfo.toucanRegToId.contains(regVal));
      codeGenInfo.regPool.push_back(regMeta);
      codeGenInfo.toucanRegToId[regVal] = regId;
    }
    // add padding regs
    sortedRegPoolSize += eachSection.size();
    auto extraSpace = getExtraAlignmentSpace(sortedRegPoolSize, partitionAlignment);
    sortedRegPoolSize += extraSpace;
    for (size_t i = 0; i < extraSpace; i++) {
      CGRegMetaInfo paddingRegMeta;

      paddingRegMeta.isPadding = true;
      paddingRegMeta.bitWidth = 0;
      paddingRegMeta.fragment_id = UINT32_MAX;
      paddingRegMeta.isIO = false;

      // auto regId = codeGenInfo.regPool.size();
      codeGenInfo.regPool.push_back(paddingRegMeta);
    }
  }
}


void MultiRegionMicroPartScheduler::fillMemPool(const PartitioningGraph &graph) {
  // Here we assume each memory has at least 1 writer
  uint64_t memBaseAddr = 0;

  auto allocateNewMem = [&](mlir::TypedValue<toucan::MemType> memVal, bool hasMultipleWriter) {
      auto memDefiningOp = memVal.getDefiningOp();

      CGMemMetaInfo memMeta;

      memMeta.namehint = getSVNameHintAttr(memDefiningOp);
      auto fragmentIdAttr = getSignalFragmentIDAttr(memDefiningOp);
      if (fragmentIdAttr) {
        memMeta.fragment_id = fragmentIdAttr->getInt();
      } else {
        assert(false && "Every memory should have a fragment id!");
        memMeta.fragment_id = UINT32_MAX;
      }
      memMeta.bitWidth = memVal.getType().getElementWidth();
      memMeta.memDepth = memVal.getType().getDepth();
      memMeta.hasMultipleWriter = hasMultipleWriter;

      // if a memory has multiple writer, add extra padding to avoid possible write conflict
      assert(memMeta.bitWidth <= 4);
      uint64_t memCapacity = (memMeta.hasMultipleWriter) ? memMeta.memDepth * multiWriterMemElemBytes : memMeta.memDepth;
      memMeta.memBase = memBaseAddr;
      memBaseAddr += (memCapacity + memPaddingSpace);

      auto memId = codeGenInfo.memPool.size();
      codeGenInfo.memPool.push_back(memMeta);
      codeGenInfo.toucanMemToId[memVal] = memId;
  };

  for (auto vtxId : boost::make_iterator_range(vertices(graph))) {
    // for each mem write
    auto vtxOpName = graph[vtxId].toucanOpName;
    if (vtxOpName == CGToucanOPName::MemWrite) {
      // Note: For now, mems with multiple write ports are still merged, so at this time, each memory will only have 1 writer.
      auto memWriteOp = cast<toucan::MemWriteOp>(graph[vtxId].op);
      auto memVal = memWriteOp.getMem();
      auto hasMultipleWriter = (graph[vtxId].opCount > 1);
      allocateNewMem(memVal, hasMultipleWriter);
    }
  }

  // llvm::dbgs() << "Collect " << codeGenInfo.toucanMemToId.size() << " mems\n";

  for (auto vtxId : boost::make_iterator_range(vertices(graph))) {
    // for each mem write
    auto vtxOpName = graph[vtxId].toucanOpName;
    if (vtxOpName == CGToucanOPName::MemRead) {
      // Note: Also allocate space for read only memory, though they are never written
      auto memReadOp = cast<toucan::MemReadOp>(graph[vtxId].op);
      auto memVal = memReadOp.getMem();

      if (!codeGenInfo.toucanMemToId.contains(memVal)) {
        auto valDefiningOp = memVal.getDefiningOp();
        valDefiningOp->emitWarning("This memory has reader but no writer!");

        allocateNewMem(memVal, false);
      }
    }
  }

  codeGenInfo.totalMemSize = memBaseAddr;
}

void MultiRegionMicroPartScheduler::generateRegMemLayout(const PartitioningGraph &rawGraph) {
  // collect all reg and memory, generate layout
  codeGenInfo.regPool.clear();
  codeGenInfo.memPool.clear();


  // 1. sort registers
  // Writer part -> val. Needs padding
  mlir::SmallVector<mlir::SmallVector<mlir::TypedValue<toucan::RegType>>> regPoolOrdered;

  // llvm::dbgs() << "Sorting registers\n";
  sortRegistersForLocality_2(regPoolOrdered);

  // Now, registers and exchangeVal location and ops are sorted
  fillRegPool(regPoolOrdered);
  // fillRegOrderTable();

  // Coaleasce register access.
  for (auto &eachPartData: regionPartData[0]) {
    sortRegReadOps(eachPartData);
  }
  for (auto &eachPartData: regionPartData.back()) {
    sortRegWriteOps(eachPartData);
  }
  
  // Allocate storage for all memories
  fillMemPool(rawGraph);

  return;
}


void MultiRegionMicroPartScheduler::sortExchangeValsForLocality(mlir::SmallVector<mlir::SmallVector<mlir::Value>> &exchangeValOrdered) {
  // Note: Exchange vals should always have at least 1 reader and 1 writer

  uint32_t numTotalParts = 0;
  for (size_t regionId = 0; regionId < regionPartData.size(); regionId++) {
    numTotalParts += regionPartData[regionId].size();
  }

  {
    // Note: Check if all exgVals has exactly 1 writer.
    mlir::DenseSet<mlir::Value> valsWritten;
    for (const auto &eachRegion: regionPartData) {
      for (const auto &eachPartData: eachRegion) {
        for (const auto &eachVal: eachPartData.allExgWriteVals) {
          assert(!valsWritten.contains(eachVal) && "Each exchange val should have exactly 1 writer. Check if it's a vector value and being split to 2 partitions. If so, consider limit it");
          valsWritten.insert(eachVal);
        }
      }
    }
  }

  // get part read/write vals
  size_t flatPartId = 0;
  for (size_t regionId = 0; regionId < regionPartData.size(); regionId++) {
    auto &currentRegionData = regionPartData[regionId];
    for (auto &partData: currentRegionData) {
      assert(flatPartId < numTotalParts);

      if (partData.allExgWriteVals.empty()) continue;

      exchangeValOrdered.emplace_back();
      exchangeValOrdered.back().assign(partData.allExgWriteVals);

      auto &thisSectionVals = exchangeValOrdered.back();

      mlir::DenseSet<mlir::Value> valsReadByCurrentPart;

      for (const auto &each_region: regionPartData) {
        for (const auto &each_part: each_region) {
          if (each_part.allExgReadVals.empty()) continue;

          valsReadByCurrentPart.clear();
          valsReadByCurrentPart.insert(each_part.allExgReadVals.begin(), each_part.allExgReadVals.end());

          std::stable_sort(
            thisSectionVals.begin(),
            thisSectionVals.end(),
            [&valsReadByCurrentPart](const mlir::Value &a, const mlir::Value &b) {
              auto a_readByCurrentPart = valsReadByCurrentPart.contains(a);
              auto b_readByCurrentPart = valsReadByCurrentPart.contains(b);

              return a_readByCurrentPart < b_readByCurrentPart;
            }
          );
        }
      }
    }
  }
}

// sort exgwrite ops at last level by order of result exchangeVal in exchange pool
void MultiRegionMicroPartScheduler::sortExchangeReadOps(RepCutPartitionCodeGenData &partData) const {
  std::sort(
    partData.allExgReadVals.begin(),
    partData.allExgReadVals.end(),
    [&](const mlir::Value &a, const mlir::Value &b) {
      auto order_a = codeGenInfo.toucanExgValToId.at(a);
      auto order_b = codeGenInfo.toucanExgValToId.at(b);
      return order_a < order_b;
    }
  );
}

// sort exgwrite ops at last level by order of result exchangeVal in exchange pool
void MultiRegionMicroPartScheduler::sortExchangeWriteOps(RepCutPartitionCodeGenData &partData) const {
  std::sort(
    partData.allExgWriteVals.begin(),
    partData.allExgWriteVals.end(),
    [&](const mlir::Value &a, const mlir::Value &b) {
      auto order_a = codeGenInfo.toucanExgValToId.at(a);
      auto order_b = codeGenInfo.toucanExgValToId.at(b);
      return order_a < order_b;
    }
  );
}

void MultiRegionMicroPartScheduler::fillExchangePool(mlir::SmallVector<mlir::SmallVector<mlir::Value>> &exgValOrdered) {

  // size_t validBytesInExchangePool = 0;
  codeGenInfo.exchangeValPool.clear();
  assert(codeGenInfo.exchangePool.empty());
  assert(codeGenInfo.toucanExgValToId.empty());

  for (auto &eachSection: exgValOrdered) {
    for (auto &exgVal: eachSection) {
      // allocate storate for every exchange value
      size_t byteCount = 1;
      if (auto vecVal = dyn_cast<mlir::TypedValue<toucan::VecType>>(exgVal)) {
        byteCount = vecVal.getType().getLength();
      }
      // validBytesInExchangePool += byteCount;

      auto valId = codeGenInfo.exchangePool.size();

      codeGenInfo.exchangeValPool.push_back(exgVal);
      assert(!codeGenInfo.toucanExgValToId.contains(exgVal));
      codeGenInfo.toucanExgValToId[exgVal] = valId;

      for (size_t i = 0; i < byteCount; i++) {
        CGExchangeValueMetaInfo exgMeta;
        
        exgMeta.isPadding = false;
        exgMeta.val = exgVal;
        exgMeta.byteCountOfVal = byteCount;
        exgMeta.byteIdx = i;

        codeGenInfo.exchangePool.push_back(exgMeta);
      }
    }
    // add padding regs
    auto extraSpace = getExtraAlignmentSpace(codeGenInfo.exchangePool.size(), partitionAlignment);

    for (size_t i = 0; i < extraSpace; i++) {
      CGExchangeValueMetaInfo paddingExgMeta;

      paddingExgMeta.isPadding = true;
      paddingExgMeta.byteCountOfVal = 0;
      paddingExgMeta.byteIdx = 0;

      codeGenInfo.exchangePool.push_back(paddingExgMeta);
    }
  }
}

void MultiRegionMicroPartScheduler::generateExchangeLayout(const mlir::SmallVector<mlir::Value> &exchangeValPool) {
  auto numRegions = regionPartData.size();
  assert(numRegions == 2);

  // region 0 has writer, region 1 has reader
  // copy to internal pool
  assert(codeGenInfo.exchangeValPool.empty());
  codeGenInfo.exchangeValPool.assign(exchangeValPool);
  
  // 4. Reorder exchangePool
  mlir::SmallVector<mlir::SmallVector<mlir::Value>> exchangeValIdOrdered;

  // ExchangeVals are already populated.
  // First reorder them, then sort
  sortExchangeValsForLocality(exchangeValIdOrdered);

  fillExchangePool(exchangeValIdOrdered);



  for (auto &eachPartData: regionPartData[0]) {
    sortExchangeWriteOps(eachPartData);
  }
  for (auto &eachPartData: regionPartData.back()) {
    sortExchangeReadOps(eachPartData);
  }
}



// Collect const decls. DOES NOT collect const vec decls
// Const vars are shared
void MultiRegionMicroPartScheduler::collectConstantVars(const PartitioningGraph &graph) {
  // Collect all consts, populate value pool
  // ConstDecl only exists in first level
  for (auto vtxId : boost::make_iterator_range(vertices(graph))) {
    auto vtxOpName = graph[vtxId].toucanOpName;
    if (vtxOpName == CGToucanOPName::ConstDecl) {
      auto op = graph[vtxId].op;

      if (auto constOp = dyn_cast<toucan::ConstantOp>(op)) {
        // regular const
        auto constVal = constOp.getValue();
        auto bitWidth = constVal.getBitWidth();
        assert(bitWidth <= 4);
        auto rawVal = static_cast<uint8_t>(constVal.getZExtValue());
        // save op result value
        assert(rawVal == (rawVal & ((1 << bitWidth) - 1)));

        constValToRawValue[constOp.getResult()] = rawVal;
      } else {
        // Ignore const vec decls for now.
      }
    }
  }
}

// Collect const vec decls for each partition
void MultiRegionMicroPartScheduler::collectConstantVecs(const RepCutPartitionCodeGenData &partData, CGPartitionMetaInfo &partInfo) {
  // Collect all const vecs, populate constVecPool
  mlir::SmallVector<toucan::DefConstVectorOp> constVecDeclOps;

  for (auto &mPartLevel: partData.mpartLevels) {
    for (auto &mPart: mPartLevel) {
      if (mPart->isRegularPart()) continue;

      // special part
      auto vtxOpName = mPart->opType;
      
      if (vtxOpName == CGToucanOPName::VecRead) {
        for (auto &rawOp: mPart->specialOps) {
          auto op = cast<toucan::VectorReadOp>(rawOp);
          auto vecHandle = op.getHandle();
          auto vecDeclOp = vecHandle.getDefiningOp();

          if (auto defConstVecOp = dyn_cast<toucan::DefConstVectorOp>(vecDeclOp)) {
            constVecDeclOps.push_back(defConstVecOp);
          }
        }
      } else if (vtxOpName == CGToucanOPName::VecArith) {
        for (auto &rawOp: mPart->specialOps) {
          auto op = cast<toucan::VectorArithOp>(rawOp);
          auto vec1 = op.getV1();
          auto vec2 = op.getV2();

          if (auto vec1ConstOp = dyn_cast<toucan::DefConstVectorOp>(vec1.getDefiningOp())) {
            constVecDeclOps.push_back(vec1ConstOp);
          }
          if (auto vec2ConstOp = dyn_cast<toucan::DefConstVectorOp>(vec2.getDefiningOp())) {
            constVecDeclOps.push_back(vec2ConstOp);
          }
        }
      } else if (vtxOpName == CGToucanOPName::VecLogic) {
        for (auto &rawOp: mPart->specialOps) {
          auto op = cast<toucan::VectorLogicOp>(rawOp);
          auto vec1 = op.getV1();
          auto vec2 = op.getV2();

          if (auto vec1ConstOp = dyn_cast<toucan::DefConstVectorOp>(vec1.getDefiningOp())) {
            constVecDeclOps.push_back(vec1ConstOp);
          }
          if (auto vec2ConstOp = dyn_cast<toucan::DefConstVectorOp>(vec2.getDefiningOp())) {
            constVecDeclOps.push_back(vec2ConstOp);
          }
        }
      }
    }
  }


  // Dedup vector
  std::map<std::vector<uint8_t>, size_t> smallVecDedupTable;
#ifdef DEBUG_PRINT_CONST_VEC_DEDUP_COUNT
  size_t constVecDedupCount = 0;
#endif

  for (auto defConstVecOp: constVecDeclOps) {
    // a const vector used in this graph/region. 
    // save op result value
    // Vec result map to first vec element
    auto vecHandle = defConstVecOp.getHandle();
    auto bitWidth = vecHandle.getType().getElementWidth();
    auto vecLength = vecHandle.getType().getLength();
    assert(bitWidth <= 4);

    bool canDedup = false;
    size_t dedupValId;
    std::vector<uint8_t> vecRawVal;
    // find if we have same vector placed
    if (bitWidth == 4) {
      for (auto &vecValElem: llvm::reverse(defConstVecOp.getValues().getValue())) {
        auto elemVal = cast<mlir::IntegerAttr>(vecValElem).getValue();
        auto elemValWidth = elemVal.getBitWidth();
        assert(elemValWidth <= 4);
  
        auto elemValMask = static_cast<uint8_t>((1 << elemValWidth) - 1);
        uint8_t rawVal = elemValMask & static_cast<uint8_t>(elemVal.getZExtValue());
        vecRawVal.push_back(rawVal);
      }

      assert(vecRawVal.size() == vecLength);
      if (smallVecDedupTable.contains(vecRawVal)) {
        canDedup = true;
        dedupValId = smallVecDedupTable.at(vecRawVal);
#ifdef DEBUG_PRINT_CONST_VEC_DEDUP_COUNT
        constVecDedupCount += 1;
#endif
      } else {
        // cannot dedup. 
      }
    }

    if (canDedup) {
      partInfo.valueToValId[vecHandle] = dedupValId;
    } else {
      // unable to dedup
      auto valId = partInfo.constVecPool.size();
      partInfo.valueToValId[vecHandle] = valId;

      if (bitWidth == 4) {
        assert(!smallVecDedupTable.contains(vecRawVal));
        smallVecDedupTable[vecRawVal] = valId;
      }

      // Why reverse? vector decl op elements are MSB first. Reorder to LSB first to make vecRead's life easier
      for (auto &vecValElem: llvm::reverse(defConstVecOp.getValues().getValue())) {
        auto elemVal = cast<mlir::IntegerAttr>(vecValElem).getValue();
        auto elemValWidth = elemVal.getBitWidth();
        assert(elemValWidth <= 4);

        auto elemValMask = static_cast<uint8_t>((1 << elemValWidth) - 1);
        uint8_t rawVal = elemValMask & static_cast<uint8_t>(elemVal.getZExtValue());
        partInfo.constVecPool.push_back(rawVal);
      }
    }
  }

#ifdef DEBUG_PRINT_CONST_VEC_DEDUP_COUNT
  llvm::dbgs() << constVecDedupCount << " const vecs deduplicated\n";
#endif
}


void MultiRegionMicroPartScheduler::scheduleRegReads(CGPartitionMetaInfo &partInfo, mlir::SmallVector<toucan::RegReadOp> &allRegReads) {
  mlir::SmallVector<CGOpMetaInfo> currentLevelOps;
  currentLevelOps.reserve(allRegReads.size());

  for (auto &regReadOp: allRegReads) {
    auto regVal = regReadOp.getReg();
    assert(codeGenInfo.toucanRegToId.contains(regVal) && "A register that never seen was read!");
    auto regValId = codeGenInfo.toucanRegToId[regVal];

    assert(partInfo.valueToValId.contains(regReadOp.getResult()));
    CGOpMetaInfo opMeta;
    opMeta.opName = CGToucanOPName::RegRead;
    opMeta.op = regReadOp;
    opMeta.regRead.reg = regValId;
    opMeta.regRead.result = partInfo.valueToValId[regReadOp.getResult()];
    populateOpMetaDebugInfo(opMeta, regReadOp);
    currentLevelOps.push_back(opMeta);
  }

  // Save ops
  std::swap(partInfo.regReadOps, currentLevelOps);
}

static bool isArrayElementIncrementalAndContinuous(const mlir::SmallVector<uint32_t> in) {
  assert(in.size() > 0);

  auto startElem = in.front();
  for (size_t i = 1; i < in.size(); i++) {
    auto elem = in[i];
    if (elem != startElem + i) return false;
  }
  return true;
}

void MultiRegionMicroPartScheduler::scheduleRegWrites(CGPartitionMetaInfo &partInfo, mlir::SmallVector<toucan::RegWriteOp> &allRegWrites) {

  // Code gen for last level

  mlir::SmallVector<CGOpMetaInfo> regWriteOps;

  for (auto &regWriteOp: allRegWrites) {
    auto regVal = regWriteOp.getReg();
    auto dataVal = regWriteOp.getData();

    assert(codeGenInfo.toucanRegToId.contains(regVal) && "A register never seen was written!");
    auto regValId = codeGenInfo.toucanRegToId[regVal];
    assert(partInfo.valueToValId.contains(dataVal) && "A value never seen was read!");
    auto dataValId = partInfo.valueToValId[dataVal];

    CGOpMetaInfo opMeta;
    opMeta.op = regWriteOp;
    opMeta.opName = CGToucanOPName::RegWrite;
    opMeta.regWrite.reg = regValId;
    opMeta.regWrite.dat = dataValId;
    populateOpMetaDebugInfo(opMeta, regWriteOp);

    // Namehint not needed for last level ops
    regWriteOps.push_back(opMeta);
  }

  if (regWriteOps.empty()) {
    // In rare case this part writes nothing
    partInfo.regWriteOps.clear();
    return;
  }

  // Check if regWrites in smem location are continuous 
  mlir::SmallVector<uint32_t> memLocations;
  memLocations.reserve(regWriteOps.size());
  for (const auto &eachOp: regWriteOps) {
    memLocations.push_back(eachOp.regWrite.reg);
  }
  // regs in smem should be aligned to 4B
  assert(memLocations.front() % 4 == 0);
  assert(isArrayElementIncrementalAndContinuous(memLocations));

  // Also check reg values in global mem are continuous
  memLocations.clear();
  for (const auto &eachOp: regWriteOps) {
    memLocations.push_back(eachOp.regWrite.dat);
  }
  // regs in global mem should also align to 4B
  assert(memLocations.front() % 4 == 0);
  assert(isArrayElementIncrementalAndContinuous(memLocations));

  std::swap(partInfo.regWriteOps, regWriteOps);
}

void MultiRegionMicroPartScheduler::scheduleMemWrites(CGPartitionMetaInfo &partInfo, mlir::SmallVector<toucan::MemWriteOp> &allMemWrites) {

  mlir::SmallVector<CGOpMetaInfo> memWriteOps;

  for (auto &memWriteOp: allMemWrites) {

    auto memVal = memWriteOp.getMem();
    assert(codeGenInfo.toucanMemToId.contains(memVal));
    auto memValId = codeGenInfo.toucanMemToId[memVal];


    auto dataVal = memWriteOp.getData();
    auto enVal = memWriteOp.getEn();

    assert(partInfo.valueToValId.contains(dataVal));
    auto dataValId = partInfo.valueToValId[dataVal];
    assert(partInfo.valueToValId.contains(enVal));
    auto enValId = partInfo.valueToValId[enVal];

    auto memAddrVec = memWriteOp.getAddrVec();
    auto memAddrVecId = partInfo.valueToValId[memAddrVec];
    assert(memAddrVec.getType().getLength() == 8 && "Memory address should be 32 bit vector");

    CGOpMetaInfo mwOpMeta;
    mwOpMeta.op = memWriteOp;
    mwOpMeta.opName = CGToucanOPName::MemWrite;

    mwOpMeta.memWrite.hasMultipleWriter = codeGenInfo.memPool[memValId].hasMultipleWriter;
    mwOpMeta.memWrite.memBase = codeGenInfo.memPool[memValId].memBase;
    mwOpMeta.memWrite.memDepth = codeGenInfo.memPool[memValId].memDepth;
    mwOpMeta.memWrite.addrVec = memAddrVecId;
    mwOpMeta.memWrite.dat = dataValId;
    mwOpMeta.memWrite.en = enValId;
    populateOpMetaDebugInfo(mwOpMeta, memWriteOp);

    // Namehint not needed for last level ops
    memWriteOps.push_back(mwOpMeta);
  }

  std::swap(partInfo.memWriteOps, memWriteOps);
}

void MultiRegionMicroPartScheduler::scheduleStops(CGPartitionMetaInfo &partInfo, mlir::SmallVector<toucan::StopOp> &allStops) {

  mlir::SmallVector<CGOpMetaInfo> stopOps;

  for (auto stopOp: allStops) {
    auto enVal = stopOp.getEn();

    assert(partInfo.valueToValId.contains(enVal));
    auto enValId = partInfo.valueToValId[enVal];

    CGOpMetaInfo opMeta;
    opMeta.op = stopOp;
    opMeta.opName = CGToucanOPName::Stop;
    opMeta.stop.en = enValId;

    // Namehint not needed for last level ops
    stopOps.push_back(opMeta);
  }

  std::swap(partInfo.stopOps, stopOps);
}


void MultiRegionMicroPartScheduler::schedulePrints(CGPartitionMetaInfo &partInfo, mlir::SmallVector<toucan::PrintOp> &allPrints) {

  mlir::SmallVector<CGOpMetaInfo> printOps;

  for (auto printOp: allPrints) {
    auto printStr = printOp.getMsg();
    auto enVal = printOp.getEn();

    assert(partInfo.valueToValId.contains(enVal));
    auto enValId = partInfo.valueToValId[enVal];
    
    assert(codeGenInfo.printStrings.contains(printStr));
    auto printStrId = codeGenInfo.printStrings[printStr];

    CGOpMetaInfo opMeta;
    opMeta.op = printOp;
    opMeta.opName = CGToucanOPName::Print;
    opMeta.print.en = enValId;
    opMeta.print.msg = printStrId;

    // Namehint not needed for last level ops
    printOps.push_back(opMeta);
  }

  std::swap(partInfo.printOps, printOps);
}


void MultiRegionMicroPartScheduler::scheduleExchangeReads(CGPartitionMetaInfo &partInfo, const mlir::SmallVector<mlir::Value> &allExgReadVals) {
  mlir::SmallVector<CGOpMetaInfo> currentLevelOps;

  // uint32_t opId = 0;
  for (auto exchangeVal: allExgReadVals) {
    assert(codeGenInfo.toucanExgValToId.contains(exchangeVal));
    auto exchangeValId = codeGenInfo.toucanExgValToId[exchangeVal];
    assert(codeGenInfo.exchangePool.size() > exchangeValId);
    assert(codeGenInfo.exchangePool[exchangeValId].isPadding == false);
    auto valByteCount = codeGenInfo.exchangePool[exchangeValId].byteCountOfVal;
    auto localValId = partInfo.valueToValId.at(exchangeVal);

    for (size_t i = 0; i < valByteCount; i++) {
      CGOpMetaInfo opMeta;

      opMeta.opName = CGToucanOPName::ExchangeRead;
      opMeta.exgRead.exchangeVal = exchangeValId + i;
      opMeta.exgRead.localVal = localValId + i;
      opMeta.op = nullptr;

      currentLevelOps.push_back(opMeta);
    }
  }

  // Save ops
  std::swap(partInfo.exchangeReadOps, currentLevelOps);
}

void MultiRegionMicroPartScheduler::scheduleExchangeWrites(CGPartitionMetaInfo &partInfo, const mlir::SmallVector<mlir::Value> &allExgWriteVals) {
  mlir::SmallVector<CGOpMetaInfo> currentLevelOps;

  // uint32_t opId = 0;
  for (auto exchangeVal: allExgWriteVals) {
    assert(codeGenInfo.toucanExgValToId.contains(exchangeVal));
    auto exchangeValId = codeGenInfo.toucanExgValToId[exchangeVal];
    assert(codeGenInfo.exchangePool.size() > exchangeValId);
    assert(codeGenInfo.exchangePool[exchangeValId].isPadding == false);
    auto valByteCount = codeGenInfo.exchangePool[exchangeValId].byteCountOfVal;
    auto localValId = partInfo.valueToValId.at(exchangeVal);

    for (size_t i = 0; i < valByteCount; i++) {
      CGOpMetaInfo opMeta;

      opMeta.opName = CGToucanOPName::ExchangeWrite;
      opMeta.exgWrite.localVal = localValId + i;
      opMeta.exgWrite.exchangeVal = exchangeValId + i;
      opMeta.op = nullptr;

      currentLevelOps.push_back(opMeta);
    }
  }

  // Save ops
  std::swap(partInfo.exchangeWriteOps, currentLevelOps);
}



void MultiRegionMicroPartScheduler::buildDummyVtxIndexInVec(const MicroPartitioner mPartitioner) {
  assert(dummyVtxIndexInVecTable.empty());

  for (auto &[vecId, _]: mPartitioner.outputVectorNopMap) {
    assert(mPartitioner.originalVectorElementsMap.contains(vecId));
    auto vecNumElements = mPartitioner.originalVectorElementsMap.at(vecId).size();
    assert(mPartitioner.outputVectorNopMap.at(vecId).size() == vecNumElements);
    for (size_t i = 0; i < vecNumElements; i++) {
      auto newNodeId = mPartitioner.outputVectorNopMap.at(vecId)[i];
      // auto oldNodeId = mPartitioner.originalVectorElementsMap.at(vecId)[i];

      assert(!dummyVtxIndexInVecTable.contains(newNodeId));

      // Note: In CIRCT, vector elements are sorted in descending order (Last element placed at first)
      // In real backend it should be reversed to ascending order
      uint32_t elemIndexInSMem = vecNumElements - 1 - i;
      dummyVtxIndexInVecTable[newNodeId] = elemIndexInSMem;
    }
  }
}

void MultiRegionMicroPartScheduler::copyVecTables(const MicroPartitioner mPartitioner) {
  outputVectorNopMap.clear();
  newNodeIdToDepNodeId.clear();
  newNodeIdToOriginalVecDeclId.clear();

  outputVectorNopMap = mPartitioner.outputVectorNopMap;
  newNodeIdToDepNodeId = mPartitioner.newNodeIdToDepNodeId;
  newNodeIdToOriginalVecDeclId = mPartitioner.newNodeIdToOriginalVecDeclId;
}



void MultiRegionMicroPartScheduler::scheduleRegularMicroPart(const PartitioningGraph &graph, CGMicroPartInfo &part, const MicroPart &mPart, const mlir::DenseMap<mlir::Value, uint32_t> &valToValId) const {
  assert(mPart.opType == CGToucanOPName::LUT);

  auto findDummyNopDepValue = [&](uint32_t dummyVtx) {
    auto vecDeclId = newNodeIdToOriginalVecDeclId.at(dummyVtx);
    auto vecOp = cast<toucan::DefVectorOp>(graph[vecDeclId].op);

    uint32_t i = 0;
    const auto &thisVecNewIds = outputVectorNopMap.at(vecDeclId);
    for (;i < thisVecNewIds.size(); i++) {
      if (thisVecNewIds[i] == dummyVtx) break;
    }
    assert(i < thisVecNewIds.size());

    assert(vecOp.getInputs().size() > i);
    return vecOp.getInputs()[i];
  };

  part.clear();
  part.opType = CGToucanOPName::LUT;

  
  auto valueToLifeCycle = mPart.extractValueLifeTime();

  auto WriteBackLevel = static_cast<uint32_t>(mPart.levels.size());

  mlir::DenseMap<mlir::Value, uint8_t> shuffleValueToId, shuffleValueToId_next;

  

  // Values that need to be written back to SMem at last level
  struct PendingWriteBack {
    mlir::Value val;
    uint16_t sMemLocation;
  };
  mlir::SmallVector<PendingWriteBack> pendingWriteBackValueAndLocation;
  pendingWriteBackValueAndLocation.reserve(32);

  {
    // first level
    uint8_t opIndex = 0;
    mlir::SmallVector<uint16_t> topLevelOpInputValIndecies;
    // For double check
    mlir::DenseSet<mlir::Value> allInputVals;
    mlir::SmallVector<mlir::Value> topLevelResultVals;

    topLevelOpInputValIndecies.reserve(3);
    allInputVals.reserve(64);
    topLevelResultVals.reserve(32);
    part.topLevel.reserve(32);

    auto getLUTOpInputIds = [&](mlir::Operation *op) {
      assert(isa<toucan::LUTOp>(op));
      auto lutOp = cast<toucan::LUTOp>(op);

      topLevelOpInputValIndecies.clear();

      auto lutInputs = lutOp.getInputs();
      auto numLutOprands = lutInputs.size();
      assert(numLutOprands <= 3);
      size_t pos = 0;
      for (; pos < 3 - numLutOprands; pos++) {
        // Note: the first elem in value pool is const 0
        assert(pos == topLevelOpInputValIndecies.size());
        topLevelOpInputValIndecies.push_back(0);
      }
      for (auto val: lutOp.getInputs()) {
        assert(valToValId.contains(val));
        auto valId = valToValId.at(val);
        assert(valId < UINT16_MAX);
        assert(pos == topLevelOpInputValIndecies.size());
        topLevelOpInputValIndecies.push_back(valId);
        pos++;

        if (!isa<toucan::ConstantOp>(val.getDefiningOp())) {
          allInputVals.insert(val);
        }
      }
      assert(topLevelOpInputValIndecies.size() == 3);
    };

    for (auto eachVtx: mPart.levels.front()) {
      auto vtxIsDummyNop = newNodeIdToDepNodeId.contains(eachVtx);

      if (vtxIsDummyNop) {
        // vecdecl
        auto depValue = findDummyNopDepValue(eachVtx);
        // This value need to be passed through entire micro part

        auto depValueIdInSMem = valToValId.at(depValue);
        assert(depValueIdInSMem < UINT16_MAX);

        auto vecVtx = newNodeIdToOriginalVecDeclId.at(eachVtx);
        auto vecVal = cast<toucan::DefVectorOp>(graph[vecVtx].op).getHandle();

        auto indexInVec = dummyVtxIndexInVecTable.at(eachVtx);
        
        auto vecValIdInSMem = valToValId.at(vecVal);
        auto resultIndex = vecValIdInSMem + indexInVec;
        assert(resultIndex < UINT16_MAX);

        // this value should be written to smem at last level!
        pendingWriteBackValueAndLocation.push_back({depValue, static_cast<uint16_t>(resultIndex)});

        // assert(!shuffleValueToId.contains(depValue));
        if (!shuffleValueToId_next.contains(depValue)) {
          // This depValue has not been brought to wrap
          // create a NOP
          part.topLevel.push_back({
            LUTOpName::LUT_Nop, 
            0, 0, static_cast<uint16_t>(depValueIdInSMem)});
          topLevelResultVals.push_back(depValue);
          auto resultValShuffleId = static_cast<uint8_t>(opIndex + 32);
          shuffleValueToId_next[depValue] = resultValShuffleId;

          if (!isa<toucan::ConstantOp>(depValue.getDefiningOp())) {
            allInputVals.insert(depValue);
          }

          assert(opIndex < 32);
          opIndex++;
          assert(opIndex == part.topLevel.size());
        } else {
          // This depValue already being read by some other NOP
          // do nothing
        }
      } else {
        // regular lut op
        auto rawOp = graph[eachVtx].op;
        auto lutOp = cast<toucan::LUTOp>(rawOp);

        getLUTOpInputIds(rawOp);

        auto resultVal = lutOp.getResult();

        if (mPart.outputValueSet.contains(resultVal)) {
          assert(valToValId.contains(resultVal));
          auto resultIndex = valToValId.at(resultVal);
          assert(resultIndex < UINT16_MAX);
          pendingWriteBackValueAndLocation.push_back({resultVal, static_cast<uint16_t>(resultIndex)});
        }

        assert(valueToLifeCycle.contains(resultVal));
        assert(valueToLifeCycle[resultVal].start == 0);

        part.topLevel.push_back({
          lutOp.getOpName(), 
          topLevelOpInputValIndecies[0], 
          topLevelOpInputValIndecies[1], 
          topLevelOpInputValIndecies[2]});
        topLevelResultVals.push_back(resultVal);

        assert(!shuffleValueToId_next.contains(resultVal));
        auto resultValShuffleId = static_cast<uint8_t>(opIndex + 32);
        shuffleValueToId_next[resultVal] = resultValShuffleId;

        assert(opIndex < 32);
        opIndex++;
        assert(opIndex == part.topLevel.size());
      }
    }

    // insert NOP for partition reads
    mlir::DenseSet<mlir::Value> valuesOnlyUsedByFirstLevel;
    valuesOnlyUsedByFirstLevel.reserve(64);

    assert(mPart.valuesUsedByEachLevel.size() > 0);
    valuesOnlyUsedByFirstLevel.insert(mPart.valuesUsedByEachLevel.front().begin(), mPart.valuesUsedByEachLevel.front().end());
    for (size_t i = 1; i < mPart.valuesUsedByEachLevel.size(); i++) {
      for (const auto &v: mPart.valuesUsedByEachLevel[i]) {
        if (valuesOnlyUsedByFirstLevel.contains(v)) {
          valuesOnlyUsedByFirstLevel.erase(v);
        }
      }
    }

    for (auto &eachInputVal: mPart.inputValues) {
      assert(!mPart.outputValueSet.contains(eachInputVal));
      if (valuesOnlyUsedByFirstLevel.contains(eachInputVal)) continue;
      if (!shuffleValueToId_next.contains(eachInputVal)) {
        // insert a dummy read op
        assert(valToValId.contains(eachInputVal));

        auto inputValIdInSmem = valToValId.at(eachInputVal);
        assert(inputValIdInSmem < UINT16_MAX);

        part.topLevel.push_back({
          LUTOpName::LUT_Nop,
          0, 0, static_cast<uint16_t>(inputValIdInSmem)
        });
        topLevelResultVals.push_back(eachInputVal);

        assert(!shuffleValueToId_next.contains(eachInputVal));
        auto resultValShuffleId = static_cast<uint8_t>(opIndex + 32);
        shuffleValueToId_next[eachInputVal] = resultValShuffleId;

        assert(!isa<toucan::ConstantOp>(eachInputVal.getDefiningOp()));
        allInputVals.insert(eachInputVal);

        opIndex++;
        assert(opIndex == part.topLevel.size());
      }
    }
    if (opIndex > 32) {
      dbgs() << "Too many ops in top level!\n";
      mPart.print();
      dbgs() << "Top level result val:\n";
      for (const auto &v: topLevelResultVals) {
        v.print(dbgs());
        dbgs() << "\n";
      }
    }

    part.maxActiveVars = part.topLevel.size();
    assert(opIndex <= 32);
    assert(shuffleValueToId.empty());
    assert(shuffleValueToId_next.size() == part.topLevel.size());
    assert(allInputVals == mPart.inputValues);
  }

  // remaining levels

  mlir::SmallVector<uint8_t> lutOpInputValIndecies;
  lutOpInputValIndecies.reserve(3);

  auto getLUTOpInputShuffleIds = [&lutOpInputValIndecies, &shuffleValueToId](mlir::Operation *op) {
    assert(isa<toucan::LUTOp>(op));
    auto lutOp = cast<toucan::LUTOp>(op);

    lutOpInputValIndecies.clear();

    auto lutInputs = lutOp.getInputs();
    auto numLutOprands = lutInputs.size();
    assert(numLutOprands <= 3);
    size_t pos = 0;
    for (; pos < 3 - numLutOprands; pos++) {
      // Note: the first elem in value pool is const 0
      assert(pos == lutOpInputValIndecies.size());
      lutOpInputValIndecies.push_back(0);
    }
    for (auto val: lutOp.getInputs()) {
      if (auto constOp = dyn_cast<toucan::ConstantOp>(val.getDefiningOp())) {
        // a const input
        auto constVal = constOp.getValue();
        auto bitWidth = constVal.getBitWidth();
        assert(bitWidth <= 4);
        auto rawVal = static_cast<uint8_t>(constVal.getZExtValue());
        // save op result value
        assert(rawVal == (rawVal & ((1 << bitWidth) - 1)));
        assert(pos == lutOpInputValIndecies.size());
        lutOpInputValIndecies.push_back(rawVal);
      } else {
        // a regular value. Must be in shuffle values
        assert(shuffleValueToId.contains(val));
        auto valId = shuffleValueToId.at(val);
        assert(pos == lutOpInputValIndecies.size());
        lutOpInputValIndecies.push_back(valId);
      }
      pos++;
    }
    assert(lutOpInputValIndecies.size() == 3);
  };

  
  for (size_t levelId = 1; levelId < mPart.levels.size(); levelId++) {
    std::swap(shuffleValueToId, shuffleValueToId_next);

    for (const auto &[val, _] : pendingWriteBackValueAndLocation) {
      if (valueToLifeCycle.contains(val)) {
        if (valueToLifeCycle.at(val).start < levelId) {
          // should be alive now
          // if (!shuffleValueToId.contains(val)) {
          //   dbgs() << "Level " << levelId << ", val:\n";
          //   val.print(dbgs());
          //   dbgs() << "\n";
          // }
          assert(shuffleValueToId.contains(val));
        }
      }
    }

    uint8_t opIndex = 0;
    part.middleLevels.emplace_back();
    shuffleValueToId_next.clear();

    part.middleLevels.back().reserve(32);
    shuffleValueToId_next.reserve(32);


    // Create NOP for pass through values
    for (const auto &[val, lifeTime]: valueToLifeCycle) {
      // if (!(lifeTime.end > lifeTime.start)) {
      //   dbgs() << "Value defined by:\n";
      //   val.getDefiningOp()->print(dbgs());
      //   dbgs() << "\nLife start at " << lifeTime.start << ", end at " << lifeTime.end << "\n";
      // }
      assert(lifeTime.end > lifeTime.start);

      if (isa<mlir::TypedValue<toucan::VecType>>(val)) {
        // a vector value, should only be used by first or last level, ignore
        continue;
      }

      // future value, ignore
      if (lifeTime.start > levelId) continue;
      if (lifeTime.start == levelId) {
        // value created by this level
        // valuesOutputOfThisLevel.insert(val);
      } else {
        // value created by previous level
        if (lifeTime.end > levelId) {
          // pass through
          assert(shuffleValueToId.contains(val));
          if (!shuffleValueToId_next.contains(val)) {
            // insert a dummy NOP op
            auto valShuffleId = shuffleValueToId[val];

            part.middleLevels.back().push_back({
              LUTOpName::LUT_Nop, 
              0, 0, valShuffleId});

            // get output id
            auto resultValShuffleId = static_cast<uint8_t>(opIndex + 32);
            shuffleValueToId_next[val] = resultValShuffleId;

            opIndex++;
            assert(opIndex == part.middleLevels.back().size());
          }
        }
      }
    }
    if (opIndex > 32) {
      dbgs() << "Too many ops in middle level as pass through vals!\n";
      mPart.print();
    }
    assert(opIndex <= 32);

    // Create NOP for pass through values that reads from outside (and thus not in valueLifeTime)
    for (const auto &[val, _]: pendingWriteBackValueAndLocation) {
      assert(shuffleValueToId.contains(val));
      if (!shuffleValueToId_next.contains(val)) {

        // insert a dummy NOP op
        auto valShuffleId = shuffleValueToId[val];

        part.middleLevels.back().push_back({
          LUTOpName::LUT_Nop, 
          0, 0, valShuffleId});

        // get output id
        auto resultValShuffleId = static_cast<uint8_t>(opIndex + 32);
        shuffleValueToId_next[val] = resultValShuffleId;

        opIndex++;
        assert(opIndex == part.middleLevels.back().size());
      }
    }
    if (opIndex > 32) {
      dbgs() << "Too many ops in middle level as pass through vals that reads from outside!\n";
      llvm::dbgs() << "\n";
      mPart.print();
    }
    assert(opIndex <= 32);

    for (auto &[val, _]: pendingWriteBackValueAndLocation) {
      if (valueToLifeCycle.contains(val)) {
        if (valueToLifeCycle[val].start <= levelId) {
          // Double check
          assert(shuffleValueToId.contains(val));
        }
      }
    }





    for (const auto eachVtx: mPart.levels[levelId]) {
      auto vtxIsDummyNop = newNodeIdToDepNodeId.contains(eachVtx);

      if (vtxIsDummyNop) {
        // vecdecl
        auto depValue = findDummyNopDepValue(eachVtx);
        // This value need to be passed through entire micro part
        assert(valueToLifeCycle.contains(depValue));
        assert(valueToLifeCycle[depValue].start < levelId);
        // Note: pass down to last level.

        // The input value must in somewhere in the shuffle network!
        assert(shuffleValueToId.contains(depValue));

        auto vecVtx = newNodeIdToOriginalVecDeclId.at(eachVtx);
        auto vecVal = cast<toucan::DefVectorOp>(graph[vecVtx].op).getHandle();

        auto indexInVec = dummyVtxIndexInVecTable.at(eachVtx);
        
        auto vecValIdInSMem = valToValId.at(vecVal);
        auto resultIndex = vecValIdInSMem + indexInVec;
        assert(resultIndex < UINT16_MAX);

        // this value should be written to smem at last level!
        pendingWriteBackValueAndLocation.push_back({depValue, static_cast<uint16_t>(resultIndex)});

        // also create a NOP if needed
        if (!shuffleValueToId_next.contains(depValue)) {
          part.middleLevels.back().push_back({
            LUTOpName::LUT_Nop, 
            0, 0, shuffleValueToId[depValue]});

          auto resultValShuffleId = static_cast<uint8_t>(opIndex + 32);
          shuffleValueToId_next[depValue] = resultValShuffleId;

          opIndex++;
          assert(opIndex == part.middleLevels.back().size());
        }

      } else {
        // regular lut op
        auto rawOp = graph[eachVtx].op;
        auto lutOp = cast<toucan::LUTOp>(rawOp);

        getLUTOpInputShuffleIds(rawOp);
        part.middleLevels.back().push_back({
          lutOp.getOpName(), 
          lutOpInputValIndecies[0], 
          lutOpInputValIndecies[1], 
          lutOpInputValIndecies[2]});

        // get output id
        auto resultVal = lutOp.getResult();

        assert(!shuffleValueToId_next.contains(resultVal));
        auto resultValShuffleId = static_cast<uint8_t>(opIndex + 32);
        shuffleValueToId_next[resultVal] = resultValShuffleId;

        opIndex++;
        assert(opIndex == part.middleLevels.back().size());

        // Result of a lut must be used somewhere, or just in output
        assert(valueToLifeCycle.contains(resultVal));
        assert(valueToLifeCycle[resultVal].start == levelId);
        // or an output value
        if (valueToLifeCycle[resultVal].end == WriteBackLevel) {
          assert(mPart.outputValueSet.contains(resultVal));
          assert(valToValId.contains(resultVal));
          
          auto resultValIdInSMem = valToValId.at(resultVal);
          pendingWriteBackValueAndLocation.push_back({resultVal, static_cast<uint16_t>(resultValIdInSMem)});
        }
      }
    }

    if (opIndex > 32) {
      dbgs() << "Too many ops in middle level!\n";
      llvm::dbgs() << "\n";
      mPart.print();
    }
    assert(opIndex <= 32);
    part.maxActiveVars = std::max(part.maxActiveVars, static_cast<int>(part.middleLevels.back().size()));
  }


  {
    // last level
    part.lastLevel.reserve(32);
    std::swap(shuffleValueToId, shuffleValueToId_next);
    // Create write back NOPs
    assert(pendingWriteBackValueAndLocation.size() <= 32);
    for (const auto &[val, sMemIdx]: pendingWriteBackValueAndLocation) {
      assert(shuffleValueToId.contains(val));
      auto valShuffleId = shuffleValueToId[val];

      part.lastLevel.push_back({valShuffleId, sMemIdx});
    }
    assert(part.lastLevel.size() <= 32);
    part.maxActiveVars = std::max(part.maxActiveVars, static_cast<int>(part.lastLevel.size()));
  }

}

void MultiRegionMicroPartScheduler::scheduleNOPMicroPart(CGMicroPartInfo &part, MicroPart &mPart, const mlir::DenseMap<mlir::Value, uint32_t> &valToValId) const {
  assert(mPart.opType == CGToucanOPName::LUT);
  assert(!mPart.nops.empty());
  assert(mPart.nops.size() <= 32);

  part.clear();
  part.opType = CGToucanOPName::LUT;

  auto getNOPInputId = [&](toucan::LUTOp &lutOp) {
    auto lutInputs = lutOp.getInputs();
    auto numLutOprands = lutInputs.size();
    assert(numLutOprands == 1);

    auto valId = valToValId.at(lutInputs[0]);
    return valId;
  };

  int opIndex = 0;
  for (auto &eachNOP: mPart.nops) {
    uint16_t inputValId = getNOPInputId(eachNOP);

    auto resultVal = eachNOP.getResult();

    assert(mPart.outputValueSet.contains(resultVal));
    assert(valToValId.contains(resultVal));

    part.topLevel.push_back({
      LUTOpName::LUT_Nop, 
      0, 
      0, 
      inputValId});

    auto resultValSMemIdx = static_cast<uint16_t>(valToValId.at(resultVal));
    part.lastLevel.push_back({static_cast<uint8_t>(opIndex), resultValSMemIdx});

    opIndex++;
  }
  assert(opIndex <= 32);
}

void MultiRegionMicroPartScheduler::scheduleSpecialMicroPart(const PartitioningGraph &graph, CGMicroPartInfo &part, const MicroPart &mPart, const CGPartitionMetaInfo &partInfo) const {
  const auto &valToValId = partInfo.valueToValId;
  const auto &toucanMemToId = codeGenInfo.toucanMemToId;
  const auto &memPool = codeGenInfo.memPool;

  assert(mPart.levels.size() == 0);
  assert(mPart.specialOps.size() != 0);
  assert(mPart.opType != CGToucanOPName::LUT);

  part.clear();
  part.opType = mPart.opType;

  uint8_t opIndex = 0;
  switch (mPart.opType) {
    case CGToucanOPName::VecRead: {
      for (auto rawOp: mPart.specialOps) {
        auto vecReadOp = cast<toucan::VectorReadOp>(rawOp);
        auto vecVal = vecReadOp.getHandle();

        assert(valToValId.contains(vecVal));
        auto vecValIdInSMem = valToValId.at(vecVal);
        assert(vecValIdInSMem < UINT16_MAX);

        auto vecLength = vecVal.getType().getLength();
        assert(vecLength < UINT16_MAX);

        bool isConstVec = isa<toucan::DefConstVectorOp>(vecVal.getDefiningOp());
        if (!isConstVec) assert(isa<toucan::DefVectorOp>(vecVal.getDefiningOp()));


        auto offset = vecReadOp.getOffset().getZExtValue();
        assert(offset < UINT16_MAX);

        auto outRangeValue = vecReadOp.getOutRangeValue();
        assert(valToValId.contains(outRangeValue));
        auto outRangeValueId = valToValId.at(outRangeValue);
        assert(outRangeValueId < UINT16_MAX);

        auto indexValues = vecReadOp.getIndicies();
        auto numIndexValues = indexValues.size();
        assert(numIndexValues <= 4 && "Index is too long");

        std::array<uint32_t, 4> vecReadOpIndexIds;
        size_t pos = 0;
        for (; pos < 4 - numIndexValues; pos++) {
          // Note: the first elem in value pool is const 0
          vecReadOpIndexIds[pos] = 0;
        }
        for (auto val: indexValues) {
          assert(valToValId.contains(val));
          auto valId = valToValId.at(val);
          assert(valId < UINT16_MAX);
          vecReadOpIndexIds[pos] = valId;
          pos++;
        }

        auto resultVal = vecReadOp.getResult();
        assert(valToValId.contains(resultVal));
        auto resultValId = valToValId.at(resultVal);
        assert(resultValId < UINT16_MAX);

        part.vecRead.push_back({
          static_cast<uint16_t>(vecValIdInSMem),
          static_cast<uint16_t>(vecLength),
          isConstVec,
          static_cast<uint16_t>(offset),
          static_cast<uint16_t>(vecReadOpIndexIds[0]),
          static_cast<uint16_t>(vecReadOpIndexIds[1]),
          static_cast<uint16_t>(vecReadOpIndexIds[2]),
          static_cast<uint16_t>(vecReadOpIndexIds[3]),
          static_cast<uint16_t>(outRangeValueId),
          static_cast<uint16_t>(resultValId)
        });

        assert(opIndex < 32);
        opIndex++;
        assert(opIndex == part.vecRead.size());
      }
      break;
    }
    case CGToucanOPName::VecLogic:{
      for (auto rawOp: mPart.specialOps) {
        auto vecLogicOp = cast<toucan::VectorLogicOp>(rawOp);

        auto v1Val = vecLogicOp.getV1();
        assert(valToValId.contains(v1Val));
        auto v1ValIdInSmem_raw = valToValId.at(v1Val);
        assert(v1ValIdInSmem_raw < UINT16_MAX);
        auto v1ValIdInSmem = static_cast<uint16_t>(v1ValIdInSmem_raw);
        auto v1VecLength = v1Val.getType().getLength();

        auto v2Val = vecLogicOp.getV2();
        assert(valToValId.contains(v2Val));
        auto v2ValIdInSmem_raw = valToValId.at(v2Val);
        assert(v2ValIdInSmem_raw < UINT16_MAX);
        auto v2ValIdInSmem = static_cast<uint16_t>(v2ValIdInSmem_raw);
        auto v2VecLength = v2Val.getType().getLength();

        assert(v1VecLength == v2VecLength);
        bool isV1ConstVec = isa<toucan::DefConstVectorOp>(v1Val.getDefiningOp());
        bool isV2ConstVec = isa<toucan::DefConstVectorOp>(v2Val.getDefiningOp());

        auto resultVal = vecLogicOp.getResult();
        assert(valToValId.contains(resultVal));
        auto resultValIdInSmem_raw = valToValId.at(resultVal);
        assert(resultValIdInSmem_raw < UINT16_MAX);
        auto resultValIdInSmem = static_cast<uint16_t>(resultValIdInSmem_raw);

        assert(v1VecLength < UINT16_MAX);

        part.vecLogic.push_back({
          v1ValIdInSmem,
          v2ValIdInSmem,
          static_cast<uint16_t>(v1VecLength),
          isV1ConstVec,
          isV2ConstVec,
          vecLogicOp.getOpName(),
          resultValIdInSmem
        });

        assert(opIndex < 32);
        opIndex++;
        assert(opIndex == part.vecLogic.size());
      }
      break;
    }
    case CGToucanOPName::VecArith:{
      for (auto rawOp: mPart.specialOps) {
        auto vecArithOp = cast<toucan::VectorArithOp>(rawOp);

        auto v1Val = vecArithOp.getV1();
        assert(valToValId.contains(v1Val));
        auto v1ValIdInSmem_raw = valToValId.at(v1Val);
        assert(v1ValIdInSmem_raw < UINT16_MAX);
        auto v1ValIdInSmem = static_cast<uint16_t>(v1ValIdInSmem_raw);
        auto v1VecLength = v1Val.getType().getLength();

        auto v2Val = vecArithOp.getV2();
        assert(valToValId.contains(v2Val));
        auto v2ValIdInSmem_raw = valToValId.at(v2Val);
        assert(v2ValIdInSmem_raw < UINT16_MAX);
        auto v2ValIdInSmem = static_cast<uint16_t>(v2ValIdInSmem_raw);
        auto v2VecLength = v2Val.getType().getLength();

        assert(v1VecLength == v2VecLength);
        bool isV1ConstVec = isa<toucan::DefConstVectorOp>(v1Val.getDefiningOp());
        bool isV2ConstVec = isa<toucan::DefConstVectorOp>(v2Val.getDefiningOp());

        auto resultVal = vecArithOp.getResult();
        assert(valToValId.contains(resultVal));
        auto resultValIdInSmem_raw = valToValId.at(resultVal);
        assert(resultValIdInSmem_raw < UINT16_MAX);
        auto resultValIdInSmem = static_cast<uint16_t>(resultValIdInSmem_raw);
        auto resultVecLength = v1Val.getType().getLength();

        assert(resultVecLength == v1VecLength);
        assert(v1VecLength < UINT16_MAX);

        part.vecArith.push_back({
          v1ValIdInSmem,
          v2ValIdInSmem,
          static_cast<uint16_t>(v1VecLength),
          isV1ConstVec,
          isV2ConstVec,
          vecArithOp.getOpName(),
          resultValIdInSmem
        });

        assert(opIndex < 32);
        opIndex++;
        assert(opIndex == part.vecArith.size());
      }
      break;
    }
    case CGToucanOPName::MemRead:{
      for (auto rawOp: mPart.specialOps) {
        auto memReadOp = cast<toucan::MemReadOp>(rawOp);
        auto memVal = memReadOp.getMem();
        auto memValId = toucanMemToId.at(memVal);
        auto memEnVal = memReadOp.getEn();
        assert(valToValId.contains(memEnVal));
        auto memEnId = valToValId.at(memEnVal);
        assert(memEnId < UINT16_MAX);

        auto memAddrVec = memReadOp.getAddrVec();
        assert(memAddrVec.getType().getLength() == 8 && "Memory address should be a 32 bit vector");
        auto memAddrVecId = valToValId.at(memAddrVec);
        assert(memAddrVecId < UINT16_MAX);

        auto hasMultipleWriter = memPool[memValId].hasMultipleWriter;
        auto memBase = memPool[memValId].memBase;
        auto memDepth = memPool[memValId].memDepth;
        assert(memDepth < UINT32_MAX);

        auto resultVal = memReadOp.getResult();
        auto resultValId = valToValId.at(resultVal);
        assert(resultValId < UINT16_MAX);

        part.memRead.push_back({
          hasMultipleWriter,
          static_cast<uint32_t>(memDepth),
          memBase,
          static_cast<uint16_t>(memEnId),
          static_cast<uint16_t>(memAddrVecId),
          static_cast<uint16_t>(resultValId)
        });

        assert(opIndex < 32);
        opIndex++;
        assert(opIndex == part.memRead.size());
      }
      break;
    }

    default: llvm_unreachable("Unexpected op type");
  }
}

void MultiRegionMicroPartScheduler::mergeSmallRegularMParts(std::vector<CGMicroPartInfo> &mPartsThisLevel) {
  auto mergeMP = [](CGMicroPartInfo &to, CGMicroPartInfo &from) {
    assert(to.opType == from.opType);
    assert(to.maxActiveVars > 0);
    assert(from.maxActiveVars > 0);

    assert(to.maxActiveVars <= 32);
    assert(from.maxActiveVars <= 32);
    // Add this limitation to simplify implementation
    assert(to.middleLevels.size() >= from.middleLevels.size());

    // maps shuffle val old -> new
    mlir::DenseMap<uint16_t, uint16_t> shuffleValMap, shuffleValMap_next;
    // std::unordered_map<uint16_t, uint16_t> shuffleValMap, shuffleValMap_next;

    // top level
    assert(from.topLevel.size() <= static_cast<size_t>(from.maxActiveVars));
    assert(to.topLevel.size() <= static_cast<size_t>(to.maxActiveVars));

    for (size_t i = 0; i < from.topLevel.size(); i++) {
      // result val i is now mapped to to.topLevel.size()
      auto newResultValId = to.topLevel.size();
      // assert(!shuffleValMap.contains(i+32));
      shuffleValMap[i + 32] = newResultValId + 32;

      to.topLevel.push_back(from.topLevel[i]);
      assert(to.topLevel.size() <= 32);
      assert(i <= 32);
    }
    to.maxActiveVars = std::max(to.maxActiveVars, static_cast<int>(to.topLevel.size()));

    auto getMappedShuffleVal = [&shuffleValMap](uint16_t oldId) {
      if (oldId < 32) return oldId;
      assert(shuffleValMap.contains(oldId));
      return shuffleValMap[oldId];
    };


    // copy ops in middle level
    for (size_t middle_level_id = 0; middle_level_id < from.middleLevels.size(); middle_level_id++) {
      shuffleValMap_next.clear();

      const auto &thisLevelOps = from.middleLevels[middle_level_id];

      for (size_t i = 0; i < thisLevelOps.size(); i++) {
        const auto &fromOp = thisLevelOps[i];

        auto newShuffleValId = to.middleLevels[middle_level_id].size();

        CGMicroPartLUTMiddleLevelOp toOp;
        toOp.opName = fromOp.opName;
        toOp.op0 = getMappedShuffleVal(fromOp.op0);
        toOp.op1 = getMappedShuffleVal(fromOp.op1);
        toOp.op2 = getMappedShuffleVal(fromOp.op2);

        to.middleLevels[middle_level_id].push_back(toOp);
        assert(to.middleLevels[middle_level_id].size() <= 32);

        shuffleValMap_next[i + 32] = newShuffleValId + 32;
      }

      to.maxActiveVars = std::max(to.maxActiveVars, static_cast<int>(to.middleLevels[middle_level_id].size()));
      std::swap(shuffleValMap, shuffleValMap_next);
    }


    // create NOPs for remaining middle levels
    for (size_t middle_level_id = from.middleLevels.size(); middle_level_id < to.middleLevels.size(); middle_level_id++) {
      shuffleValMap_next.clear();

      for (const auto &[shuffleValFrom, oldShuffleValId]: shuffleValMap) {

        auto newShuffleValId = to.middleLevels[middle_level_id].size();

        to.middleLevels[middle_level_id].push_back({
              LUTOpName::LUT_Nop, 
              0, 0, static_cast<uint8_t>(oldShuffleValId)});
        assert(to.middleLevels[middle_level_id].size() <= 32);

        shuffleValMap_next[shuffleValFrom] = newShuffleValId + 32;
      }

      to.maxActiveVars = std::max(to.maxActiveVars, static_cast<int>(to.middleLevels[middle_level_id].size()));
      std::swap(shuffleValMap, shuffleValMap_next);
    }


    // last level
    for (size_t i = 0; i < from.lastLevel.size(); i++) {
      const auto &op = from.lastLevel[i];

      CGMicroPartLUTLastLevelWriteBack newOp;

      newOp.shuffleId = getMappedShuffleVal(op.shuffleId);
      newOp.result = op.result;

      to.lastLevel.push_back(newOp);
      assert(to.lastLevel.size() <= 32);
    }
    to.maxActiveVars = std::max(to.maxActiveVars, static_cast<int>(to.lastLevel.size()));

    assert(to.maxActiveVars <= 32);

    // clear
    from.clear();
  };


  size_t mergeCount;
  size_t iterCount = 0;

  std::vector<uint32_t> mPartIds;
  mlir::DenseSet<uint32_t> removedMParts;

  while (true) {
    mergeCount = 0;
    iterCount++;
    assert(iterCount < 100 && "This merge usually finish in 2-3 iterations.");

    mPartIds.clear();
    for (uint32_t i = 0; i < mPartsThisLevel.size(); i++) {
      if (!removedMParts.contains(i)) {
        mPartIds.push_back(i);
        assert(mPartsThisLevel[i].maxActiveVars != -1);
      } else {
        assert(mPartsThisLevel[i].maxActiveVars == -1);
      }
    }

    if (mPartIds.size() <= 1) break;

    std::sort(mPartIds.begin(), mPartIds.end(), [&mPartsThisLevel](const auto a, const auto b) {
      return mPartsThisLevel.at(a).middleLevels.size() > mPartsThisLevel.at(b).middleLevels.size();
    });
    std::stable_sort(mPartIds.begin(), mPartIds.end(), [&mPartsThisLevel](const auto a, const auto b) {
      return mPartsThisLevel.at(a).maxActiveVars > mPartsThisLevel.at(b).maxActiveVars;
    });

    size_t frontPtr = 0;
    size_t lastPtr = mPartIds.size() - 1;

    while ((frontPtr < mPartIds.size()) && (mPartsThisLevel[mPartIds.at(frontPtr)].maxActiveVars >= 32)) {
      assert(mPartsThisLevel[mPartIds.at(frontPtr)].maxActiveVars == 32);
      frontPtr++;
    }

    while (frontPtr < lastPtr) {
      auto frontMpId = mPartIds.at(frontPtr);
      auto lastMpId = mPartIds.at(lastPtr);
      assert(frontMpId != lastMpId);
      auto &frontMPart = mPartsThisLevel[frontMpId];
      auto &lastMPart = mPartsThisLevel[lastMpId];

      assert(!removedMParts.contains(frontMpId));
      assert(!removedMParts.contains(lastMpId));
      assert(frontMPart.maxActiveVars > 0);
      assert(lastMPart.maxActiveVars > 0);
      if (frontMPart.maxActiveVars + lastMPart.maxActiveVars <= 32) {
        // ok to merge
        auto front_mpart_levels = frontMPart.middleLevels.size();
        auto last_mpart_levels = lastMPart.middleLevels.size();

        if (front_mpart_levels >= last_mpart_levels) {
          // merge lastMP into frontMP. After this, lastMP become invalid
          mergeMP(frontMPart, lastMPart);
          removedMParts.insert(lastMpId);
          lastPtr--;
        } else {
          // merge frontMP into lastMP. After this, frontMP become invalid
          mergeMP(lastMPart, frontMPart);
          removedMParts.insert(frontMpId);
          frontPtr++;
          // Though lastMP is still valid, skip it and merge in next iteration
          lastPtr--;
        }

        mergeCount++;
      } else {
        frontPtr++;
      }
    }

    // llvm::outs() << "Perform " << mergeCount << " merges at iteration " << iterCount << "\n";

    if (mergeCount == 0) break;
  }

  std::vector<CGMicroPartInfo> newMParts;
  for (uint32_t i = 0; i < mPartsThisLevel.size(); i++) {
    if (!removedMParts.contains(i)) {
      newMParts.push_back(mPartsThisLevel[i]);
    }
  }

  std::swap(newMParts, mPartsThisLevel);
}



void MultiRegionMicroPartScheduler::fillSignalDebugInfoForSinglePart(const MicroPartLocalValueAllocator &valAllocator, uint32_t partId) {
  std::lock_guard<std::mutex> lock_guard(debugSymbolLock);

  // collect signal info, only in valuePool
  for (const auto &val: valAllocator.activeValuesAtLast) {
    auto valId = valAllocator.valToValId.at(val);

    auto valDefiningOp = val.getDefiningOp();
    auto namehint = getSVNameHintAttr(valDefiningOp);

    if (namehint) {
      auto name_str = namehint.value().getValue();
      uint32_t fragmentId = 0;
      auto fragment_id = getSignalFragmentIDAttr(valDefiningOp);
      if (fragment_id) {
        auto fragment_id_int = fragment_id.value().getInt();
        fragmentId = fragment_id_int;
      }

      if (!codeGenInfo.signalDebugInfo.contains(name_str)) {
        codeGenInfo.signalDebugInfo.try_emplace(name_str);
      }

      uint32_t expectedSectionCount = fragmentId + 1;

      // fill placeholder for non existing value
      while (codeGenInfo.signalDebugInfo[name_str].size() < expectedSectionCount) {
        codeGenInfo.signalDebugInfo[name_str].push_back({UINT32_MAX, UINT32_MAX, UINT32_MAX});
      }

      auto valBitWidth = hw::getBitWidth(val.getType());
      codeGenInfo.signalDebugInfo[name_str][fragmentId] = {partId, valId, valBitWidth};
    }
  }

  // TODO: complete dump for regular signals
  // Consider: Is it necessary? most signals are hidden in warp shuffle, saving limited number of signals may not be helpful for debugging
  return;
}

void MultiRegionMicroPartScheduler::fillDebugInfo() {

  // Collect io signals and extern module signals
  for (size_t regId = 0; regId < codeGenInfo.regPool.size(); regId++) {
    auto &regMeta = codeGenInfo.regPool[regId];
    if (regMeta.isIO) {
      assert(regMeta.namehint && "An io signal must have a name");
      auto namehint = regMeta.namehint.value();
      codeGenInfo.ioSignals.insert(namehint);
    }
  }


  // collect reg info
  for (size_t regId = 0; regId < codeGenInfo.regPool.size(); regId++) {
    auto &regMeta = codeGenInfo.regPool[regId];
    if (regMeta.namehint) {
      // has name hint
      auto namehint = regMeta.namehint.value();
      // every named reg should have a fragment id
      auto fragment_id = regMeta.fragment_id;
      assert(fragment_id != UINT32_MAX);
      if (!codeGenInfo.regDebugInfo.contains(namehint)) {
        codeGenInfo.regDebugInfo.try_emplace(namehint);
      }
      codeGenInfo.regDebugInfo[namehint].push_back(regId);
    }
  }
  // TODO: A temporary fix that removes fragment collision
  mlir::SmallVector<mlir::StringRef> toRemove;
  // sort by fragment id
  for (auto &elem: codeGenInfo.regDebugInfo) {
    auto &k = elem.getFirst();
    auto &v = elem.getSecond();
    std::sort(v.begin(), v.end(), [&](const uint32_t a, const uint32_t b) {
      auto a_fragmentId = codeGenInfo.regPool[a].fragment_id;
      auto b_fragmentId = codeGenInfo.regPool[b].fragment_id;
      // fragment Id should not duplicate
      // if (a_fragmentId == b_fragmentId) {
      //   llvm::dbgs() << "Reg a " << a << ", fragment id " << a_fragmentId << ", b " << b << ", fragment id " << b_fragmentId << "\n";
      // }
      if (a_fragmentId == b_fragmentId) {
        toRemove.push_back(k);
      }
      // assert(a_fragmentId != b_fragmentId);
      return a_fragmentId > b_fragmentId;
    });
  }

  for (auto k: toRemove) {
    codeGenInfo.regDebugInfo.erase(k);
  }
  // Note: Some register might be removed for optimization purpose, thus the debug info might not be complete.
  // // Verify fragment id correctness
  // for (auto &elem: codeGenInfo.regDebugInfo) {
  //   auto &v = elem.getSecond();
  //   if (v.size() == 1) continue;
  //   uint32_t expected_id = v.size() - 1;
  //   for (const auto &ei: v) {
  //     assert(expected_id != UINT32_MAX);
  //     auto fragment_id = codeGenInfo.regPool[ei].fragment_id;
  //     assert(fragment_id == expected_id);
  //     expected_id--;
  //   }
  // }




  // collect mem info
  for (size_t memId = 0; memId < codeGenInfo.memPool.size(); memId++) {
    auto &memMeta = codeGenInfo.memPool[memId];
    if (memMeta.namehint) {
      // has name hint
      auto namehint = memMeta.namehint.value();
      // Every named memory should have a fragment id
      auto fragment_id = memMeta.fragment_id;
      assert(fragment_id != UINT32_MAX);
      if (!codeGenInfo.memDebugInfo.contains(namehint)) {
        codeGenInfo.memDebugInfo.try_emplace(namehint);
      }
      codeGenInfo.memDebugInfo[namehint].push_back(memId);
    }
  }
  // sort by fragment id
  for (auto &elem: codeGenInfo.memDebugInfo) {
    auto &v = elem.getSecond();
    std::sort(v.begin(), v.end(), [&](const uint32_t a, const uint32_t b) {
      auto a_fragmentId = codeGenInfo.memPool[a].fragment_id;
      auto b_fragmentId = codeGenInfo.memPool[b].fragment_id;
      // fragment Id should not duplicate
      assert(a_fragmentId != b_fragmentId);
      return a_fragmentId > b_fragmentId;
    });
  }
  // // Verify fragment id correctness
  // for (auto &elem: codeGenInfo.memDebugInfo) {
  //   auto &v = elem.getSecond();
  //   if (v.size() == 1) continue;
  //   uint32_t expected_id = v.size() - 1;
  //   for (const auto &ei: v) {
  //     assert(expected_id != UINT32_MAX);
  //     auto fragment_id = codeGenInfo.memPool[ei].fragment_id;
  //     assert(fragment_id == expected_id);
  //     expected_id--;
  //   }
  // }



  // Ignore signals for now

  // assert(codeGenInfo.signalDebugInfo.size() != 0);


  // // Refactor signal namehint
  // // Note: some signal might be duplicated. Here we only need 1 copy
  // std::unordered_set<uint32_t> signalFragments;
  // mlir::SmallVector<std::tuple<uint32_t, uint32_t>> dedupInfos;
  // for (auto &elem: codeGenInfo.signalDebugInfo) {
  //   // auto namehint = elem.getFirst();
  //   auto &infos = elem.getSecond();

  //   if (infos.size() > 1) {
  //     // Possibly duplication
  //     dedupInfos.clear();
  //     signalFragments.clear();

  //     for (const auto &eachFragment: infos) {
  //       auto partId = std::get<0>(eachFragment);
  //       auto valId = std::get<1>(eachFragment);
  //       uint32_t fragment_id = codeGenInfo.partitionInfo[partId].valuePool[valId].fragment_id;

  //       if (!signalFragments.contains(fragment_id)) {
  //         // a new fragment
  //         signalFragments.insert(fragment_id);
  //         dedupInfos.push_back(eachFragment);
  //       }
  //     }

  //     if (dedupInfos.size() != infos.size()) {
  //       std::swap(dedupInfos, infos);
  //     }
  //   }
  // }

  // reverse signal fragments. (MSB first)
  for (auto &elem: codeGenInfo.signalDebugInfo) {
    auto &v = elem.getSecond();
    std::reverse(v.begin(), v.end());
  }

  // Don't verify fragment id correctness
  // For regular signals, some fragments can be missing due to optimizations

}

void MultiRegionMicroPartScheduler::printPartInfo(const CGPartitionMetaInfo &partInfo) {
  llvm::dbgs() << "  Part has " << partInfo.regReadOps.size() << " RegReads, " << partInfo.exchangeReadOps.size() << " ExchangeReads, " << partInfo.microPartOps.size() << " MP levels\n";

  std::unordered_map<size_t, size_t> mPartMaxActiveVarsToCount;
  std::unordered_map<size_t, size_t> mPartSizeToCount;
  std::unordered_map<size_t, size_t> mPartLevelsToCount;
  std::unordered_map<size_t, size_t> specialMPSizeToCount;

  size_t allRegularMPs = 0;
  size_t allSpecialMPs = 0;
  for (size_t levelId = 0; levelId < partInfo.microPartOps.size(); levelId++) {
    size_t regularMPCount = 0;
    size_t specialVRMPCount = 0;
    size_t specialVAMPCount = 0;
    size_t specialVLMPCount = 0;
    size_t specialMRMPCount = 0;

    for (const auto &eachMP: partInfo.microPartOps[levelId]) {
      if (eachMP.opType == CGToucanOPName::LUT) {
        // a regular part
        regularMPCount++;

        size_t activeVars = eachMP.maxActiveVars;
        size_t partLevels = eachMP.middleLevels.size() + 2;
        size_t partSize = 0;
        for (const auto &el: eachMP.middleLevels) {
          partSize += el.size();
        }
        partSize += eachMP.topLevel.size();
        partSize += eachMP.lastLevel.size();

        if (!mPartMaxActiveVarsToCount.contains(activeVars)) {
          mPartMaxActiveVarsToCount[activeVars] = 0;
        }
        mPartMaxActiveVarsToCount[activeVars]++;

        if (!mPartSizeToCount.contains(partSize)) {
          mPartSizeToCount[partSize] = 0;
        }
        mPartSizeToCount[partSize]++;

        if (!mPartLevelsToCount.contains(partLevels)) {
          mPartLevelsToCount[partLevels] = 0;
        }
        mPartLevelsToCount[partLevels]++;
      } else {
        if (!eachMP.vecRead.empty()) {
          specialVRMPCount++;
        }
        if (!eachMP.vecArith.empty()) {
          specialVAMPCount++;
        }
        if (!eachMP.vecLogic.empty()) {
          specialVLMPCount++;
        }
        if (!eachMP.memRead.empty()) {
          specialMRMPCount++;
        }

        size_t partSize = eachMP.vecRead.size() + eachMP.vecArith.size() + eachMP.vecLogic.size() + eachMP.memRead.size();

        if (!specialMPSizeToCount.contains(partSize)) {
          specialMPSizeToCount[partSize] = 0;
        }
        specialMPSizeToCount[partSize]++;
      }
    }

    llvm::dbgs() << "    Level " << levelId << " has " << regularMPCount << " regular MP, " << specialVRMPCount << " VectorRead MP, " << specialVAMPCount << " VecArith MP, " << specialVLMPCount << " VecLogic MP, " << specialMRMPCount << " MemRead MP.\n";

    allRegularMPs += regularMPCount;
    allSpecialMPs += specialVRMPCount;
    allSpecialMPs += specialVAMPCount;
    allSpecialMPs += specialVLMPCount;
    allSpecialMPs += specialMRMPCount;
  }

  llvm::dbgs() << "  This part has " << allRegularMPs << " regular MPs, " << allSpecialMPs << " special MPs.\n";

  size_t avgMaxActiveVars = 0;
  size_t mpCount = 0;
  for (const auto &[k, v]: mPartMaxActiveVarsToCount) {
    avgMaxActiveVars += (k * v);
    mpCount += v;
  }
  llvm::dbgs() << "  Average max active vars is " << (static_cast<float>(avgMaxActiveVars) / mpCount) << "\n";

  llvm::dbgs() << "  Part has " << partInfo.regWriteOps.size() << " RegWrites, " << partInfo.exchangeWriteOps.size() << " ExchangeWrites, " << partInfo.memWriteOps.size() << " MemWrites, " << partInfo.stopOps.size() << " Stops, " << partInfo.printOps.size() << " Prints.\n";

}

// Scheduler entry point
mlir::LogicalResult MultiRegionMicroPartScheduler::schedule(mlir::MLIRContext *context, const PartitioningGraph &rawGraph, const mlir::SmallVector<mlir::Value> &exchangeValPool) {


  // schedule all registers, memories and exchange values. Also sort registers and exchange writes
  generateRegMemLayout(rawGraph);
  generateExchangeLayout(exchangeValPool);

  // dedup strings
  collectPrintString(rawGraph, codeGenInfo.printStrings);
  if (codeGenInfo.printStrings.size() > UINT16_MAX) {
    llvm::errs() << "Too many print strings! (current max is " << UINT16_MAX << ")\n";
    llvm_unreachable("Too many print strings");
  }
  assert(codeGenInfo.printStrings.size() <= UINT16_MAX);

  collectConstantVars(rawGraph);


  
  size_t totalNumParts = 0;
  for (const auto &eachRegion: regionPartData) {
    totalNumParts += eachRegion.size();
  }
  codeGenInfo.partitionInfo.resize(totalNumParts);
  

  // schedule ops
  uint32_t startPartIdInThisRegion = 0;
  for (uint32_t regionId = 0; regionId < regionPartData.size(); regionId++) {
    auto numPartsInThisRegion = regionPartData[regionId].size();

    codeGenInfo.regionPartitionIds.emplace_back();
    for (auto i: llvm::seq(static_cast<uint32_t>(0), static_cast<uint32_t>(numPartsInThisRegion))) {
      codeGenInfo.regionPartitionIds.back().push_back(startPartIdInThisRegion + i);
    }
    

    llvm::outs() << "Schedule ops for region " << regionId << "\n";


    auto scheduleStats = mlir::failableParallelForEachN(context, 0, numPartsInThisRegion, [&](size_t partIndexInThisRegion) {
      size_t partId = startPartIdInThisRegion + partIndexInThisRegion;


      auto start = std::chrono::system_clock::now();

      std::ostringstream oss;
      assert(codeGenInfo.partitionInfo.size() > partId);
      auto &partInfo = codeGenInfo.partitionInfo[partId];
      auto &partData = regionPartData[regionId].at(partIndexInThisRegion);

      std::memset(&partInfo.opStatistics, 0, sizeof(CGOpStatistics));

      collectConstantVecs(partData, partInfo);



      // allocate space for values in shared mem
      MicroPartLocalValueAllocator valAllocator;
      valAllocator.collectValueLifetime(partData);
      valAllocator.populateInitialPinnedVals(partData, constValToRawValue);

      #ifdef VAL_ALLOCATOR_DONT_RECLAIM
      valAllocator.allocateLocalValuesWithoutReclaim();
      #else
      valAllocator.allocateLocalValues();
      #endif
      // save const value pool
      std::swap(partInfo.constValuePool, valAllocator.compactConstValPool);
      partInfo.numConstsInValuePool = partInfo.constValuePool.size();
      // save value location
      for (auto [k, v]: valAllocator.valToValId) {
        assert(!partInfo.valueToValId.contains(k));
        partInfo.valueToValId[k] = v;
      }
      // ensure const vector is not handled. They are in const vec pool
      for (auto [k, _]: valAllocator.valToValId) {
        assert(!isa<toucan::DefConstVectorOp>(k.getDefiningOp()));
      }


      partInfo.numTotalValues = valAllocator.numTotalValSize;
      oss
        << "Region " << regionId
        << " part " << partId
        << " has " << partData.mpartLevels.size()
        << " levels, "
        << partInfo.valueToValId.size() 
        << " active values, allocator requires " << valAllocator.numTotalValSize 
        << " total bytes (" << valAllocator.numConsts 
        << " consts, " << valAllocator.numOutputVals 
        << " for output, " << valAllocator.numInputVals 
        << " for input). constVecPool size of " << partInfo.constVecPool.size() << "\n";
      llvm::outs() << oss.str();

      if (valAllocator.numTotalValSize >= UINT16_MAX) {
        oss.str("");
        oss
          << "Region " << regionId
          << " part " << partId
          << " allocator requires " << valAllocator.numTotalValSize
          << " total bytes, exceeds UINT16_MAX, cannot proceed.\n";
        llvm::outs() << oss.str();
        // llvm_unreachable("Consider lower PARTITION_MAX_WEIGHT");
        return failure();
      }

      scheduleRegReads(partInfo, partData.allRegReads);
      scheduleExchangeReads(partInfo, partData.allExgReadVals);
      assert(partInfo.regReadOps.empty() || partInfo.exchangeReadOps.empty());

      // Save statistics
      {
        CGLayerValueStatistics stats;
        std::memset(&stats, 0, sizeof(CGLayerValueStatistics));
        stats.numRegReads = partInfo.regReadOps.size();
        partInfo.opStatisticsPerLevel.push_back(stats);
        partInfo.opStatistics.numRegReads = partInfo.regReadOps.size();
      }




      for (uint32_t levelId = 0; levelId < partData.mpartLevels.size(); levelId++) {
        partInfo.microPartOps.emplace_back();
        std::vector<CGMicroPartInfo> regularMPartsThisLevel;

        for (const auto &eachMPart: partData.mpartLevels[levelId]) {
          // schedule each mpart
          bool isRegularPart = eachMPart->isRegularPart();
          bool isNOPPart = eachMPart->isNOPPart;

          CGMicroPartInfo newPart;

          if (isNOPPart) {
            assert(isRegularPart);
            scheduleNOPMicroPart(newPart, *eachMPart, partInfo.valueToValId);
            partInfo.microPartOps.back().push_back(newPart);
          } else {
            if (isRegularPart) {
              scheduleRegularMicroPart(rawGraph, newPart, *eachMPart, partInfo.valueToValId);
              // Save for merge
              regularMPartsThisLevel.push_back(newPart);
            } else {
              scheduleSpecialMicroPart(rawGraph, newPart, *eachMPart, partInfo);
              partInfo.microPartOps.back().push_back(newPart);
            }
          }

          // add code gen meta
        }
        // llvm::dbgs() << "Merge at level " << levelId << ", " << regularMPartsThisLevel.size() << " regular mparts\n";
        mergeSmallRegularMParts(regularMPartsThisLevel);
        partInfo.microPartOps.back().reserve(partInfo.microPartOps.back().size() + regularMPartsThisLevel.size());
        for (size_t i = 0; i < regularMPartsThisLevel.size(); i++) {
          partInfo.microPartOps.back().push_back(regularMPartsThisLevel[i]);
        }
      }


      // last level
      scheduleRegWrites(partInfo, partData.allRegWrites);
      scheduleMemWrites(partInfo, partData.allMemWrites);
      scheduleStops(partInfo, partData.allStops);
      schedulePrints(partInfo, partData.allPrints);
      scheduleExchangeWrites(partInfo, partData.allExgWriteVals);
      assert(partInfo.exchangeWriteOps.empty() || (
        (
          partInfo.regWriteOps.empty()
          && partInfo.memWriteOps.empty()
          && partInfo.stopOps.empty()
          && partInfo.printOps.empty()
        )
      ));


      // Add statistics
      {
        CGLayerValueStatistics stats;
        std::memset(&stats, 0, sizeof(CGLayerValueStatistics));
        stats.numRegWrites = partInfo.regWriteOps.size();
        stats.numMemWrites = partInfo.memWriteOps.size();
        stats.numPrints = partInfo.printOps.size();
        stats.numStops = partInfo.stopOps.size();
        stats.numExchangeReads = partInfo.exchangeReadOps.size();
        stats.numExchangeWrites = partInfo.exchangeWriteOps.size();
      
        partInfo.opStatisticsPerLevel.push_back(stats);
        partInfo.opStatistics.numRegWrites = stats.numRegWrites;
        partInfo.opStatistics.numMemWrites = stats.numMemWrites;
        partInfo.opStatistics.numPrints = stats.numPrints;
        partInfo.opStatistics.numStops = stats.numStops;
        partInfo.opStatistics.numExchangeReads = stats.numExchangeReads;
        partInfo.opStatistics.numExchangeWrites = stats.numExchangeWrites;
      }

      fillSignalDebugInfoForSinglePart(valAllocator, partId);

      auto stop = std::chrono::system_clock::now();
      auto duration = std::chrono::duration_cast<std::chrono::milliseconds>(stop - start);
      uint64_t time_ms = duration.count();

      oss.str("");
      oss << "Done schedule ops for partition " << partId << " in " << time_ms << "ms\n";
      llvm::outs() << oss.str();

      return success();
    });

    assert(succeeded(scheduleStats));
    startPartIdInThisRegion += numPartsInThisRegion;
  }
  assert(startPartIdInThisRegion == totalNumParts);

  fillDebugInfo();

  return success();
}





