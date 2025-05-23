#include "circt/Dialect/HW/HWTypes.h"
#include "circt/Support/LLVM.h"
#include "circt/Dialect/Comb/CombDialect.h"
#include "circt/Dialect/Comb/CombOps.h"
#include "circt/Dialect/Seq/SeqOps.h"

#include "mlir/IR/Value.h"
#include "mlir/Pass/AnalysisManager.h"
#include "mlir/Support/LLVM.h"
#include "toucan/MicroPartLocalValueAllocator.h"
#include "toucan/MicroPartitioner.h"
#include "toucan/ToucanAnalysis.h"
#include "toucan/ToucanAttributes.h"
#include "toucan/ToucanCodeGenInfo.h"
#include "toucan/ToucanOps.h"
#include "toucan/ToucanTypes.h"
#include "toucan/ToucanUtils.h"

#include "llvm/ADT/STLExtras.h"
#include "llvm/ADT/SmallVector.h"
#include "llvm/Support/Casting.h"
#include "llvm/Support/Debug.h"
#include "llvm/Support/ErrorHandling.h"
#include <cassert>
#include <cstddef>
#include <cstdint>

#include <boost/graph/topological_sort.hpp>
#include <cstring>
#include <iterator>
#include <array>
#include <optional>
#include <tuple>
#include <vector>


using namespace toucan;

using namespace mlir;
using namespace llvm;
using namespace circt;





#include "circt/Dialect/HW/HWTypes.h"
#include "circt/Support/LLVM.h"
#include "circt/Dialect/Comb/CombDialect.h"
#include "circt/Dialect/Comb/CombOps.h"
#include "circt/Dialect/Seq/SeqOps.h"

#include "mlir/IR/Operation.h"
#include "mlir/IR/PatternMatch.h"
#include "mlir/IR/Value.h"
#include "mlir/Pass/AnalysisManager.h"
#include "mlir/Support/LLVM.h"
#include "mlir/IR/Builders.h"

#include "toucan/PartitioningGraph.h"
#include "toucan/ToucanAnalysis.h"
#include "toucan/ToucanAttributes.h"
#include "toucan/ToucanOps.h"
#include "toucan/ToucanTypes.h"
#include "toucan/ToucanUtils.h"

#include "llvm/ADT/BitVector.h"
#include "llvm/ADT/STLExtras.h"
#include "llvm/ADT/SmallVector.h"
#include "llvm/Support/Casting.h"
#include "llvm/Support/Debug.h"
#include "llvm/Support/ErrorHandling.h"
#include "llvm/Support/raw_ostream.h"
#include <cstddef>
#include <cstdint>

#include <boost/graph/topological_sort.hpp>
#include <cstring>
#include <iterator>
#include <array>
#include <optional>
#include <tuple>
#include <utility>
#include <vector>
#include <numeric>
#include <algorithm>

using namespace toucan;

using namespace mlir;
using namespace llvm;
using namespace circt;



// Done!
void SingleRegionMicroPartScheduler::sortRegistersForLocality(const PartitioningGraph &graph, const mlir::SmallVector<mlir::SmallVector<uint32_t>> &partNodeList,  mlir::SmallVector<mlir::SmallVector<mlir::TypedValue<toucan::RegType>>> &regOrdered) {

  regOrdered.clear();

  mlir::SmallVector<mlir::SmallVector<uint32_t>> partToRegReads;
  mlir::SmallVector<mlir::SmallVector<uint32_t>> partToRegWrites;
  // Collect all reg reads
  for (size_t partId = 0; partId < partNodeList.size(); partId++) {
    partToRegReads.emplace_back();
    partToRegWrites.emplace_back();

    for (auto eachVtx: partNodeList[partId]) {
      auto tOpName = graph[eachVtx].toucanOpName;
      // auto rawOp = graph[eachVtx].op;

      if (tOpName == CGToucanOPName::RegRead) {
        partToRegReads.back().push_back(eachVtx);
      } else if (tOpName == CGToucanOPName::RegWrite) {
        partToRegWrites.back().push_back(eachVtx);
      }
    }
  }

  // Here we assume replication rate is relatively small
  mlir::DenseSet<mlir::TypedValue<toucan::RegType>> regValsWithMultipleReads;
  mlir::DenseMap<mlir::TypedValue<toucan::RegType>, uint32_t> regValToReaderPartId, regValToWriterPartId;
  mlir::DenseSet<mlir::TypedValue<toucan::RegType>> regValRead, regValWrite;
  mlir::DenseMap<mlir::TypedValue<toucan::RegType>, mlir::Value> regValToWriterDataVal;

  // Collect reg RW info
  for (size_t partId = 0; partId < partToRegReads.size(); partId++) {
    for (auto vtxId: partToRegReads[partId]) {
      auto regReadOp = cast<toucan::RegReadOp>(graph[vtxId].op);
      auto regVal = regReadOp.getReg();
      regValRead.insert(regVal);

      if (!regValsWithMultipleReads.contains(regVal)) {
        if (regValToReaderPartId.contains(regVal)) {
          // has at least 2 writer
          regValsWithMultipleReads.insert(regVal);
          regValToReaderPartId.erase(regVal);
        } else {
          regValToReaderPartId[regVal] = partId;
        }
      }
    }
  }
  for (size_t partId = 0; partId < partToRegWrites.size(); partId++) {
    for (auto vtxId: partToRegWrites[partId]) {
      auto regWriteOp = cast<toucan::RegWriteOp>(graph[vtxId].op);
      auto regVal = regWriteOp.getReg();
      assert(!regValWrite.contains(regVal) && "Each reg should have only 1 writer");
      regValWrite.insert(regVal);
      regValToWriterPartId[regVal] = partId;

      auto dataVal = regWriteOp.getData();
      regValToWriterDataVal[regVal] = dataVal;
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

  auto numReadParts = partToRegReads.size();
  auto numWriteParts = partToRegWrites.size();

  groupedSharedReadVals.resize(numWriteParts);
  groupedReadOnceVals.resize(numWriteParts);
  groupedWriteOnlyVals.resize(numWriteParts);


  // Segment by writer, then reader
  // Sort by reader: shared read, read by p0, p1, p2, ...

  // First, group by writer
  for (size_t writerPartId = 0; writerPartId < partToRegWrites.size(); writerPartId++) {
    // segment by reader
    mlir::SmallVector<mlir::SmallVector<mlir::TypedValue<toucan::RegType>>> currentSectionReadOnceValSegments;
    currentSectionReadOnceValSegments.resize(numReadParts);

    for (auto vtxId: partToRegWrites[writerPartId]) {
      auto regWriteOp = cast<toucan::RegWriteOp>(graph[vtxId].op);
      auto regVal = regWriteOp.getReg();

      // for each writer section
      if (regValsWithMultipleReads.contains(regVal)) {
        // reg with multiple reader
        groupedSharedReadVals[writerPartId].push_back(regVal);
      } else if (regValToReaderPartId.contains(regVal)) {
        // reg with 1 reader
        auto readerId = regValToReaderPartId[regVal];
        currentSectionReadOnceValSegments[readerId].push_back(regVal);
      } else {
        // write only
        groupedWriteOnlyVals[writerPartId].push_back(regVal);
      }
    }
  
    // Second, within each writer group, further sort by reader

    // readOnceVals: multiple vals might share same source.
    // segment them to ensure each segment has no reg vals with same source
    for (auto &eachReaderSegment: currentSectionReadOnceValSegments) {
      mlir::SmallVector<mlir::TypedValue<toucan::RegType>> orderedRegVals;
      mlir::DenseSet<mlir::TypedValue<toucan::RegType>> scheduledRegVals;
      mlir::DenseSet<mlir::Value> scheduledRegValDataSource;

      while (scheduledRegVals.size() != eachReaderSegment.size()) {
        for (auto &eachRegVal: eachReaderSegment) {
          if (!scheduledRegVals.contains(eachRegVal)) {
            auto dataVal = regValToWriterDataVal[eachRegVal];
            if (!scheduledRegValDataSource.contains(dataVal)) {
              // schedule it
              orderedRegVals.push_back(eachRegVal);
              scheduledRegValDataSource.insert(dataVal);
              scheduledRegVals.insert(eachRegVal);
            }
          }
        }
        scheduledRegValDataSource.clear();
      }
      std::swap(eachReaderSegment, orderedRegVals);
    }

    
    for (auto & eachReaderSegment: currentSectionReadOnceValSegments) {
      std::copy(eachReaderSegment.begin(), eachReaderSegment.end(), 
        std::back_inserter(groupedReadOnceVals[writerPartId]));
    }

    // TODO: sort sharedVals by what?

  }


  // Special handling for read-only vals
  mlir::SmallVector<mlir::TypedValue<toucan::RegType>> readOnlySharedVals, readOnlyOnceVals;

  for (auto &eachReadVal: regValRead) {
    if (!regValToWriterPartId.contains(eachReadVal)) {
      // read only, no writer
      if (regValsWithMultipleReads.contains(eachReadVal)) {
        readOnlySharedVals.push_back(eachReadVal);
      } else {
        assert(regValToReaderPartId.contains(eachReadVal));
        readOnlyOnceVals.push_back(eachReadVal);
      }
    }
  }
  // TODO: sort readOnlySharedVals by what? They are relatively rare
  std::sort(readOnlyOnceVals.begin(), readOnlyOnceVals.end(), 
  [&] (const mlir::TypedValue<toucan::RegType>& a, const mlir::TypedValue<toucan::RegType>& b) {
    auto readerPartId_a = regValToReaderPartId[a];
    auto readerPartId_b = regValToReaderPartId[b];
    return readerPartId_a < readerPartId_b;
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

// Update allRegWrites in each MicroPartitioner
void SingleRegionMicroPartScheduler::sortRegWriteOps(const PartitioningGraph &graph, mlir::SmallVector<uint32_t> &allRegWrites) const {

  mlir::DenseMap<uint32_t, uint32_t> vtxIdToOrder;
  for (const auto &eachRegWrite: allRegWrites) {
    auto regWriteOp = cast<toucan::RegWriteOp>(graph[eachRegWrite].op);
    auto regVal = regWriteOp.getReg();
    assert(codeGenInfo.toucanRegToId.contains(regVal) && "Every register should appear in this map!");

    assert(!vtxIdToOrder.contains(eachRegWrite && "Cannot write to one register multiple times!"));
    vtxIdToOrder[eachRegWrite] = codeGenInfo.toucanRegToId.at(regVal);
  }

  // sort
  std::stable_sort(allRegWrites.begin(), allRegWrites.end(), [&](uint32_t a, uint32_t b) {
    assert(vtxIdToOrder.contains(a));
    assert(vtxIdToOrder.contains(b));
    auto a_order = vtxIdToOrder[a];
    auto b_order = vtxIdToOrder[b];
    return a_order < b_order;
  });
}

void SingleRegionMicroPartScheduler::sortRegReadOps(const PartitioningGraph &graph, mlir::SmallVector<uint32_t> &allRegReads) const {
  mlir::DenseMap<uint32_t, uint32_t> vtxIdToOrder;
  for (const auto &eachRegRead: allRegReads) {
    auto regReadOp = cast<toucan::RegReadOp>(graph[eachRegRead].op);
    auto regVal = regReadOp.getReg();
    assert(codeGenInfo.toucanRegToId.contains(regVal) && "Every register should appear in this map!");
    assert(!vtxIdToOrder.contains(eachRegRead && "Cannot read from one register multiple times!"));
    vtxIdToOrder[eachRegRead] = codeGenInfo.toucanRegToId.at(regVal);
  }

  // sort
  std::stable_sort(allRegReads.begin(), allRegReads.end(), [&](uint32_t a, uint32_t b) {
    assert(vtxIdToOrder.contains(a));
    assert(vtxIdToOrder.contains(b));
    auto a_order = vtxIdToOrder[a];
    auto b_order = vtxIdToOrder[b];
    return a_order < b_order;
  });
}


void SingleRegionMicroPartScheduler::fillRegPool(mlir::SmallVector<mlir::SmallVector<mlir::TypedValue<toucan::RegType>>> regPoolOrdered) {
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

void SingleRegionMicroPartScheduler::fillMemPool(const PartitioningGraph &graph) {
  // Here we assume each memory has at least 1 writer
  uint64_t memBaseAddr = 0;
  for (auto vtxId : boost::make_iterator_range(vertices(graph))) {
    // for each mem write
    auto vtxOpName = graph[vtxId].toucanOpName;
    if (vtxOpName == CGToucanOPName::MemWrite) {
      // Note: For now, mems with multiple write ports are still merged, so at this time, each memory will only have 1 writer.
      auto memWriteOp = cast<toucan::MemWriteOp>(graph[vtxId].op);
      auto memVal = memWriteOp.getMem();
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
      memMeta.hasMultipleWriter = (graph[vtxId].opCount > 1);

      // if a memory has multiple writer, add extra padding to avoid possible write conflict
      assert(memMeta.bitWidth <= 4);
      uint64_t memCapacity = (memMeta.hasMultipleWriter) ? memMeta.memDepth * multiWriterMemElemBytes : memMeta.memDepth;
      memMeta.memBase = memBaseAddr;
      memBaseAddr += (memCapacity + memPaddingSpace);

      auto memId = codeGenInfo.memPool.size();
      codeGenInfo.memPool.push_back(memMeta);
      codeGenInfo.toucanMemToId[memVal] = memId;
    }
  }
  codeGenInfo.totalMemSize = memBaseAddr;

  // Double check
  for (auto vtxId : boost::make_iterator_range(vertices(graph))) {
    // for each mem write
    auto vtxOpName = graph[vtxId].toucanOpName;
    if (vtxOpName == CGToucanOPName::MemRead) {
      // Note: For now, mems with multiple write ports are still merged, so at this time, each memory will only have 1 writer.
      auto memReadOp = cast<toucan::MemReadOp>(graph[vtxId].op);
      auto memVal = memReadOp.getMem();

      assert(codeGenInfo.toucanMemToId.contains(memVal));
    }
  }
}

void SingleRegionMicroPartScheduler::generateRegMemLayout(const PartitioningGraph &graph, const mlir::SmallVector<mlir::SmallVector<uint32_t>> &partNodeLis) {
  // collect all reg and memory, generate layout
  codeGenInfo.regPool.clear();
  codeGenInfo.memPool.clear();


  // 1. sort registers
  // Writer part -> val. Needs padding
  mlir::SmallVector<mlir::SmallVector<mlir::TypedValue<toucan::RegType>>> regPoolOrdered;

  // llvm::dbgs() << "Sorting registers\n";
  sortRegistersForLocality(graph, partNodeLis, regPoolOrdered);

  // Now, registers and exchangeVal location and ops are sorted
  fillRegPool(regPoolOrdered);

  // Coaleasce register access.
  for (size_t partId = 0; partId < mpartitioners.size(); partId++) {
    sortRegWriteOps(graph, mpartitioners[partId].allRegWrites);
    sortRegReadOps(graph, mpartitioners[partId].allRegReads);
  }

  
  // Allocate storage for all memories
  fillMemPool(graph);

  return;
}





// Collect const decls. DOES NOT collect const vec decls
// Const vars are shared
void SingleRegionMicroPartScheduler::collectConstantVars(const PartitioningGraph &graph) {
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
void SingleRegionMicroPartScheduler::collectConstantVecs(const PartitioningGraph &graph, CGPartitionMetaInfo &partInfo, const mlir::SmallVector<uint32_t> &partNodes) {
  // Collect all const vecs, populate constVecPool
  mlir::SmallVector<toucan::DefConstVectorOp> constVecDeclOps;

  for (const auto vtxId: partNodes) {
    auto vtxOpName = graph[vtxId].toucanOpName;
    if (vtxOpName == CGToucanOPName::VecRead) {
      auto op = cast<toucan::VectorReadOp>(graph[vtxId].op);
      auto vecHandle = op.getHandle();
      auto vecDeclOp = vecHandle.getDefiningOp();

      if (auto defConstVecOp = dyn_cast<toucan::DefConstVectorOp>(vecDeclOp)) {
        constVecDeclOps.push_back(defConstVecOp);
      }
    } else if (vtxOpName == CGToucanOPName::VecArith) {
      auto op = cast<toucan::VectorArithOp>(graph[vtxId].op);
      auto vec1 = op.getV1();
      auto vec2 = op.getV2();

      if (auto vec1ConstOp = dyn_cast<toucan::DefConstVectorOp>(vec1.getDefiningOp())) {
        constVecDeclOps.push_back(vec1ConstOp);
      }
      if (auto vec2ConstOp = dyn_cast<toucan::DefConstVectorOp>(vec2.getDefiningOp())) {
        constVecDeclOps.push_back(vec2ConstOp);
      }
    } else if (vtxOpName == CGToucanOPName::VecLogic) {
      auto op = cast<toucan::VectorLogicOp>(graph[vtxId].op);
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

  // Dedup vector from 2 ~ 16 sections;
  #define ConstVecDedupMaxLength 16
  mlir::SmallVector<mlir::DenseMap<uint64_t, size_t>> smallVecDedupTable;
  smallVecDedupTable.resize(ConstVecDedupMaxLength);
  size_t constVecDedupCount = 0;

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
    uint64_t rawVecVal = 0;
    auto dedupTableIndex = vecLength - 1;
    // find if we have same vector placed
    if (bitWidth == 4 && dedupTableIndex < smallVecDedupTable.size()) {
      for (auto &vecValElem: llvm::reverse(defConstVecOp.getValues().getValue())) {
        rawVecVal = rawVecVal << 4;
        auto elemVal = cast<mlir::IntegerAttr>(vecValElem).getValue();
        auto elemValWidth = elemVal.getBitWidth();
        assert(elemValWidth <= 4);
  
        auto elemValMask = static_cast<uint8_t>((1 << elemValWidth) - 1);
        uint8_t rawVal = elemValMask & static_cast<uint8_t>(elemVal.getZExtValue());
        rawVecVal = rawVecVal | rawVal;
      }

      assert(smallVecDedupTable.size() > dedupTableIndex);
      if (smallVecDedupTable[dedupTableIndex].contains(rawVecVal)) {
        canDedup = true;
        dedupValId = smallVecDedupTable[dedupTableIndex][rawVecVal];
        constVecDedupCount += 1;
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

      if (bitWidth == 4 && dedupTableIndex < smallVecDedupTable.size()) {
        assert(!smallVecDedupTable[dedupTableIndex].contains(rawVecVal));
        smallVecDedupTable[dedupTableIndex][rawVecVal] = valId;
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

  llvm::dbgs() << constVecDedupCount << " small const vecs deduplicated\n";
}


void SingleRegionMicroPartScheduler::scheduleRegReads(const PartitioningGraph &graph, CGPartitionMetaInfo &partInfo, const mlir::SmallVector<uint32_t> &allRegReads) {
  mlir::SmallVector<CGOpMetaInfo> currentLevelOps;
  currentLevelOps.reserve(allRegReads.size());

  for (auto vtxId: allRegReads) {
    auto tOpName = graph[vtxId].toucanOpName;

    auto op = graph[vtxId].op;
    CGOpMetaInfo opMeta;
    opMeta.opName = tOpName;
    opMeta.op = op;
    opMeta.vtxId = vtxId;
    populateOpMetaDebugInfo(opMeta, op);

    if (tOpName == CGToucanOPName::RegRead) {
      // a regread
      auto regReadOp = cast<toucan::RegReadOp>(op);
      auto regVal = regReadOp.getReg();
      assert(codeGenInfo.toucanRegToId.contains(regVal) && "A register that never seen was read!");
      auto regValId = codeGenInfo.toucanRegToId[regVal];

      assert(partInfo.valueToValId.contains(regReadOp.getResult()));

      opMeta.regRead.reg = regValId;
      opMeta.regRead.result = partInfo.valueToValId[regReadOp.getResult()];
      currentLevelOps.push_back(opMeta);

      
    } else {
      llvm_unreachable("Should not reach here");
    }
  }

  // Save ops
  std::swap(partInfo.regReadOps, currentLevelOps);
}


void SingleRegionMicroPartScheduler::scheduleRegWrites(const PartitioningGraph &graph, CGPartitionMetaInfo &partInfo, const mlir::SmallVector<uint32_t> &allRegWrites) {

  // Code gen for last level

  mlir::SmallVector<CGOpMetaInfo> regWriteOps;

  for (auto vtxId: allRegWrites) {
    auto vtxOpName = graph[vtxId].toucanOpName;
    auto rawOp = graph[vtxId].op;

    CGOpMetaInfo opMeta;
    opMeta.op = rawOp;
    opMeta.opName = vtxOpName;
    opMeta.vtxId = vtxId;

    if (vtxOpName == CGToucanOPName::RegWrite) {
      // regwrite

      auto regWriteOp = cast<toucan::RegWriteOp>(rawOp);
      auto regVal = regWriteOp.getReg();
      auto dataVal = regWriteOp.getData();

      assert(codeGenInfo.toucanRegToId.contains(regVal) && "A register never seen was written!");
      auto regValId = codeGenInfo.toucanRegToId[regVal];
      assert(partInfo.valueToValId.contains(dataVal) && "A value never seen was read!");
      auto dataValId = partInfo.valueToValId[dataVal];

      opMeta.regWrite.reg = regValId;
      opMeta.regWrite.dat = dataValId;

      // Namehint not needed for last level ops
      regWriteOps.push_back(opMeta);
    } else {
      llvm_unreachable("Other type of ops should not appear in last level");
    }
  }

  std::swap(partInfo.regWriteOps, regWriteOps);
}

void SingleRegionMicroPartScheduler::scheduleMemWrites(const PartitioningGraph &graph, CGPartitionMetaInfo &partInfo, const mlir::SmallVector<uint32_t> &allMemWrites) {

  mlir::SmallVector<CGOpMetaInfo> memWriteOps;

  for (auto vtxId: allMemWrites) {
    auto vtxOpName = graph[vtxId].toucanOpName;
    auto rawOp = graph[vtxId].op;

    CGOpMetaInfo opMeta;
    opMeta.op = rawOp;
    opMeta.opName = vtxOpName;
    opMeta.vtxId = vtxId;

    if (vtxOpName == CGToucanOPName::MemWrite) {
      // memWrite
      // Note: if there is multiple writers, they are all merged into 1 vtx
      // Split
      auto memVal = cast<toucan::MemWriteOp>(rawOp).getMem();
      assert(codeGenInfo.toucanMemToId.contains(memVal));
      auto memValId = codeGenInfo.toucanMemToId[memVal];

      for (auto userOp: memVal.getUsers()) {
        if (auto memWriteOp = dyn_cast<toucan::MemWriteOp>(userOp)) {
          // for each writer

          CGOpMetaInfo mwOpMeta;
          mwOpMeta.op = rawOp;
          mwOpMeta.opName = vtxOpName;
          mwOpMeta.vtxId = vtxId;

          auto dataVal = memWriteOp.getData();
          auto enVal = memWriteOp.getEn();

          assert(partInfo.valueToValId.contains(dataVal));
          auto dataValId = partInfo.valueToValId[dataVal];
          assert(partInfo.valueToValId.contains(enVal));
          auto enValId = partInfo.valueToValId[enVal];

          auto memAddrVec = memWriteOp.getAddrVec();
          auto memAddrVecId = partInfo.valueToValId[memAddrVec];
          assert(memAddrVec.getType().getLength() == 8 && "Memory address should be 32 bit vector");

          mwOpMeta.memWrite.hasMultipleWriter = codeGenInfo.memPool[memValId].hasMultipleWriter;
          mwOpMeta.memWrite.memBase = codeGenInfo.memPool[memValId].memBase;
          mwOpMeta.memWrite.memDepth = codeGenInfo.memPool[memValId].memDepth;

          mwOpMeta.memWrite.addrVec = memAddrVecId;

          mwOpMeta.memWrite.dat = dataValId;
          mwOpMeta.memWrite.en = enValId;

          // Namehint not needed for last level ops
          memWriteOps.push_back(mwOpMeta);
        }
      }
    } else {
      llvm_unreachable("Other type of ops should not appear in last level");
    }
  }

  std::swap(partInfo.memWriteOps, memWriteOps);
}

void SingleRegionMicroPartScheduler::scheduleStops(const PartitioningGraph &graph, CGPartitionMetaInfo &partInfo, const mlir::SmallVector<uint32_t> &allStops) {

  mlir::SmallVector<CGOpMetaInfo> stopOps;

  for (auto vtxId: allStops) {
    auto vtxOpName = graph[vtxId].toucanOpName;
    auto rawOp = graph[vtxId].op;

    CGOpMetaInfo opMeta;
    opMeta.op = rawOp;
    opMeta.opName = vtxOpName;
    opMeta.vtxId = vtxId;

    if (vtxOpName == CGToucanOPName::Stop) {
      // stop

      auto stopOp = cast<toucan::StopOp>(rawOp);
      auto enVal = stopOp.getEn();

      assert(partInfo.valueToValId.contains(enVal));
      auto enValId = partInfo.valueToValId[enVal];

      opMeta.stop.en = enValId;

      // Namehint not needed for last level ops
      stopOps.push_back(opMeta);

    } else {
      llvm_unreachable("Other type of ops should not appear in last level");
    }
  }

  std::swap(partInfo.stopOps, stopOps);
}


void SingleRegionMicroPartScheduler::schedulePrints(const PartitioningGraph &graph, CGPartitionMetaInfo &partInfo, const mlir::SmallVector<uint32_t> &allPrints) {

  mlir::SmallVector<CGOpMetaInfo> printOps;

  for (auto vtxId: allPrints) {
    auto vtxOpName = graph[vtxId].toucanOpName;
    auto rawOp = graph[vtxId].op;

    CGOpMetaInfo opMeta;
    opMeta.op = rawOp;
    opMeta.opName = vtxOpName;
    opMeta.vtxId = vtxId;


    if (vtxOpName == CGToucanOPName::Print) {
      // print

      auto printOp = cast<toucan::PrintOp>(rawOp);
      auto printStr = printOp.getMsg();
      auto enVal = printOp.getEn();

      assert(partInfo.valueToValId.contains(enVal));
      auto enValId = partInfo.valueToValId[enVal];
      
      assert(codeGenInfo.printStrings.contains(printStr));
      auto printStrId = codeGenInfo.printStrings[printStr];

      opMeta.print.en = enValId;
      opMeta.print.msg = printStrId;

      // Namehint not needed for last level ops
      printOps.push_back(opMeta);

    } else {
      llvm_unreachable("Other type of ops should not appear in last level");
    }
  }

  std::swap(partInfo.printOps, printOps);
}




// function: extract value life time; fill dummy op; fill normal lut

struct ValLifeCycle {
  uint32_t start, end;
};

static void extractMicroPartValueLifeTime(const PartitioningGraph &graph, const MicroPart &mPart, mlir::DenseMap<mlir::Value, ValLifeCycle> &valueToLifeCycle) {

  auto writeBackLevel = static_cast<uint32_t>(mPart.levels.size());
  assert(mPart.nodeToInputVals.size() != 0);
  for (const auto [vtx, outputVal]: mPart.nodeToOutputVal) {
    assert(!valueToLifeCycle.contains(outputVal));
    auto vtxLevel = mPart.nodeToLevel.at(vtx);
    assert(vtxLevel < writeBackLevel);
    valueToLifeCycle[outputVal] = {vtxLevel, vtxLevel};
  }
  for (const auto &[vtx, inputVals]: mPart.nodeToInputVals) {
    auto vtxLevel = mPart.nodeToLevel.at(vtx);

    for (const auto &eachVal: inputVals) {
      if (!valueToLifeCycle.contains(eachVal)) {
        // an external input value that is not used by first level
        assert(mPart.inputValues.contains(eachVal));
        valueToLifeCycle[eachVal] = {0, vtxLevel};
      } else {
        // max
        auto oldEndTime = valueToLifeCycle[eachVal].end;
        if (oldEndTime < vtxLevel) {
          valueToLifeCycle[eachVal].end = vtxLevel;
        }
      }
    }
  }
  for (const auto &eachOutputVal: mPart.outputValues) {
    assert(valueToLifeCycle[eachOutputVal].end == writeBackLevel);
  }
}


static mlir::DenseMap<uint32_t, uint32_t> buildDummyVtxIndexInVec(const MicroPartitioner mPartitioner) {
  mlir::DenseMap<uint32_t, uint32_t> ret;
  assert(mPartitioner.outputVectorNopMap.size() == mPartitioner.originalVectorElementsMap.size());

  for (auto &[vecId, _]: mPartitioner.outputVectorNopMap) {
    assert(mPartitioner.originalVectorElementsMap.contains(vecId));
    auto vecNumElements = mPartitioner.originalVectorElementsMap.at(vecId).size();
    assert(mPartitioner.outputVectorNopMap.at(vecId).size() == vecNumElements);
    for (size_t i = 0; i < vecNumElements; i++) {
      auto newNodeId = mPartitioner.outputVectorNopMap.at(vecId)[i];
      // auto oldNodeId = mPartitioner.originalVectorElementsMap.at(vecId)[i];

      assert(!ret.contains(newNodeId));
      ret[newNodeId] = static_cast<uint32_t>(i);
    }
  }

  return ret;
}


static void scheduleRegularMicroPart(const PartitioningGraph &graph, CGMicroPartInfo &part, const MicroPart &mPart, const mlir::DenseMap<mlir::Value, uint32_t> &valToValId, const MicroPartitioner mPartitioner, const mlir::DenseMap<uint32_t, uint32_t> &dummyVtxIndexInVecTable) {
  assert(mPart.opType == CGToucanOPName::LUT);

  // auto findDummyVtxIndexInVec = [&](uint32_t vtx) {
  //   assert(mPartitioner.newNodeIdToOriginalVecDeclId.contains(vtx));
  //   auto vecId = mPartitioner.newNodeIdToOriginalVecDeclId.at(vtx);
  //   assert(mPartitioner.outputVectorNopMap.contains(vecId));
  //   assert(mPartitioner.originalVectorElementsMap.contains(vecId));
  //   auto vecNumElements = mPartitioner.originalVectorElementsMap.at(vecId).size();
  //   assert(mPartitioner.outputVectorNopMap.at(vecId).size() == vecNumElements);

  //   for (size_t i = 0; i < vecNumElements; i++) {
  //     auto newNodeId = mPartitioner.outputVectorNopMap.at(vecId)[i];
  //     auto oldNodeId = mPartitioner.originalVectorElementsMap.at(vecId)[i];

  //     if (newNodeId == vtx) return std::tuple<uint32_t, uint32_t>(static_cast<uint32_t>(i), oldNodeId);
  //   }
  //   llvm_unreachable("This procedure should always success");
  // };

  part.clear();
  part.opType = CGToucanOPName::LUT;

  
  mlir::DenseMap<mlir::Value, ValLifeCycle> valueToLifeCycle;
  extractMicroPartValueLifeTime(graph, mPart, valueToLifeCycle);
  auto MaxLiveLevel = static_cast<uint32_t>(mPart.levels.size());



  mlir::DenseMap<mlir::Value, uint8_t> shuffleValueToId, shuffleValueToId_next;
  // mlir::DenseSet<mlir::Value> passThroughValuesThisLevel;

  

  // Values that need to be written back to SMem at last level
  struct PendingWriteBack {
    mlir::Value val;
    uint16_t sMemLocation;
  };
  mlir::SmallVector<PendingWriteBack> pendingWriteBackValueAndLocation;

  {
    // first level
    uint8_t opIndex = 0;
    mlir::SmallVector<uint16_t> topLevelOpInputValIndecies;


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
        topLevelOpInputValIndecies[pos] = 0;
      }
      for (auto val: lutOp.getInputs()) {
        assert(valToValId.contains(val));
        auto valId = valToValId.at(val);
        assert(valId < UINT16_MAX);
        topLevelOpInputValIndecies[pos] = valId;
        pos++;
      }
      assert(topLevelOpInputValIndecies.size() == 3);
    };

    for (auto eachVtx: mPart.levels[0]) {
      auto vtxIsDummyNop = mPartitioner.newNodeIdToDepNodeId.contains(eachVtx);

      if (vtxIsDummyNop) {
        // vecdecl
        auto depNode = mPartitioner.newNodeIdToDepNodeId.at(eachVtx);
        auto depValue = getOpResultValue(graph[depNode].op);
        // This value need to be passed through entire micro part
        assert(valueToLifeCycle.contains(depValue));
        assert(valueToLifeCycle[depValue].start == 0);
        assert(valueToLifeCycle[depValue].end == MaxLiveLevel);

        auto depValueIdInSMem = valToValId.at(depValue);
        assert(depValueIdInSMem < UINT16_MAX);

        auto vecVtx = mPartitioner.newNodeIdToOriginalVecDeclId.at(eachVtx);
        auto vecVal = cast<toucan::DefVectorOp>(graph[vecVtx].op).getHandle();

        auto indexInVec = dummyVtxIndexInVecTable.at(eachVtx);
        
        auto vecValIdInSMem = valToValId.at(vecVal);
        auto resultIndex = vecValIdInSMem + indexInVec;
        assert(resultIndex < UINT16_MAX);

        // this value should be written to smem at last level!
        pendingWriteBackValueAndLocation.push_back({depValue, static_cast<uint16_t>(resultIndex)});

        // also create a NOP
        part.topLevel.push_back({
          LUTOpName::LUT_Nop, 
          0, 0, static_cast<uint16_t>(depValueIdInSMem)});

        assert(!shuffleValueToId.contains(depValue));
        auto resultValShuffleId = static_cast<uint8_t>(opIndex + 32);
        shuffleValueToId[depValue] = resultValShuffleId;

        assert(opIndex <= 32);
        opIndex++;
        assert(opIndex == part.topLevel.size());
      } else {
        // regular lut op
        auto rawOp = graph[eachVtx].op;
        auto lutOp = cast<toucan::LUTOp>(rawOp);

        getLUTOpInputIds(rawOp);

        // get output id
        auto resultVal = lutOp.getResult();
        assert(valToValId.contains(resultVal));
        auto resultValId = valToValId.at(resultVal);
        assert(resultValId < UINT16_MAX);

        assert(valueToLifeCycle.contains(resultVal));
        assert(valueToLifeCycle[resultVal].start == 0);

        part.topLevel.push_back({
          lutOp.getOpName(), 
          topLevelOpInputValIndecies[0], 
          topLevelOpInputValIndecies[1], 
          topLevelOpInputValIndecies[2]});

        assert(!shuffleValueToId.contains(resultVal));
        auto resultValShuffleId = static_cast<uint8_t>(opIndex + 32);
        shuffleValueToId[resultVal] = resultValShuffleId;

        assert(opIndex <= 32);
        opIndex++;
        assert(opIndex == part.topLevel.size());
      }
    }

    // insert NOP for partition reads
    for (auto &eachInputVal: mPart.inputValues) {
      if (!shuffleValueToId.contains(eachInputVal)) {
        // insert a dummy read op
        assert(valToValId.contains(eachInputVal));
        assert(valueToLifeCycle.contains(eachInputVal));
        assert(valueToLifeCycle[eachInputVal].start == 0);

        auto inputValIdInSmem = valToValId.at(eachInputVal);
        assert(inputValIdInSmem < UINT16_MAX);

        part.topLevel.push_back({
          LUTOpName::LUT_Nop,
          0, 0, static_cast<uint16_t>(inputValIdInSmem)
        });

        assert(!shuffleValueToId.contains(eachInputVal));
        auto resultValShuffleId = static_cast<uint8_t>(opIndex + 32);
        shuffleValueToId[eachInputVal] = resultValShuffleId;

        // passThroughValuesThisLevel.insert(eachInputVal);

        assert(opIndex <= 32);
        opIndex++;
        assert(opIndex == part.topLevel.size());
      }
    }
  }

  // remaining levels
  // mlir::DenseSet<mlir::Value> valuesOutputOfThisLevel;
  // mlir::DenseSet<mlir::Value> valuesUsedByThisLevel;

  mlir::SmallVector<uint8_t> lutOpInputValIndecies;

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
      lutOpInputValIndecies[pos] = 0;
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
        lutOpInputValIndecies[pos] = rawVal;
      } else {
        // a regular value. Must be in shuffle values
        assert(shuffleValueToId.contains(val));
        auto valId = shuffleValueToId.at(val);
        lutOpInputValIndecies[pos] = valId;
      }
      pos++;
    }
    assert(lutOpInputValIndecies.size() == 3);
  };

  
  for (size_t levelId = 1; levelId < mPart.levels.size(); levelId++) {

    uint8_t opIndex = 0;
    part.middleLevels.emplace_back();
    shuffleValueToId_next.clear();


    for (const auto eachVtx: mPart.levels[levelId]) {
      auto vtxIsDummyNop = mPartitioner.newNodeIdToDepNodeId.contains(eachVtx);

      if (vtxIsDummyNop) {
        // vecdecl
        auto depNode = mPartitioner.newNodeIdToDepNodeId.at(eachVtx);
        auto depValue = getOpResultValue(graph[depNode].op);
        // This value need to be passed through entire micro part
        // ?
        assert(valueToLifeCycle.contains(depValue));
        assert(valueToLifeCycle[depValue].start < levelId);
        assert(valueToLifeCycle[depValue].end == MaxLiveLevel);

        // The input value must in somewhere in the shuffle network!
        assert(shuffleValueToId.contains(depValue));

        auto vecVtx = mPartitioner.newNodeIdToOriginalVecDeclId.at(eachVtx);
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

          assert(opIndex <= 32);
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

        // if (!shuffleValueToId_next.contains(resultVal)) {
        //   auto resultValShuffleId = static_cast<uint8_t>(opIndex + 32);
        //   shuffleValueToId_next[resultVal] = resultValShuffleId;

        //   assert(opIndex <= 32);
        //   opIndex++;
        //   assert(opIndex == part.middleLevels.back().size());
        // }
        assert(!shuffleValueToId_next.contains(resultVal));
        auto resultValShuffleId = static_cast<uint8_t>(opIndex + 32);
        shuffleValueToId_next[resultVal] = resultValShuffleId;
        assert(opIndex <= 32);
        opIndex++;
        assert(opIndex == part.middleLevels.back().size());

        // Result of a lut must be used somewhere, or just in output
        assert(valueToLifeCycle.contains(resultVal));
        assert(valueToLifeCycle[resultVal].start == levelId);
        // or an output value
        if (valueToLifeCycle[resultVal].end == MaxLiveLevel) {
          assert(mPart.outputValues.contains(resultVal));
          assert(valToValId.contains(resultVal));
          
          auto resultValIdInSMem = valToValId.at(resultVal);
          pendingWriteBackValueAndLocation.push_back({resultVal, static_cast<uint16_t>(resultValIdInSMem)});
        }
      }

    }

    // Create NOP for pass through values
    for (const auto &[val, lifeTime]: valueToLifeCycle) {
      assert(lifeTime.end > lifeTime.start);

      // future value, ignore
      if (lifeTime.start > levelId) continue;
      if (lifeTime.start == levelId) {
        // value created by this level
        // valuesOutputOfThisLevel.insert(val);
      } else {
        // value created by previous level
        if (lifeTime.end > levelId) {
          // pass through
          assert(!isa<mlir::TypedValue<toucan::VecType>>(val));
          assert(shuffleValueToId.contains(val));
          assert(!shuffleValueToId_next.contains(val));


          // insert a dummy NOP op
          auto valShuffleId = shuffleValueToId[val];

          part.middleLevels.back().push_back({
            LUTOpName::LUT_Nop, 
            0, 0, valShuffleId});

          // get output id
          auto resultValShuffleId = static_cast<uint8_t>(opIndex + 32);
          shuffleValueToId_next[val] = resultValShuffleId;
          assert(opIndex <= 32);
          opIndex++;
          assert(opIndex == part.middleLevels.back().size());

        }
      }
    }

    std::swap(shuffleValueToId, shuffleValueToId_next);

    for (auto &[val, _]: pendingWriteBackValueAndLocation) {
      if (valueToLifeCycle[val].start <= levelId) {
        assert(shuffleValueToId.contains(val));
      }
    }
  }


  // Create write back NOPs
  assert(pendingWriteBackValueAndLocation.size() <= 32);
  for (const auto &[val, sMemIdx]: pendingWriteBackValueAndLocation) {
    assert(shuffleValueToId.contains(val));
    auto valShuffleId = shuffleValueToId[val];

    part.lastLevel.push_back({valShuffleId, sMemIdx});
  }



  // TODO: remove all NOP levels
}

static void scheduleSpecialMicroPart(const PartitioningGraph &graph, CGMicroPartInfo &part, const MicroPart &mPart, const CGInfo &codeGenInfo, const CGPartitionMetaInfo &partInfo) {
  const auto &valToValId = partInfo.valueToValId;
  const auto &toucanMemToId = codeGenInfo.toucanMemToId;
  const auto &memPool = codeGenInfo.memPool;

  assert(mPart.levels.size() == 1);
  assert(mPart.opType != CGToucanOPName::LUT);

  part.clear();
  part.opType = mPart.opType;

  uint8_t opIndex = 0;
  switch (mPart.opType) {
    case CGToucanOPName::VecRead: {
      for (auto vtx: mPart.levels[0]) {
        auto rawOp = graph[vtx].op;
        auto vecVal = cast<toucan::VectorReadOp>(rawOp).getHandle();

        assert(valToValId.contains(vecVal));
        auto vecValIdInSMem = valToValId.at(vecVal);
        assert(vecValIdInSMem < UINT16_MAX);

        auto vecLength = vecVal.getType().getLength();
        assert(vecLength < UINT16_MAX);

        bool isConstVec = isa<toucan::DefConstVectorOp>(vecVal.getDefiningOp());
        if (!isConstVec) assert(isa<toucan::DefVectorOp>(vecVal.getDefiningOp()));


        for (auto userOp: vecVal.getUsers()) {
          if (auto vecReadOp = dyn_cast<toucan::VectorReadOp>(userOp)) {

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

            assert(opIndex <= 32);
            opIndex++;
            assert(opIndex == part.vecRead.size());
          }
        }

      }
      break;
    }
    case CGToucanOPName::VecLogic:{
      for (auto vtx: mPart.levels[0]) {
        auto rawOp = graph[vtx].op;
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

        assert(opIndex <= 32);
        opIndex++;
        assert(opIndex == part.vecLogic.size());
      }
      break;
    }
    case CGToucanOPName::VecArith:{
      for (auto vtx: mPart.levels[0]) {
        auto rawOp = graph[vtx].op;
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

        assert(opIndex <= 32);
        opIndex++;
        assert(opIndex == part.vecArith.size());
      }
      break;
    }
    case CGToucanOPName::MemRead:{
      for (auto vtx: mPart.levels[0]) {
        auto rawOp = graph[vtx].op;
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

        assert(opIndex <= 32);
        opIndex++;
        assert(opIndex == part.memRead.size());
      }
      break;
    }

    default: llvm_unreachable("Unexpected op type");
  }
}


// Scheduler entry point
void SingleRegionMicroPartScheduler::schedule(const PartitioningGraph &graph, const mlir::SmallVector<mlir::SmallVector<uint32_t>> &partNodeList) {
  assert(mpartitioners.size() == partNodeList.size());


  // schedule all registers, memories and exchange values. Also sort registers and exchange writes
  generateRegMemLayout(graph, partNodeList);

  // dedup strings
  collectPrintString(graph, codeGenInfo.printStrings);
  if (codeGenInfo.printStrings.size() > UINT16_MAX) {
    llvm::errs() << "Too many print strings! (current max is " << UINT16_MAX << ")\n";
    llvm_unreachable("Too many print strings");
  }
  assert(codeGenInfo.printStrings.size() <= UINT16_MAX);

  collectConstantVars(graph);

  // schedule ops
  {
    codeGenInfo.regionPartitionIds.emplace_back();

    auto numPartitions = partNodeList.size();

    for (uint32_t partId = 0; partId < numPartitions; partId++) {

      CGPartitionMetaInfo partInfo;
      std::memset(&partInfo.opStatistics, 0, sizeof(CGOpStatistics));

      collectConstantVecs(graph, partInfo, partNodeList[partId]);

      auto &currentMicroPartitioner = mpartitioners[partId];

      // allocate space for values in shared mem
      MicroPartLocalValueAllocator valAllocator;
      valAllocator.collectValueLifetime(graph, currentMicroPartitioner);
      valAllocator.populateInitialPinnedVals(graph, constValToRawValue, currentMicroPartitioner);
      valAllocator.allocateLocalValues();
      // save const value pool
      std::swap(partInfo.constValuePool, valAllocator.compactConstValPool);
      partInfo.numConstsInValuePool = partInfo.constValuePool.size();
      // save value location
      for (auto [k, v]: valAllocator.valToValId) {
        assert(!partInfo.valueToValId.contains(k));
        partInfo.valueToValId[k] = v;
      }
      // ensure const vector is not handled. They are in const vec pool
      for (auto [k, _]: partInfo.valueToValId) {
        assert(!isa<toucan::DefConstVectorOp>(k.getDefiningOp()));
      }


      
      partInfo.numTotalValues = valAllocator.numTotalValSize;
      llvm::outs() 
        << "Part " << partId
        << " has " << currentMicroPartitioner.partLevels.size()
        << " levels, "
        << partInfo.valueToValId.size() 
        << " active values, allocator requires " << valAllocator.numTotalValSize 
        << " total bytes (" << valAllocator.numConsts 
        << " consts, " << valAllocator.numOutputVals 
        << " for output, " << valAllocator.numInputVals 
        << " for input). constVecPool size of " << partInfo.constVecPool.size() << "\n";

      scheduleRegReads(graph, partInfo, currentMicroPartitioner.allRegReads);

      // Save statistics
      {
        CGLayerValueStatistics stats;
        std::memset(&stats, 0, sizeof(CGLayerValueStatistics));
        stats.numRegReads = partInfo.regReadOps.size();
        partInfo.opStatisticsPerLevel.push_back(stats);
        partInfo.opStatistics.numRegReads = partInfo.regReadOps.size();
      }

      auto dummyVtxIndexInVecTable = buildDummyVtxIndexInVec(currentMicroPartitioner);
      for (uint32_t levelId = 0; levelId < currentMicroPartitioner.partLevels.size(); levelId++) {
        partInfo.microPartOps.emplace_back();

        for (auto &eachPart: currentMicroPartitioner.partLevels[levelId]) {
          // schedule each mpart
          bool isLUTPart = (eachPart.opType == CGToucanOPName::LUT);

          CGMicroPartInfo newPart;

          if (isLUTPart) {
            scheduleRegularMicroPart(graph, newPart, eachPart, partInfo.valueToValId, currentMicroPartitioner, dummyVtxIndexInVecTable);
          } else {
            scheduleSpecialMicroPart(graph, newPart, eachPart, codeGenInfo, partInfo);
          }

          partInfo.microPartOps.back().emplace_back(newPart);
          // add code gen meta
        }
      }


      // last level
      scheduleRegWrites(graph, partInfo, currentMicroPartitioner.allRegWrites);
      scheduleMemWrites(graph, partInfo, currentMicroPartitioner.allMemWrites);
      scheduleStops(graph, partInfo, currentMicroPartitioner.allStops);
      schedulePrints(graph, partInfo, currentMicroPartitioner.allPrints);


      // Add statistics
      {
        CGLayerValueStatistics stats;
        std::memset(&stats, 0, sizeof(CGLayerValueStatistics));
        stats.numRegWrites = partInfo.regWriteOps.size();
        stats.numMemWrites = partInfo.memWriteOps.size();
        stats.numPrints = partInfo.printOps.size();
        stats.numStops = partInfo.stopOps.size();
      
        partInfo.opStatisticsPerLevel.push_back(stats);
        partInfo.opStatistics.numRegWrites = stats.numRegWrites;
        partInfo.opStatistics.numMemWrites = stats.numMemWrites;
        partInfo.opStatistics.numPrints = stats.numPrints;
        partInfo.opStatistics.numStops = stats.numStops;
      }
      codeGenInfo.partitionInfo.push_back(std::move(partInfo));
    }
  }
  

  return;
}





