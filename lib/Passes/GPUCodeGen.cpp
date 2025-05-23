
#include "circt/Dialect/SV/SVDialect.h"
#include "circt/Dialect/OM/OMDialect.h"
#include "circt/Dialect/Seq/SeqDialect.h"
#include "circt/Support/LLVM.h"

#include "mlir/IR/BuiltinOps.h"
#include "mlir/IR/BuiltinTypes.h"
#include "mlir/IR/OpImplementation.h"
#include "mlir/IR/Operation.h"
#include "mlir/IR/Visitors.h"
#include "mlir/Interfaces/SideEffectInterfaces.h"
#include "mlir/Support/LLVM.h"
#include "mlir/IR/Threading.h"
#include "mlir/Support/LogicalResult.h"
#include "toucan/MicroPartitioner.h"
#include "toucan/PartitioningGraph.h"
#include "toucan/ToucanAnalysis.h"

#include "llvm/ADT/DenseMap.h"
#include "llvm/ADT/SmallPtrSet.h"
#include "llvm/ADT/SmallVector.h"
#include "llvm/ADT/STLExtras.h"

#include "llvm/ADT/StringExtras.h"
#include "llvm/Support/Casting.h"
#include "llvm/Support/Debug.h"
#include "llvm/Support/ErrorHandling.h"
#include "llvm/Support/raw_ostream.h"

#include <cstddef>
#include <cstdint>
#include <algorithm>
#include <iterator>
#include <memory>
#include <filesystem>
#include <fstream>
#include <tuple>
#include <vector>


#define GEN_PASS_DEF_GPUCODEGEN
#include "toucan/ToucanPassCommon.h"

#include "toucan/MicroPartitioner.h"
#include "toucan/CodeGenCommon.h"
#include "ToucanGPUSim/ToucanGPUGenDataTypes.h"

using namespace toucan;
using namespace circt;
using namespace mlir;
using namespace llvm;

#define DEBUG_TYPE "GPUCodeGenPass"

// #define ENABLE_REG_READ_DEBUG_PRINT
// #define ENABLE_REG_WRITE_DEBUG_PRINT
// #define ENABLE_EXG_READ_DEBUG_PRINT
// #define ENABLE_EXG_WRITE_DEBUG_PRINT

struct GPUCodeGenPass : toucan::impl::GPUCodeGenBase<GPUCodeGenPass>, CodeGenHelper {
  using GPUCodeGenBase<GPUCodeGenPass>::GPUCodeGenBase;

  toucanGPUSim::SimDesignInfo designInfo;
  toucanGPUSim::SimDebugInfo debugInfo;

  /*
  static void packRegRead(mlir::SmallVector<toucanGPUSim::CGRegReadMetaInfo> &regReadBulks) {
    // use packed SIMD for reg reads if possible
    assert(!regReadBulks.empty());

    // allocate thread work
    mlir::SmallVector<toucanGPUSim::CGRegReadMetaInfo> raw_reg_read_ops;
    for (auto bulk: regReadBulks) {
      if (bulk.byteCount == 1) {
        // nothing we can do for 1 byte reads. simply copy them
        raw_reg_read_ops.push_back(bulk);
      } else {
        auto alignment_padding_bytes = getExtraAlignmentSpace(bulk.reg, 4 * GPU_THREAD_WARP_SIZE);
        auto tail_bytes = (bulk.reg + bulk.byteCount) % (GPU_THREAD_WARP_SIZE * 4);

        auto bytes_to_pack = bulk.byteCount - alignment_padding_bytes - tail_bytes;
        if ((bulk.byteCount < (GPU_THREAD_WARP_SIZE * 4)) || (bytes_to_pack < (GPU_THREAD_WARP_SIZE * 4))) {
          // too small for packed SIMD
          for (int i = 0; i < bulk.byteCount; i++) {
            toucanGPUSim::CGRegReadMetaInfo info;
            info.byteCount = 1;
            info.reg = bulk.reg + i;
            info.result = bulk.result + i;
            raw_reg_read_ops.push_back(info);
          }
        } else {
          // large bulk, worth packed SIMD copy
          assert(bytes_to_pack <= bulk.byteCount && "bytes_to_pack should not underflow");

          // copy heading aligment bytes
          for (size_t i = 0; i < alignment_padding_bytes; i++) {
            toucanGPUSim::CGRegReadMetaInfo info;
            info.byteCount = 1;
            info.reg = bulk.reg + i;
            info.result = bulk.result + i;
            raw_reg_read_ops.push_back(info);
          }

          // reg should be aligned
          assert((bulk.reg + alignment_padding_bytes) % 4 == 0);
          assert(bytes_to_pack <= bulk.byteCount);

          auto numCopyIterations = bytes_to_pack / (GPU_THREAD_WARP_SIZE * 4);
          assert(bytes_to_pack % (GPU_THREAD_WARP_SIZE * 4) == 0);
          auto numWarpNeeded = (numCopyIterations + POLICY_PACKED_MAX_COPY_INT_COUNT - 1) / POLICY_PACKED_MAX_COPY_INT_COUNT;
          size_t copy_offset = alignment_padding_bytes;

          for (size_t warp_id = 0; warp_id < numWarpNeeded; warp_id++) {
            auto iterations = (warp_id + 1 == numWarpNeeded) ? (numCopyIterations - warp_id * POLICY_PACKED_MAX_COPY_INT_COUNT) : POLICY_PACKED_MAX_COPY_INT_COUNT;

            for (size_t i = 0; i < GPU_THREAD_WARP_SIZE; i++) {
              // assign task for every threads in a warp
              toucanGPUSim::CGRegReadMetaInfo info;
              info.byteCount = iterations * 4;
              info.reg = bulk.reg + copy_offset + iterations * 4 * i;
              info.result = bulk.result + copy_offset + iterations * 4 * i;
              raw_reg_read_ops.push_back(info);
            }

            copy_offset += iterations * GPU_THREAD_WARP_SIZE * 4;
          }

          // tail
          assert(copy_offset + tail_bytes == bulk.byteCount);
          for (size_t i = 0; i < tail_bytes; i++) {
            toucanGPUSim::CGRegReadMetaInfo info;
            info.byteCount = 1;
            info.reg = bulk.reg + copy_offset + i;
            info.result = bulk.result + copy_offset + i;
            raw_reg_read_ops.push_back(info);
          }
        }
      }
    }

    // sort (stable) by byte size, then add padding
    regReadBulks.clear();
    assert(!raw_reg_read_ops.empty());

    std::stable_sort(raw_reg_read_ops.begin(), raw_reg_read_ops.end(), [&](const toucanGPUSim::CGRegReadMetaInfo a, const toucanGPUSim::CGRegReadMetaInfo b) {
      return a.byteCount < b.byteCount;
    });

    // insert proper padding
    uint16_t last_copy_byte_count = 0;
    for (auto op: raw_reg_read_ops) {
      if (last_copy_byte_count == 1 && op.byteCount != 1) {
        // done
        auto numPaddingOps = getExtraAlignmentSpace(regReadBulks.size(), GPU_THREAD_WARP_SIZE);
        for (size_t i = 0; i < numPaddingOps; i++) {
          // insert a dummy copy (0 bytes)
          regReadBulks.push_back({0, 0, 0});
        }
      }
      last_copy_byte_count = op.byteCount;
      regReadBulks.push_back(op);
    }
  }

  static void packExgRead(mlir::SmallVector<toucanGPUSim::CGExchangeReadMetaInfo> &exgReadBulks) {
    // use packed SIMD for exg reads if possible
    assert(!exgReadBulks.empty());

    // allocate thread work
    mlir::SmallVector<toucanGPUSim::CGExchangeReadMetaInfo> raw_exg_read_ops;
    for (auto bulk: exgReadBulks) {
      if (bulk.byteCount == 1) {
        // nothing we can do for 1 byte reads. simply copy them
        raw_exg_read_ops.push_back(bulk);
      } else {
        auto alignment_padding_bytes = getExtraAlignmentSpace(bulk.exchangeVal, 4 * GPU_THREAD_WARP_SIZE);
        auto tail_bytes = (bulk.exchangeVal + bulk.byteCount) % (GPU_THREAD_WARP_SIZE * 4);

        auto bytes_to_pack = bulk.byteCount - alignment_padding_bytes - tail_bytes;
        if ((bulk.byteCount < (GPU_THREAD_WARP_SIZE * 4)) || (bytes_to_pack < (GPU_THREAD_WARP_SIZE * 4))) {
          // too small for packed SIMD
          for (int i = 0; i < bulk.byteCount; i++) {
            toucanGPUSim::CGExchangeReadMetaInfo info;
            info.byteCount = 1;
            info.exchangeVal = bulk.exchangeVal + i;
            info.localVal = bulk.localVal + i;
            raw_exg_read_ops.push_back(info);
          }
        } else {
          // large bulk, worth packed SIMD copy
          assert(bytes_to_pack <= bulk.byteCount && "bytes_to_pack should not underflow");

          // copy heading aligment bytes
          for (size_t i = 0; i < alignment_padding_bytes; i++) {
            toucanGPUSim::CGExchangeReadMetaInfo info;
            info.byteCount = 1;
            info.exchangeVal = bulk.exchangeVal + i;
            info.localVal = bulk.localVal + i;
            raw_exg_read_ops.push_back(info);
          }

          // reg should be aligned
          assert((bulk.exchangeVal + alignment_padding_bytes) % 4 == 0);
          assert(bytes_to_pack <= bulk.byteCount);

          auto numCopyIterations = bytes_to_pack / (GPU_THREAD_WARP_SIZE * 4);
          assert(bytes_to_pack % (GPU_THREAD_WARP_SIZE * 4) == 0);
          auto numWarpNeeded = (numCopyIterations + POLICY_PACKED_MAX_COPY_INT_COUNT - 1) / POLICY_PACKED_MAX_COPY_INT_COUNT;
          size_t copy_offset = alignment_padding_bytes;

          for (size_t warp_id = 0; warp_id < numWarpNeeded; warp_id++) {
            auto iterations = (warp_id + 1 == numWarpNeeded) ? (numCopyIterations - warp_id * POLICY_PACKED_MAX_COPY_INT_COUNT) : POLICY_PACKED_MAX_COPY_INT_COUNT;

            for (size_t i = 0; i < GPU_THREAD_WARP_SIZE; i++) {
              // assign task for every threads in a warp
              toucanGPUSim::CGExchangeReadMetaInfo info;
              info.byteCount = iterations * 4;
              info.exchangeVal = bulk.exchangeVal + copy_offset + iterations * 4 * i;
              info.localVal = bulk.localVal + copy_offset + iterations * 4 * i;
              raw_exg_read_ops.push_back(info);
            }

            copy_offset += iterations * GPU_THREAD_WARP_SIZE * 4;
          }

          // tail
          assert(copy_offset + tail_bytes == bulk.byteCount);
          for (size_t i = 0; i < tail_bytes; i++) {
            toucanGPUSim::CGExchangeReadMetaInfo info;
            info.byteCount = 1;
            info.exchangeVal = bulk.exchangeVal + copy_offset + i;
            info.localVal = bulk.localVal + copy_offset + i;
            raw_exg_read_ops.push_back(info);
          }
        }
      }
    }

    // sort (stable) by byte size, then add padding
    exgReadBulks.clear();
    assert(!raw_exg_read_ops.empty());

    std::stable_sort(raw_exg_read_ops.begin(), raw_exg_read_ops.end(), [&](const toucanGPUSim::CGExchangeReadMetaInfo a, const toucanGPUSim::CGExchangeReadMetaInfo b) {
      return a.byteCount < b.byteCount;
    });

    // insert proper padding
    uint16_t last_copy_byte_count = 0;
    for (auto op: raw_exg_read_ops) {
      if (last_copy_byte_count == 1 && op.byteCount != 1) {
        // done
        auto numPaddingOps = getExtraAlignmentSpace(exgReadBulks.size(), GPU_THREAD_WARP_SIZE);
        for (size_t i = 0; i < numPaddingOps; i++) {
          // insert a dummy copy (0 bytes)
          exgReadBulks.push_back({0, 0, 0});
        }
      }
      last_copy_byte_count = op.byteCount;
      exgReadBulks.push_back(op);
    }
  }


  void populateSinglePartition(const CGPartitionMetaInfo &part, uint32_t regionId, uint32_t partId) {
    toucanGPUSim::SimPartitionInfo partInfo;
    // have at least 2 levels. one for input, one for output
    assert(part.opPool.size() >= 2);
    partInfo.valuePool.resize(part.numConstsInValuePool);
    for (size_t i = 0; i < part.numConstsInValuePool; i++) {
      partInfo.valuePool[i] = part.constValuePool[i];
    }
    assert(part.constValuePool.size() == part.numConstsInValuePool);
    partInfo.valuePoolSize = part.numTotalValues;
    partInfo.numConstsInValuePool = part.numConstsInValuePool;
    if (partInfo.valuePoolSize > UINT16_MAX) {
      llvm::errs() << "Value pool size is " << partInfo.valuePoolSize << ", which exceeds UINT16_MAX. This should not happen.\n";
    }
    assert(partInfo.valuePoolSize <= UINT16_MAX && "Value pool is too large");

    // copy const vec data
    std::copy(part.constVecPool.begin(), part.constVecPool.end(), std::back_inserter(partInfo.constVecPool));

#if defined (ENABLE_EXG_READ_DEBUG_PRINT) || defined (ENABLE_REG_READ_DEBUG_PRINT) || defined (ENABLE_EXG_WRITE_DEBUG_PRINT) || defined (ENABLE_REG_WRITE_DEBUG_PRINT)
    llvm::dbgs() << "Info in a new partition ========\n";
#endif

    mlir::SmallVector<toucanGPUSim::CGRegReadMetaInfo> regReadBulks;
    toucanGPUSim::CGRegReadMetaInfo rr_bulk = {0, 0, 0};

    mlir::SmallVector<toucanGPUSim::CGExchangeReadMetaInfo> exgReadBulks;
    toucanGPUSim::CGExchangeReadMetaInfo er_bulk = {0, 0, 0};

    for (size_t i = 0; i < part.opPool[0].size(); i++) {
      auto &opMeta = part.opPool[0][i];

      switch (opMeta.opName) {
        case CGToucanOPName::RegRead: {
          assert(opMeta.regRead.result <= UINT16_MAX);


          if (rr_bulk.byteCount == 0) {
            // begin
            rr_bulk.byteCount = 1;
            rr_bulk.reg = opMeta.regRead.reg;
            rr_bulk.result = opMeta.regRead.result;
          } else {
            if ((rr_bulk.reg + rr_bulk.byteCount == opMeta.regRead.reg) && (rr_bulk.result + rr_bulk.byteCount == opMeta.regRead.result)) {
              // continus 
              rr_bulk.byteCount++;
            } else {
              // break
              regReadBulks.push_back(rr_bulk);

              rr_bulk.byteCount = 1;
              rr_bulk.reg = opMeta.regRead.reg;
              rr_bulk.result = opMeta.regRead.result;
            }
          }

          // partInfo.ops_l0_regRead.push_back(info);
          break;
        }
        case CGToucanOPName::ExchangeRead: {
          assert(opMeta.exgRead.localVal <= UINT16_MAX);

          if (er_bulk.byteCount == 0) {
            // begin
            er_bulk.byteCount = 1;
            er_bulk.exchangeVal = opMeta.exgRead.exchangeVal;
            er_bulk.localVal = opMeta.exgRead.localVal;
          } else {
            if ((er_bulk.exchangeVal + er_bulk.byteCount == opMeta.exgRead.exchangeVal) && (er_bulk.localVal + er_bulk.byteCount == opMeta.exgRead.localVal)) {
              // continus 
              er_bulk.byteCount++;
            } else {
              // break
              exgReadBulks.push_back(er_bulk);

              er_bulk.byteCount = 1;
              er_bulk.exchangeVal = opMeta.exgRead.exchangeVal;
              er_bulk.localVal = opMeta.exgRead.localVal;
            }
          }

          // partInfo.ops_l0_exgRead.push_back(info);
          break;
        }
        default: {
          llvm_unreachable("Unknow op name appears in first level");
        }
      }
    }

    if (rr_bulk.byteCount != 0) {
      regReadBulks.push_back(rr_bulk);
    }

    if (er_bulk.byteCount != 0) {
      exgReadBulks.push_back(er_bulk);
    }

#ifdef ENABLE_REG_READ_DEBUG_PRINT
    for (auto eachBulk: regReadBulks) {
      llvm::dbgs() << "Reg read bulk, from reg " << eachBulk.reg << " to local " << eachBulk.result << ", bulk size " << eachBulk.byteCount << "\n";
    }
#endif

#ifdef ENABLE_EXG_READ_DEBUG_PRINT
    for (auto eachBulk: exgReadBulks) {
      llvm::dbgs() << "Exchange read bulk, from exchange pool " << eachBulk.exchangeVal << " to local " << eachBulk.localVal << ", bulk size " << eachBulk.byteCount << "\n";
    }
#endif

    // use packed SIMD for reg reads if possible
    if (!regReadBulks.empty()) {
      packRegRead(regReadBulks);
      
      partInfo.ops_l0_regRead.reserve(regReadBulks.size());
      std::copy(regReadBulks.begin(), regReadBulks.end(), std::back_inserter(partInfo.ops_l0_regRead));
    }


#ifdef ENABLE_REG_READ_DEBUG_PRINT
    mlir::SmallVector<uint32_t> rr_copyCounts;
    rr_copyCounts.resize(POLICY_PACKED_MAX_COPY_INT_COUNT * 4 + 1, 0);

    for (auto op: partInfo.ops_l0_regRead) {
      rr_copyCounts[op.byteCount]++;
    }

    for (size_t numBytes = 0; numBytes < POLICY_PACKED_MAX_COPY_INT_COUNT * 4 + 1; numBytes++) {
      auto opCount = rr_copyCounts[numBytes];
      if (opCount != 0) {
        llvm::dbgs() << " " << opCount << " ops write to " << numBytes << " bytes\n";
      }
    }
#endif

    // use packed SIMD for exchange reads if possible
    if (!exgReadBulks.empty()) {
      llvm::dbgs() << "Exgread ops bulks " << exgReadBulks.size() << "\n";

      packExgRead(exgReadBulks);
      
      partInfo.ops_l0_exgRead.reserve(exgReadBulks.size());
      std::copy(exgReadBulks.begin(), exgReadBulks.end(), std::back_inserter(partInfo.ops_l0_exgRead));
    }


#ifdef ENABLE_EXG_READ_DEBUG_PRINT
    mlir::SmallVector<uint32_t> er_copyCounts;
    er_copyCounts.resize(POLICY_PACKED_MAX_COPY_INT_COUNT * 4 + 1, 0);

    llvm::dbgs() << "Exgread ops " << partInfo.ops_l0_exgRead.size() << "\n";

    for (auto op: partInfo.ops_l0_exgRead) {
      assert(op.byteCount < er_copyCounts.size());
      er_copyCounts[op.byteCount]++;
    }

    for (size_t numBytes = 0; numBytes < POLICY_PACKED_MAX_COPY_INT_COUNT * 4 + 1; numBytes++) {
      auto opCount = er_copyCounts[numBytes];
      if (opCount != 0) {
        llvm::dbgs() << " Exchange read: " << opCount << " ops write to " << numBytes << " bytes\n";
      }
    }
#endif

    // ops, middle levels
    for (size_t levelId = 1; levelId < part.opPool.size() - 1; levelId++) {
      auto execOpsIdx = levelId - 1;
      auto &currentLevel = part.opPool[levelId];

      partInfo.ops_exec_memRead.emplace_back();
      partInfo.ops_exec_vecRead.emplace_back();
      partInfo.ops_exec_lut.emplace_back();

      for (size_t i = 0; i < currentLevel.size(); i++) {
        auto &opMeta = currentLevel[i];
        switch (opMeta.opName) {
          case CGToucanOPName::LUT: {
            assert(opMeta.lut.op0 <= UINT16_MAX);
            assert(opMeta.lut.op1 <= UINT16_MAX);
            assert(opMeta.lut.op2 <= UINT16_MAX);
            assert(opMeta.lut.result <= UINT16_MAX);
            toucanGPUSim::CGLUTMetaInfo info;

            auto lutIndex = lutPos[static_cast<uint32_t>(opMeta.lut.lutId)];
            info.lutIndex = lutIndex;
            info.op0 = opMeta.lut.op0;
            info.op1 = opMeta.lut.op1;
            info.op2 = opMeta.lut.op2;
            info.result = opMeta.lut.result;

            partInfo.ops_exec_lut.back().push_back(info);
            break;
          }
          case CGToucanOPName::VecRead: {
            assert(opMeta.vec.index0 <= UINT16_MAX);
            assert(opMeta.vec.index1 <= UINT16_MAX);
            assert(opMeta.vec.index2 <= UINT16_MAX);
            assert(opMeta.vec.index3 <= UINT16_MAX);
            assert(opMeta.vec.outRangeValue <= UINT16_MAX);
            assert(opMeta.vec.result <= UINT16_MAX);

            assert(opMeta.vec.vecLength < UINT16_MAX && "Vector is too long");

            toucanGPUSim::CGVecReadMetaInfo info;

            info.vecBase = opMeta.vec.vecBase;
            info.vecLength = opMeta.vec.vecLength;
            info.isConstVec = opMeta.vec.isConstVec;
            info.index0 = opMeta.vec.index0;
            info.index1 = opMeta.vec.index1;
            info.index2 = opMeta.vec.index2;
            info.index3 = opMeta.vec.index3;
            info.outRangeValue = opMeta.vec.outRangeValue;
            info.offset = opMeta.vec.offset;
            info.result = opMeta.vec.result;

            partInfo.ops_exec_vecRead.back().push_back(info);
            break;
          }
          case CGToucanOPName::MemRead: {
            assert(opMeta.memRead.en <= UINT16_MAX);
            assert(opMeta.memRead.addrVec <= UINT16_MAX);
            assert(opMeta.memRead.result <= UINT16_MAX);

            toucanGPUSim::CGMemReadMetaInfo info;

            info.hasMultipleWriter = opMeta.memRead.hasMultipleWriter;
            // info.memDepth = opMeta.memRead.memDepth;
            info.memBase = opMeta.memRead.memBase;
            info.en = opMeta.memRead.en;
            info.addrVec = opMeta.memRead.addrVec;
            info.result = opMeta.memRead.result;

            partInfo.ops_exec_memRead.back().push_back(info);
            break;
          }
          default: {
            llvm::dbgs() << static_cast<uint32_t>(opMeta.opName);
            llvm_unreachable("Other ops should not appear here");
          }
        }
      }
    }

    uint32_t ew_bulk_size = 0;
    uint32_t ew_bulk_start_local = 0;
    uint32_t ew_bulk_start_exchange = 0;  

    uint32_t rw_bulk_size = 0;
    uint32_t rw_bulk_start_dat = 0;
    uint32_t rw_bulk_start_reg = 0;

    // ops, last level
    for (size_t i = 0; i < part.opPool.back().size(); i++) {
      auto &opMeta = part.opPool.back()[i];
      switch (opMeta.opName) {
        case CGToucanOPName::Print: {
          assert(opMeta.print.en <= UINT16_MAX);
          assert(opMeta.print.msg <= UINT16_MAX);

          toucanGPUSim::CGPrintMetaInfo info;
          info.en = opMeta.print.en;
          info.msg = opMeta.print.msg;
          
          partInfo.ops_last_print.push_back(info);
          break;
        }
        case CGToucanOPName::Stop: {
          assert(opMeta.stop.en <= UINT16_MAX);

          toucanGPUSim::CGStopMetaInfo info;
          info.en = opMeta.stop.en;
          
          partInfo.ops_last_stop.push_back(info);
          break;
        }
        case CGToucanOPName::RegWrite: {
          // special handling for regWrites
          assert(opMeta.regWrite.dat <= UINT16_MAX);

          if (rw_bulk_size == 0) {
            rw_bulk_start_dat = opMeta.regWrite.dat;
            rw_bulk_start_reg = opMeta.regWrite.reg;
            rw_bulk_size = 1;
          } else {
            if (opMeta.regWrite.dat == rw_bulk_start_dat + rw_bulk_size && opMeta.regWrite.reg == rw_bulk_start_reg + rw_bulk_size) {
              // bulk
              rw_bulk_size += 1;
            } else {
              // a new bulk. This should not happen
              assert(false && "Each partition should only write to a contiguous range of registers");
            }
          }

          break;
        }
        case CGToucanOPName::MemWrite: {
          assert(opMeta.memWrite.addrVec <= UINT16_MAX);
          assert(opMeta.memWrite.dat <= UINT16_MAX);
          assert(opMeta.memWrite.en <= UINT16_MAX);

          toucanGPUSim::CGMemWriteMetaInfo info;
          info.hasMultipleWriter = opMeta.memWrite.hasMultipleWriter;
          // info.memDepth = opMeta.memWrite.memDepth;
          info.memBase = opMeta.memWrite.memBase;
          info.addrVec = opMeta.memWrite.addrVec;
          info.dat = opMeta.memWrite.dat;
          info.en = opMeta.memWrite.en;
          
          partInfo.ops_last_memWrite.push_back(info);
          break;
        }
        case CGToucanOPName::ExchangeWrite: {
          assert(opMeta.exgWrite.localVal <= UINT16_MAX);

          if (ew_bulk_size == 0) {
            ew_bulk_start_local = opMeta.exgWrite.localVal;
            ew_bulk_start_exchange = opMeta.exgWrite.exchangeVal;
            ew_bulk_size = 1;
          } else {
            if (opMeta.exgWrite.localVal == ew_bulk_start_local + ew_bulk_size && opMeta.exgWrite.exchangeVal == ew_bulk_start_exchange + ew_bulk_size) {
              // bulk
              ew_bulk_size += 1;
            } else {
              // a new bulk. This should not happen
              assert(false && "Each partition should only write to a contiguous range of exchange pool");
            }
          }

          break;
        }
        default: {
          llvm_unreachable("Other type of ops should not appear in last level!");
        }
      }
    }

    // add extra padding
    if (rw_bulk_size != 0) {
      rw_bulk_size += getExtraAlignmentSpace(rw_bulk_size, 16);
    }

    if (ew_bulk_size != 0) {
      ew_bulk_size += getExtraAlignmentSpace(ew_bulk_size, 16);
    }

    // special handling for exchange writes
#ifdef ENABLE_EXG_WRITE_DEBUG_PRINT
    if (ew_bulk_size != 0) {
      llvm::dbgs() << "Exchange write bulk, from local pool " << ew_bulk_start_local << " to exchange pool " << ew_bulk_start_exchange << ", bulk size " << ew_bulk_size << "\n";
    }
#endif

    if (ew_bulk_size != 0) {
      assert(ew_bulk_start_local == 16 && "Local data should starts from shared memory addr 16");
      assert(ew_bulk_start_exchange % 16 == 0 && "Register should be aligned to 16B");
      assert(ew_bulk_size < UINT16_MAX && "Register write count should fit in uint16");

      toucanGPUSim::CGExchangeWriteMetaInfo info;
      info.localVal = ew_bulk_start_local;
      info.exchangeVal = ew_bulk_start_exchange;
      info.count = ew_bulk_size;

      partInfo.ops_last_exgWrite.push_back(info);
    }

    // special handling for regWrites
#ifdef ENABLE_REG_WRITE_DEBUG_PRINT
    if (rw_bulk_size != 0) {
      llvm::dbgs() << "Reg write bulk, from data " << rw_bulk_start_dat << " to reg " << rw_bulk_start_reg << ", bulk size " << rw_bulk_size << "\n";
    }
#endif
    if (rw_bulk_size != 0) {
      assert(rw_bulk_start_dat + rw_bulk_size - 1 <= UINT16_MAX);
      // Note: This is guarenteed by LocalValueAllocator
      assert(rw_bulk_start_dat == 16 && "Local data should starts from shared memory addr 16");
      assert(rw_bulk_start_reg % 16 == 0 && "Register should be aligned to 16B");
      assert(rw_bulk_size < UINT16_MAX && "Register write count should fit in uint16");

      toucanGPUSim::CGRegWriteMetaInfo info;
      info.reg = rw_bulk_start_reg;
      info.dat = rw_bulk_start_dat;
      info.count = rw_bulk_size;

      partInfo.ops_last_regWrite.push_back(info);
    }

    designInfo.parts.push_back(std::move(partInfo));
  }

  void populateDebugInfo(const CGInfo& codeGenInfo) {
    std::vector<std::tuple<uint32_t, uint32_t>> eachRegDbgInfo;
    std::vector<std::tuple<uint32_t, uint32_t, uint32_t>> eachMemDbgInfo;
    std::vector<std::tuple<uint32_t, uint32_t, uint32_t>> eachSignalDbgInfo;

    for (auto [regNameRef, ids]: codeGenInfo.regDebugInfo) {
      eachRegDbgInfo.clear();
      for (auto regId: ids) {
        eachRegDbgInfo.push_back(std::make_tuple(regId, codeGenInfo.regPool[regId].bitWidth));
      }
      debugInfo.regDebugInfo[regNameRef.str()] = eachRegDbgInfo;
    }
    for (auto [memNameRef, ids]: codeGenInfo.memDebugInfo) {
      eachMemDbgInfo.clear();
      for (auto memId: ids) {
        auto startPos = codeGenInfo.memPool[memId].memBase;
        auto bitWidth = codeGenInfo.memPool[memId].bitWidth;
        auto memDepth = codeGenInfo.memPool[memId].memDepth;
        eachMemDbgInfo.push_back(std::make_tuple(startPos, bitWidth, memDepth));
      }
      debugInfo.memDebugInfo[memNameRef.str()] = eachMemDbgInfo;
    }
    for (auto [sigNameRef, sigLocs]: codeGenInfo.signalDebugInfo) {
      eachSignalDbgInfo.clear();

      for (auto sigLoc: sigLocs) {
        auto partId = std::get<0>(sigLoc);
        auto sigId = std::get<1>(sigLoc);
        auto sigBitWidth = codeGenInfo.partitionInfo[partId].valuePool[sigId].bitWidth;
        eachSignalDbgInfo.push_back(std::make_tuple(partId, sigId, sigBitWidth));
      }
      debugInfo.signalDebugInfo[sigNameRef.str()] = eachSignalDbgInfo;
    }
  }
    */

  void runOnOperation() final {
    // Mark all analyses as preserved. This is a read only pass
    markAllAnalysesPreserved();

    auto graph = getAnalysis<DesignGraph>();

    auto p = RepCutPartitioner(outputDirectory.getValue());

    // Levelize
    llvm::outs() << "====================Levelize And Cut====================\n";

    // Looks like only 1 region is enough
    p.doNotCutGraph(graph);


    // p.levelizeGraphForCut(graph);

    // // Cut into 2 subgraph
    // p.findCutPoints(graph);
    // p.cutGraph(graph);
    p.breakDirectIOConnection();

    // Detect number of partitions in each region by heuristic.
    p.setPartitionTarget();


    assert(ibFactor > 0.0f);
    p.targetIb = ibFactor;

    // auto result = p.partitionAndSchedule(&getContext(), graph);
    // Just partition.
    auto result = p._partition(&getContext(), graph);

    if (failed(result)) {
      return signalPassFailure();
    }
    p.dumpAllPartitionsToFile();

    // Second level partitioning
    // For now, don't cut in the middle
    auto numRegions = p.regionGraphs.size();
    assert(numRegions == 1);

    std::vector<MicroPartitioner> mps;
    for (size_t regionId = 0; regionId < numRegions; regionId++) {
      auto regionGraph = p.regionGraphs[regionId];
      
      auto numParts = p.regionPartitions[regionId].size();
      mps.reserve(numParts);
  
      auto thisRegionWorkDirectory = p.regionWorkDirectory[regionId];
  
      for (size_t partId = 0; partId < numParts; partId++) {
        llvm::outs() << "Partitioning region " << regionId << " part " << partId << "\n";

        mps.emplace_back(MicroPartitioner(thisRegionWorkDirectory, partId, p.originalVectorElementsMap));
        auto &mp = mps.back();

        auto ret = mp.partition();

        if (failed(ret)) {
          signalPassFailure();
          return;
        }

        mp.collectPartIOValues(regionGraph);

      }
    }

    // Schedule
    // Note: For now only use 1 region
    assert(p.regionPartitions.size() == 1);

    auto scheduler = SingleRegionMicroPartScheduler();
    scheduler.mpartitioners.swap(mps);
    scheduler.repcutPartitions = p.regionPartitions[0];

    auto rGraph = p.regionGraphs[0];
    assert(p.regionGraphs.size() == 1 && "SingleRegionMicroPartScheduler only supports 1 region");
    auto partNodeList = p.regionPartitions[0];
    assert(partNodeList.size() == p.regionPartitionNumbers[0]);

    scheduler.schedule(rGraph, partNodeList);
    // p.regionPartitions


    // Under construction...

    return;

    


    // Fill lut
    populateLUT();
    // designInfo.lut.assign(lutContent.begin(), lutContent.end());

    // // copy pool size
    // designInfo.regPoolSize = p.codeGenInfo.regPool.size();
    // designInfo.memPoolSize = p.codeGenInfo.totalMemSize;
    // designInfo.exchangePoolSize = p.codeGenInfo.exchangePool.size();

    // uint32_t partId = 0;
    // uint32_t regionId = 0;
    // for (const auto &eachRegionParts: p.codeGenInfo.regionPartitionIds) {
    //   designInfo.regionPartitionIds.emplace_back();
    //   for (const auto &eachPartId: eachRegionParts) {
    //     assert(eachPartId == partId);

    //     // save partition region info
    //     designInfo.regionPartitionIds.back().push_back(partId);
    //     // codegen for a single partition
    //     populateSinglePartition(p.codeGenInfo.partitionInfo[partId], regionId, partId);
    //     partId++;
    //   }
    //   regionId++;
    // }

    // // Fill print msgs
    // designInfo.printMsgs.resize(p.codeGenInfo.printStrings.size());
    // for (auto [k, v]: p.codeGenInfo.printStrings) {
    //   assert(designInfo.printMsgs[v].empty());
    //   designInfo.printMsgs[v] = k;
    // }

    // // Fill debug info
    // populateDebugInfo(p.codeGenInfo);



    // Save. serialize
    auto outputDesignFileFullName = std::filesystem::path(outputDirectory.getValue()) / outputDesignFilename.getValue();
    std::ofstream ofs_design(outputDesignFileFullName, std::ios::binary | std::ios::out);
    toucanGPUSim::serializeSimDesignInfo(ofs_design, designInfo);
    ofs_design.close();


    // Debug symbols
    auto outputSymbolFileFullName = std::filesystem::path(outputDirectory.getValue()) / outputSymbolFilename.getValue();
    std::ofstream ofs_symbol(outputSymbolFileFullName, std::ios::binary | std::ios::out);
    toucanGPUSim::serializeSimDebugInfo(ofs_symbol, debugInfo);
    ofs_symbol.close();

    // IO signal only
    debugInfo.memDebugInfo.clear();
    debugInfo.signalDebugInfo.clear();
    std::erase_if(debugInfo.regDebugInfo, [&](auto const &item) {
      auto const& [k, v] = item;
      return !(p.codeGenInfo.ioSignals.contains(k));
    });

    auto outputIOSymbolFileFullName = std::filesystem::path(outputDirectory.getValue()) / outputIOSymbolFilename.getValue();
    std::ofstream ofs_io_symbol(outputIOSymbolFileFullName, std::ios::binary | std::ios::out);
    toucanGPUSim::serializeSimDebugInfo(ofs_io_symbol, debugInfo);
    ofs_io_symbol.close();

  }

};

std::unique_ptr<mlir::Pass> toucan::createGPUCodeGenPass(GPUCodeGenOptions option) {
  return std::make_unique<GPUCodeGenPass>(option);
}
