
#include "circt/Dialect/HW/HWOps.h"
#include "circt/Dialect/HW/HWTypes.h"
#include "circt/Dialect/Seq/SeqAttributes.h"
#include "circt/Support/LLVM.h"
#include "circt/Dialect/Comb/CombDialect.h"
#include "circt/Dialect/Comb/CombOps.h"
#include "circt/Dialect/Seq/SeqOps.h"

#include "mlir/IR/Builders.h"
#include "mlir/IR/BuiltinAttributes.h"
#include "mlir/IR/PatternMatch.h"
#include "mlir/IR/BuiltinOps.h"
#include "mlir/IR/BuiltinTypes.h"
#include "mlir/IR/OpImplementation.h"
#include "mlir/IR/Operation.h"
#include "mlir/IR/Value.h"
#include "mlir/IR/Visitors.h"
#include "mlir/Support/LLVM.h"
#include "mlir/IR/Threading.h"
#include "mlir/Support/LogicalResult.h"
#include "toucan/ToucanTypes.h"

#include "llvm/ADT/SmallPtrSet.h"
#include "llvm/ADT/SmallVector.h"
#include "llvm/ADT/STLExtras.h"

#include "llvm/Support/Casting.h"
#include "llvm/Support/Debug.h"
#include "llvm/Support/raw_ostream.h"

#include <memory>
#include <string>
#include <atomic>


#define GEN_PASS_DEF_REMOVEMEMMASK
#include "toucan/ToucanPassCommon.h"

#include "toucan/ToucanOps.h"
#include "toucan/ToucanUtils.h"

using namespace toucan;
using namespace circt;
using namespace mlir;
using namespace llvm;

#define DEBUG_TYPE "RemoveMemMaskPass"

static std::atomic<uint64_t> numMemWithMaskInModule;
static std::atomic<uint64_t> numNewMemInModule;

struct RemoveMemMaskPass : toucan::impl::RemoveMemMaskBase<RemoveMemMaskPass> {
  using RemoveMemMaskBase<RemoveMemMaskPass>::RemoveMemMaskBase;

  LogicalResult runOnModule(hw::HWModuleOp mod) {
    SmallVector<Operation*> toRemove;

    for (auto &stmt : mod.getOps()) {
      if (auto memOp = dyn_cast<seq::FirMemOp>(stmt)) {
        auto firMemDataType = memOp.getMemory().getType();
        auto memName = memOp.getName().value_or(StringRef(""));

        OpBuilder builder(memOp);
        IRRewriter rewriter(builder);
        rewriter.setInsertionPointAfter(memOp);

        auto memWidth = firMemDataType.getWidth();
        auto memDepth = firMemDataType.getDepth();

        SmallVector<mlir::Value> newMemValues;

        assert(memOp.getReadLatency() == 0);
        assert(memOp.getWriteLatency() == 1);

        if (memOp.getWuw() != seq::WUW::Undefined) {
          auto numWriters = count_if(memOp.getMemory().getUsers(), [](Operation *op) {
            return isa<seq::FirMemReadOp>(op) || isa<seq::FirMemReadWriteOp>(op);
          });
          if (numWriters == 0) {
            memOp->emitError() << "Memory %" << memName << " has no writer!";
            signalPassFailure();
          }
          if (numWriters > 1) {
            memOp->emitWarning() << "Memory %" << memName << " has a Write-Under-Write behavior of [" << seq::stringifyWUW(memOp.getWuw()) << "], which is not guaranteed. This may lead to incorrect result in simulation. (Memory has " << numWriters << " writers)";
          }
        }

        if (memOp.getRuw() != seq::RUW::Undefined) {
          memOp->emitWarning() << "Memory %" << memName << " has a Read-Under-Write behavior of [" << seq::stringifyRUW(memOp.getRuw()) << "], which is not guaranteed. This may lead to incorrect result in simulation.";
        }

        if (auto maskWidth = firMemDataType.getMaskWidth()) {
          // Has mask
          auto maskLaneWidth = memWidth / (*maskWidth);
          auto numSplittedMems = *maskWidth;

          numMemWithMaskInModule++;
          numNewMemInModule += numSplittedMems;

          if (memWidth % (*maskWidth) != 0) {
            memOp.emitError() << "Incorrect mask width, got " << (*maskWidth) << " for a " << memWidth << " width memory";
            return failure();
          }

          auto splitMemElemType = rewriter.getIntegerType(maskLaneWidth);
          // auto splitMemType = rewriter.getType<toucan::MemType>(memDepth, splitMemElemType);
          auto splitMemTypeAttr = rewriter.getAttr<toucan::MemType>(memDepth, splitMemElemType);
          size_t accumulated_mem_width = 0;
          for (uint32_t i = 0; i < numSplittedMems; i++) {
            // Create i'th memory
            // auto splitMemName = rewriter.getStringAttr(memName + "_" + std::to_string(i));

            auto newMemOp = rewriter.create<toucan::DefMemOp>(memOp.getLoc(), splitMemTypeAttr);
            auto newMem = newMemOp.getHandle();

            newMemValues.push_back(newMem);
            if (memName.size() != 0) {
              auto namehint = rewriter.getStringAttr(memName);
              setSVNameHintAttr(newMemOp, namehint);
            }

            auto accumulateMemWidthAttr = rewriter.getI32IntegerAttr(accumulated_mem_width);
            setAccumulatedMemWidthAttr(newMemOp, accumulateMemWidthAttr);
            // auto memMaskFragmentIdAttr = rewriter.getI32IntegerAttr(i);
            // setMemMaskFragmentIDAttr(newMemOp, memMaskFragmentIdAttr);
            
            accumulated_mem_width += newMem.getType().getElementWidth();
          }

        } else {
          // memory with no mask
          // don't split
          auto memElemType = rewriter.getIntegerType(memWidth);
          // auto memType = rewriter.getType<toucan::MemType>(memDepth, memElemType);
          auto memTypeAttr = rewriter.getAttr<toucan::MemType>(memDepth, memElemType);
          auto newMemOp = rewriter.create<toucan::DefMemOp>(memOp.getLoc(), memTypeAttr);
          auto newMem = newMemOp.getHandle();

          newMemValues.push_back(newMem);

          if (memName.size() != 0) {
            auto namehint = rewriter.getStringAttr(memName);
            setSVNameHintAttr(newMemOp, namehint);
          }
        }

        toRemove.push_back(memOp);
        assert(newMemValues.size() > 0);

        for (auto op: memOp.getMemory().getUsers()) {
          // Replace all memory use
          rewriter.setInsertionPointAfter(op);
          if (auto memReadOp = dyn_cast<seq::FirMemReadOp>(op)) {

            auto memAddr = memReadOp.getAddress();
            auto memAddrs = split_value_4B(op, memAddr, rewriter);
            auto memEn = memReadOp.getEnable();

            auto trueVal = rewriter.getIntegerAttr(rewriter.getI1Type(), 1);
            auto enSignalOp = rewriter.create<hw::ConstantOp>(op->getLoc(), trueVal);
            auto constTrue = cast<TypedValue<IntegerType>>(enSignalOp.getResult());

            auto memEnSignal = (memEn) ? memEn : constTrue;

            if (newMemValues.size() == 1) {
              // only 1, no mask
              auto newMem = newMemValues.front();

              auto readOp = rewriter.create<toucan::MemReadOp>(op->getLoc(), newMem, memAddrs, memEnSignal);

              auto namehint = getSVNameHintAttr(newMem.getDefiningOp()).value();
              setSVNameHintAttr(readOp, namehint);

              rewriter.replaceAllUsesWith(memReadOp, readOp);
            } else {
              // has multiple memory
              SmallVector<mlir::Value> memReadResults;
              
              for (auto &newMem: newMemValues) {
                auto readOp = rewriter.create<toucan::MemReadOp>(op->getLoc(), newMem, memAddrs, memEnSignal);
                memReadResults.push_back(readOp.getResult());

                auto namehint = getSVNameHintAttr(newMem.getDefiningOp()).value();
                setSVNameHintAttr(readOp, namehint);
              }

              auto concatOp = rewriter.create<comb::ConcatOp>(op->getLoc(), memReadResults);
              auto namehint = rewriter.getStringAttr(memName);
              setSVNameHintAttr(concatOp, namehint);

              rewriter.replaceAllUsesWith(memReadOp, concatOp);
            }
          } else if (auto memWriteOp = dyn_cast<seq::FirMemWriteOp>(op)) {
            auto memAddr = memWriteOp.getAddress();
            auto memAddrs = split_value_4B(op, memAddr, rewriter);

            auto memEn = memWriteOp.getEnable();
            auto memData = memWriteOp.getData();

            auto trueVal = rewriter.getIntegerAttr(rewriter.getI1Type(), 1);
            auto enSignalOp = rewriter.create<hw::ConstantOp>(op->getLoc(), trueVal);
            auto constTrue = cast<TypedValue<IntegerType>>(enSignalOp.getResult());

            auto memEnSignal = (memEn) ? memEn : constTrue;

            if (newMemValues.size() == 1){
              auto newMem = newMemValues.front();
              
              auto newMemWriteOp = rewriter.create<toucan::MemWriteOp>(op->getLoc(), newMem, memAddrs, memData, memEnSignal);auto namehint = getSVNameHintAttr(newMem.getDefiningOp()).value();
              setSVNameHintAttr(newMemWriteOp, namehint);
            } else {
              auto memMask = memWriteOp.getMask();
              assert(memMask.getType().getWidth() > 0);

              for (uint32_t memId = 0; memId < newMemValues.size(); memId ++) {
                auto newMem = newMemValues[memId];
                auto maskLaneWidth = newMem.getType().cast<MemType>().getElementWidth();

                auto memMaskWidth = hw::getBitWidth(memMask.getType());
                assert(memId < memMaskWidth);

                auto maskBitOp = rewriter.create<comb::ExtractOp>(op->getLoc(), memMask, memId, 1);
                auto newMemEnOp = rewriter.create<comb::AndOp>(op->getLoc(), maskBitOp.getResult(), memEnSignal, true);

                auto newMemEn = newMemEnOp.getResult();

                auto dataSliceOp = rewriter.create<comb::ExtractOp>(op->getLoc(), memData, memId * maskLaneWidth, maskLaneWidth);
                auto dataSlice = dataSliceOp.getResult();

                auto newMemWriteOp = rewriter.create<toucan::MemWriteOp>(op->getLoc(), newMem, memAddrs, dataSlice, newMemEn);
                
                auto namehint = getSVNameHintAttr(newMem.getDefiningOp()).value();
                setSVNameHintAttr(newMemWriteOp, namehint);
              }
            }
          } else {
            op->emitError() << "Unknow operator consumes a memory symbol";
            llvm::dbgs() << op->getName() << "\n";
            return failure();
          }
          toRemove.push_back(op);
        }
      }
    }

    for (auto op: llvm::reverse(toRemove)) {
      op->erase();
    }
    return success();
  }


  void runOnOperation() final {
    auto mod = getOperation();

    numMemWithMaskInModule = 0;
    numNewMemInModule = 0;

    SmallVector<hw::HWModuleOp> modulesToProcess;
    for(auto & inner: mod.getOps()) {
      if(auto mod = dyn_cast<hw::HWModuleOp>(&inner)) {
        modulesToProcess.push_back(mod);
      }
    }
    //  // Sequential
    //  for (auto mod: modulesToProcess) {
    //    auto ret = runOnModule(mod);
    //    if (failed(ret)) return signalPassFailure();
    //  }

   // Parallel
   auto result = mlir::failableParallelForEach(&getContext(), modulesToProcess.begin(), modulesToProcess.end(), [&](auto mod) {
     return runOnModule(mod);
   });
   if (failed(result)) return signalPassFailure();

   numMemWithMask = numMemWithMaskInModule;
   numNewMem = numNewMemInModule;
  }

};

std::unique_ptr<mlir::Pass> toucan::createRemoveMemMaskPass() {
  return std::make_unique<RemoveMemMaskPass>();
}
