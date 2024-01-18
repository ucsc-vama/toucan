

#include "circt/Dialect/HW/HWOps.h"
#include "mlir/IR/Builders.h"
#include "mlir/IR/BuiltinOps.h"
#include "mlir/IR/BuiltinTypes.h"
#include "mlir/IR/MLIRContext.h"
#include "mlir/IR/OpImplementation.h"
#include "mlir/IR/Operation.h"
#include "mlir/IR/Visitors.h"
#include "mlir/IR/PatternMatch.h"
#include "mlir/Support/LLVM.h"
#include "mlir/IR/Threading.h"
#include "mlir/Support/LogicalResult.h"

#include "llvm/ADT/SmallPtrSet.h"
#include "llvm/ADT/SmallVector.h"
#include "llvm/ADT/STLExtras.h"
#include "llvm/ADT/STLFunctionalExtras.h"

#include "llvm/Support/Casting.h"
#include "llvm/Support/Debug.h"
#include "llvm/Support/raw_ostream.h"
#include "llvm/Support/Format.h"


#include "circt/Dialect/Seq/SeqDialect.h"
#include "circt/Dialect/Seq/SeqOps.h"
#include "circt/Support/LLVM.h"


#include <cassert>
#include <stdint.h>
#include <memory>




#define GEN_PASS_DEF_EXPANDMEMORYDELAY
#include "toucan/ToucanPassCommon.h"

#include "toucan/ToucanUtils.h"

using namespace toucan;
using namespace circt;
using namespace mlir;
using namespace llvm;

#define DEBUG_TYPE "ExpandMemoryDelayPass"



static StringRef getMarkerStringRef() {
  return "_isMemPipelineHeadingRegister";
}



struct ExpandMemoryDelayPass : toucan::impl::ExpandMemoryDelayBase<ExpandMemoryDelayPass> {
  using ExpandMemoryDelayBase<ExpandMemoryDelayPass>::ExpandMemoryDelayBase;

  void unsetPipelineHeadingRegister(Operation *op) {
    op->removeAttr(getMarkerStringRef());
  }

  void markAsPipelineHeadingRegister(Operation *op) {
    op->setAttr(getMarkerStringRef(), BoolAttr::get(&getContext(), true));
  }

  bool isPipelineHeadingRegister(Operation *op) {
    if (op->hasAttr(getMarkerStringRef())) {
      return op->getAttrOfType<BoolAttr>(getMarkerStringRef()).getValue();
    }
    return false;
  }

  LogicalResult pipeliningMemDelayOnModule(hw::HWModuleOp &mod) {
    for (auto &stmt: mod.getOps()) {
      if (auto memReadOp = dyn_cast<seq::FirMemReadOp>(stmt)) {
        // For memory read ports with ReadLatency > 0

        auto mem = memReadOp.getMemory();
        auto memDefiningOp = cast<seq::FirMemOp>(mem.getDefiningOp());
        auto memReadLatency = memDefiningOp.getReadLatency();

        if (memReadLatency > 0) {
          LLVM_DEBUG(
            char buffer[256];
            format("Found memory %s with read latency of %d\n", memDefiningOp->getName().getStringRef().str().c_str(), memReadLatency).snprint(buffer, 256);
            llvm::dbgs() << buffer
            );

          auto clockSignal = memReadOp.getClk();

          OpBuilder builder(memReadOp);
          builder.setInsertionPointAfter(memReadOp);
          IRRewriter rewriter(builder);
        

          auto memName = memDefiningOp.getName().value_or(mlir::StringRef("default_mem_name"));
          seq::FirRegOp *lastReg = nullptr;
          for (uint32_t i = 0; i < memReadLatency; i++) {
            // insert a new pipeline register
            auto regName = mlir::StringAttr::get(memReadOp->getContext(), memName.str() + "_pipeline_" + std::to_string(i));
            auto next = (i == 0) ? memReadOp.getResult() : lastReg->getData();
            auto pipelineRegister = rewriter.create<seq::FirRegOp>(memReadOp->getLoc(), next, clockSignal, regName);
            lastReg = &pipelineRegister;
            if (i == 0) markAsPipelineHeadingRegister(pipelineRegister);
          }
          // Replace all reference with new value, except pipelining registers
          memReadOp.getResult().replaceUsesWithIf(lastReg->getResult(), [=](mlir::OpOperand& oprand) { 
            auto op = oprand.getOwner();
            if (isPipelineHeadingRegister(op)) {
              unsetPipelineHeadingRegister(op);
              return false;
            }
            return true;
          });
        }
      } else if (auto memWriteOp = dyn_cast<seq::FirMemWriteOp>(stmt)) {
        // I don't expect existance of memory with WriteLatency != 1. Just assert here for safe.
        auto mem = memWriteOp.getMemory();
        auto memDefiningOp = cast<seq::FirMemOp>(mem.getDefiningOp());
        auto memWriteLatency = memDefiningOp.getWriteLatency();

        assert(memWriteLatency == 1 && "Expect memory write latency always be 1");
      }
    }

    return success();
  }

  LogicalResult legalizeMemoryParameterOnModule(hw::HWModuleOp &mod) {
    for (auto &stmt: mod) {
      if (auto memOp = dyn_cast<seq::FirMemOp>(stmt)) {
        if (memOp.getReadLatency() != 0 || memOp.getWriteLatency() != 1) {
          LLVM_DEBUG(
            llvm::dbgs() << "Rewrite memory decl ";
            memOp->getLoc()->print(llvm::dbgs());
            memOp.print(llvm::dbgs());
            llvm::dbgs() << "\n"
            );

          // OpBuilder builder(memOp);
          // builder.setInsertionPointAfter(memOp);
          // IRRewriter rewriter(builder);

          memOp.setReadLatency(0);
          memOp.setWriteLatency(1);

        }
      }
    }

    return success();
  }



  LogicalResult runOnModule(hw::HWModuleOp mod) {
    auto ret = pipeliningMemDelayOnModule(mod);
    if (failed(ret)) return ret;

    return legalizeMemoryParameterOnModule(mod);
  }

  void runOnOperation() final {
    auto mod = getOperation();

    SmallVector<hw::HWModuleOp> modulesToProcess;

    for(auto & inner: mod.getOps()) {
      if(auto mod = dyn_cast<hw::HWModuleOp>(&inner)) {
        modulesToProcess.push_back(mod);
      }
    }

    // // Sequential
    // for (auto mod: modulesToProcess) {
    //   auto ret = runOnModule(mod);
    //   if (failed(ret)) return signalPassFailure();
    // }

    auto result = mlir::failableParallelForEach(&getContext(), modulesToProcess.begin(), modulesToProcess.end(), [&](auto mod) {
      return runOnModule(mod);
    });
    if (failed(result)) return signalPassFailure();
  }
};

std::unique_ptr<mlir::Pass> toucan::createExpandMemoryDelayPass() {
  return std::make_unique<ExpandMemoryDelayPass>();
}

