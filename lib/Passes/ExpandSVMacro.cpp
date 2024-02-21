#include "circt/Dialect/HW/HWOps.h"
#include "circt/Dialect/SV/SVDialect.h"
#include "circt/Dialect/SV/SVOps.h"
#include "circt/Support/LLVM.h"

#include "toucan/ToucanDialect.h"
#include "toucan/ToucanOps.h"

#include "mlir/IR/Builders.h"
#include "mlir/IR/BuiltinAttributes.h"
#include "mlir/IR/BuiltinOps.h"
#include "mlir/IR/BuiltinTypes.h"
#include "mlir/IR/Diagnostics.h"
#include "mlir/IR/OpImplementation.h"
#include "mlir/IR/Operation.h"
#include "mlir/IR/Visitors.h"
#include "mlir/Support/LLVM.h"
#include "mlir/IR/Threading.h"
#include "mlir/Support/LogicalResult.h"

#include "llvm/ADT/STLExtras.h"
#include "llvm/ADT/SmallVector.h"
#include "llvm/ADT/StringMap.h"

#include "llvm/ADT/StringRef.h"
#include "llvm/ADT/iterator_range.h"
#include "llvm/Support/Casting.h"
#include "llvm/Support/Debug.h"
#include "llvm/Support/raw_ostream.h"
#include "llvm/Support/Format.h"

#include <_types/_uint64_t.h>
#include <memory>
#include <atomic>



#define GEN_PASS_DEF_EXPANDSVMACRO
#include "toucan/ToucanPassCommon.h"

#include "toucan/ToucanUtils.h"

using namespace toucan;
using namespace circt;
using namespace mlir;
using namespace llvm;

#define DEBUG_TYPE "ExpandSVMacroPass"

struct SVMacroContext {
  StringMap<StringRef> macroTable;
  SmallVector<Operation*> toRemove;

  void updateWithGlobalTable(StringMap<StringRef> &globalTable) {
    for (auto key: globalTable.keys()) {
      macroTable.insert_or_assign(key, globalTable.lookup(key));
    }
  }
};


static StringRef getEmptyStringRef() {
  return "";
}


std::atomic<uint64_t> removedOpsInModules;
std::atomic<uint64_t> printOpsInModules;
std::atomic<uint64_t> stopOpsInModules;

struct ExpandSVMacro : toucan::impl::ExpandSVMacroBase<ExpandSVMacro> {
  using ExpandSVMacroBase<ExpandSVMacro>::ExpandSVMacroBase;

  SVMacroContext globalMC;


  static void removeOps(SmallVector<Operation*> &toRemove) {
    for(auto op: llvm::reverse(toRemove)) {
      op->erase();
    }
  }

  void markAsUnused(Operation *op) {
    op->setAttr("unused", BoolAttr::get(&getContext(), true));
  }

  bool isUnused(Operation *op) {
    if (op->hasAttr("unused")) {
      return op->getAttrOfType<BoolAttr>("unused").getValue();
    }
    return false;
  }

  LogicalResult moveBlockStmtsOutsideThenTraverse(SVMacroContext &mc, Operation *currentOp, Block *block) {
    SmallVector<Operation*> newOps;

    // Insert after currentOp
    OpBuilder builder(currentOp);
    for (mlir::Operation &op : *block) {
      auto newOp = op.clone();
      builder.insert(newOp);
      op.replaceAllUsesWith(newOp);

      newOps.push_back(newOp);
      markAsUnused(&op);
    }
    markAsUnused(currentOp);

    for (mlir::Operation *op : newOps) {
      if (isa_and_nonnull<sv::SVDialect>(op->getDialect())) {
        auto ret = onSVStmt(mc, op);
        if (failed(ret)) return ret;
      }
    }

    return success();
  }

  LogicalResult onIfDef(SVMacroContext& mc, sv::IfDefOp &ifDefOp) {
    auto macroName = ifDefOp.getCond().getIdent();

    auto validBlock = (mc.macroTable.contains(macroName) ? ifDefOp.getThenBlock() : ifDefOp.getElseBlock());

    return moveBlockStmtsOutsideThenTraverse(mc, ifDefOp, validBlock);
  }

  LogicalResult onMacroRef(SVMacroContext& mc, sv::MacroRefExprOp &macroRefOp) {
    auto macroName = macroRefOp.getMacroName();
    auto macroValue = mc.macroTable.lookup(macroName);

    if (macroValue.empty()) {
      macroRefOp->emitError() << "Here: Referenced macro doesn't exist " << macroName;
      return failure();
    }
    LLVM_DEBUG(llvm::dbgs() << "Macro " << macroName << " has value " << macroValue << "\n");

    int64_t macroIntValue;
    auto failed = macroValue.empty() || macroValue.getAsInteger(10, macroIntValue);

    if (failed) {
      macroRefOp.emitError() << "Macro " << macroName << " has a value [ " << macroValue.str() << " ] and cannot be converted to int";
      return failure();
    }
    
    auto builder = OpBuilder(macroRefOp);
    auto constantOp = builder.create<hw::ConstantOp>(macroRefOp.getLoc(),
                                                macroRefOp.getType(),
                                                builder.getIntegerAttr(macroRefOp.getType(), macroIntValue));
    // builder.insert(constantOp); 
    macroRefOp.replaceAllUsesWith(constantOp.getResult());

    markAsUnused(macroRefOp);
    return success();
  }


  LogicalResult onAlways(SVMacroContext& mc, sv::AlwaysOp &alwaysOp) {
    // Note: Drop all clocks, we don't support multiple clock domain. Simply pull out body statements.
    auto block = alwaysOp.getBodyBlock();
    auto ret = moveBlockStmtsOutsideThenTraverse(mc, alwaysOp, block);
    return ret;
  }

  LogicalResult onIfOp(SVMacroContext& mc, sv::IfOp &ifOp) {
    // Note: Only consideres assertion and stop
    auto enSignal = ifOp.getCond();

    if (ifOp.hasElse()) {
      ifOp->emitError() << "sv.if with else is not supported";
      return failure();
    }
    
    auto thenBlock = ifOp.getThenBlock();
    if (thenBlock->getOperations().size() != 1) {
      ifOp->emitError() << "sv.if is not fully supported";
      return failure();
    }

    auto stmt = thenBlock->getOperations().begin();
    OpBuilder builder(ifOp);

    if (auto fwriteOp = dyn_cast<sv::FWriteOp>(stmt)) {
      auto fdOp = fwriteOp.getFd().getDefiningOp();
      auto fdConstOp = dyn_cast<hw::ConstantOp>(fdOp);
      if (fdConstOp == nullptr) {
        fwriteOp->emitError() << "fd must be a constant";
        return failure();
      }
      // Here ignore fd values, simply create a printOp
      // auto fdVal = fdConstOp.getValue().getRawData();

      auto substitutions = fwriteOp.getSubstitutions();
      if (substitutions.size() > 0) {
        fwriteOp->emitWarning() << "Substitutions are not supported. Will only print format string";
      }

      auto printOpMsg = fwriteOp.getFormatString();
      auto printOp = builder.create<toucan::PrintOp>(ifOp->getLoc(),
                                                              enSignal, printOpMsg);
      // builder.insert(printOp);
      ifOp->replaceAllUsesWith(printOp);
      printOpsInModules ++;
    } else if (auto fatalOp = dyn_cast<sv::FatalOp>(stmt)) {
      // ignore message and verbosity
      auto stopOp = builder.create<toucan::StopOp>(ifOp->getLoc(), enSignal);
      
      ifOp->replaceAllUsesWith(stopOp);
      stopOpsInModules ++;
    } else {
      stmt->emitError() << "Unsupported Op";
      return failure();
    }

    markAsUnused(ifOp);
    return success();
  }


  LogicalResult onMacroDeclOp(SVMacroContext& mc, sv::MacroDeclOp &declOp) {
    auto macroName = declOp.getSymNameAttr().getValue();
    // If this macro has never been defined, define it as empty string
    if (!mc.macroTable.contains(macroName)) {
      mc.macroTable.insert_or_assign(macroName, getEmptyStringRef());
    }
    markAsUnused(declOp);
    return success();
  }

  LogicalResult onMacroDefOp(SVMacroContext& mc, sv::MacroDefOp &defOp) {
    auto macroName = defOp.getMacroName();
    auto macroValue = defOp.getFormatString();

    if (macroValue.str().find('@') != std::string::npos) {
      defOp->emitWarning() << "System Verilog macro formatting is not supported. This may lead to an error";
    }
    mc.macroTable.insert_or_assign(macroName, macroValue);
    markAsUnused(defOp);
    return success();
  }

  LogicalResult onSVStmt(SVMacroContext& mc, Operation *op) {
    if (isUnused(op)) {
      LLVM_DEBUG(llvm::dbgs() << "Traversing into unused op.\n");
      return success();
    }

    if (auto declOp = dyn_cast<sv::MacroDeclOp>(op)) {
      return onMacroDeclOp(mc, declOp);
    } else if (auto defOp = dyn_cast<sv::MacroDefOp>(op)) {
      return onMacroDefOp(mc, defOp);
    } else if (auto ifDefOp = dyn_cast<sv::IfDefOp>(op)) {
      return onIfDef(mc, ifDefOp);
    } else if (auto macroRefOp = dyn_cast<sv::MacroRefExprOp>(op)) {
      return onMacroRef(mc, macroRefOp);
    } else if (auto alwaysOp = dyn_cast<sv::AlwaysOp>(op)) {
      return onAlways(mc, alwaysOp);
    } else if (auto ifOp = dyn_cast<sv::IfOp>(op)) {
      return onIfOp(mc, ifOp);
    };

    // We don't handle all SV Ops. Other SV Ops are not needed, if this mlir file is lowered from FIRRTL
    return success();
  }

  LogicalResult runOnModule(hw::HWModuleOp mod) {
    SVMacroContext mc;
    mc.updateWithGlobalTable(globalMC.macroTable);

    for(auto &op: mod.getOps()) {
      if (isa_and_nonnull<sv::SVDialect>(op.getDialect())) {
        auto ret = onSVStmt(mc, &op);
        if (failed(ret))
        return failure();
      }
    }

    for(auto &op: mod.getOps()) {
      if (isUnused(&op)) {
        mc.toRemove.push_back(&op);
      }
    }
    removeOps(mc.toRemove);
    removedOpsInModules += mc.toRemove.size();

    return success();
  }

  void runOnOperation() final {
    auto mod = getOperation();

    SmallVector<hw::HWModuleOp> modulesToProcess;

    removedOpsInModules = 0;
    printOpsInModules = 0;
    stopOpsInModules = 0;

    for(auto & inner: mod.getOps()) {
      if(auto mod = dyn_cast<hw::HWModuleOp>(&inner)) {
        modulesToProcess.push_back(mod);
      } else if(isa_and_nonnull<sv::SVDialect>(inner.getDialect())) {
        auto ret = onSVStmt(globalMC, &inner);
        if (failed(ret)) return signalPassFailure();
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

    for(auto & inner: mod.getOps()) {
      if(isUnused(&inner)) {
        globalMC.toRemove.push_back(&inner);
      }
    }

    removeOps(globalMC.toRemove);
    removedOpsInModules += globalMC.toRemove.size();

    removedOps = removedOpsInModules;
    printOps = printOpsInModules;
    stopOps = stopOpsInModules;
  }
};

std::unique_ptr<mlir::Pass> toucan::createExpandSVMacroPass() {
  return std::make_unique<ExpandSVMacro>();
}