#include "circt/Dialect/HW/HWOpInterfaces.h"
#include "circt/Dialect/HW/HWOps.h"
#include "circt/Dialect/HW/HWInstanceGraph.h"
#include "circt/Support/Namespace.h"

#include "mlir/IR/Builders.h"
#include "mlir/IR/BuiltinAttributes.h"
#include "mlir/IR/BuiltinOps.h"
#include "mlir/IR/BuiltinTypes.h"
#include "mlir/IR/IRMapping.h"
#include "mlir/IR/MLIRContext.h"
#include "mlir/Support/LLVM.h"
#include "mlir/Support/LogicalResult.h"
#include "toucan/ToucanUtils.h"

#include "llvm/Support/Casting.h"
#include "llvm/Support/Debug.h"
#include "llvm/Support/GraphWriter.h"
#include "llvm/Support/raw_ostream.h"
#include "llvm/ADT/PostOrderIterator.h"

#define GEN_PASS_DEF_FLATTEN
#include "toucan/ToucanPassCommon.h"

using namespace circt;
using namespace mlir;
using namespace llvm;
using namespace toucan;

// TODO: Inline external module

struct FlattenPass : toucan::impl::FlattenBase<FlattenPass> {
  using toucan::impl::FlattenBase<FlattenPass>::FlattenBase;

  static void flattenName(MLIRContext *ctx, Operation *op, StringRef prefix) {
    if(op->hasAttr("sv.namehint")) {
      auto attr = op->getAttrOfType<StringAttr>("sv.namehint");
      auto newName = prefix + "." + attr.getValue();
      // Here, we allow name collision, as splitted signals share same name
      op->setAttr("sv.namehint", StringAttr::get(ctx, newName));
    }
  }

  static LogicalResult inlineExternalModule(hw::HWModuleOp parent, hw::InstanceOp inst, hw::HWModuleExternOp innerModule) {
    auto instName = inst.getInstanceName();
    // Treat each signal as a register

    // TODO

    return failure();
  }

  static LogicalResult inlineOne(hw::HWModuleOp parent, hw::InstanceOp inst, hw::HWModuleOp innerModule) {
    auto instName = inst.getInstanceName();

    auto builder = OpBuilder(parent.getBodyBlock(), inst->getIterator());
    auto bodyBlock = innerModule.getBodyBlock();

    IRMapping mapping;

    SmallVector<Operation*> cloneOps;
    cloneOps.reserve(bodyBlock->getOperations().size());

    // Map all input signals
    mapping.map(bodyBlock->getArguments(), inst->getOperands());

    // clone all ops
    bodyBlock->walk([&](Operation *op) {
      if (auto outputOp = dyn_cast<hw::OutputOp>(op)) {
        mapping.map(inst.getResults(), outputOp.getOutputs());
      } else {
        // clone
        auto clone = op->clone();
        mapping.map(op->getResults(), clone->getResults());
        cloneOps.push_back(clone);
      }
    });

    auto getMap = [&](mlir::Value x) {
      while(mapping.contains(x)) {
        x = mapping.lookup(x);
      }
      return x;
    };

    // rename
    for (auto val: bodyBlock->getArguments()) {
      auto argId = val.getArgNumber();
      auto argName = innerModule.getArgName(argId).getValue();
      auto newVal = getMap(val);
      if (auto valDefOp = newVal.getDefiningOp()) {
        // Add namehint
        auto newName = instName + "." + argName;
        auto namehint = builder.getStringAttr(newName);
        setSVNameHintAttr(valDefOp, namehint);
        setIOSignalMarker(valDefOp);
      }
    }
    for (auto val: inst.getResults()) {
      auto argId = val.getResultNumber();
      auto argName = inst.getResultName(argId).getValue();
      auto newVal = getMap(val);
      if (auto valDefOp = newVal.getDefiningOp()) {
        // Add namehint
        auto newName = instName + "." + argName;
        auto namehint = builder.getStringAttr(newName);
        setSVNameHintAttr(valDefOp, namehint);
        setIOSignalMarker(valDefOp);
      }
    }

    // flatten
    auto context = inst.getContext();
    for(auto op: cloneOps) {
      if (!hasIOSignalMarker(op)) {
        flattenName(context, op, instName);
      }
      for(auto& opOperand: op->getOpOperands()) {
        opOperand.set(getMap(opOperand.get()));
      }
      builder.insert(op);
    }
    for(auto outval: inst.getResults()) {
      outval.replaceAllUsesWith(getMap(outval));
    }
    assert(inst->getUses().empty());
    inst->erase();

    return success();
  }


  static void inlineChild(hw::HWModuleOp mod, hw::InstanceGraph &graph) {
    auto context = mod->getContext();
    auto block = mod.getBodyBlock();
    auto insts = to_vector(block->getOps<hw::InstanceOp>());
    for(auto inst: insts) {
      auto innerNode = graph.lookup(StringAttr::get(context, inst.getModuleName()));
      auto modlike = innerNode->getModule();
      if(auto modop = dyn_cast<hw::HWModuleOp>(modlike.getOperation())) {
        inlineOne(mod, inst, modop);
      }
    }
  }
  static void resetNames(MLIRContext * ctx, Operation * op) {
    Namespace ns;
    op->walk([&](Operation * op){
      if(op->hasAttr("sv.namehint")) {
        auto attr = op->getAttrOfType<StringAttr>("sv.namehint");
        auto name = ns.newName(attr.getValue());
        if(name != attr.str()) {
          op->setAttr("sv.namehint", StringAttr::get(ctx, name));
        }
      }
    });
  }
  void runOnOperation() final {
    auto modlist = getOperation();
    hw::HWModuleOp topModule = nullptr;
    SmallVector<hw::HWModuleOp> nodesToDelete;
    modlist->walk([&](hw::HWModuleOp op) {
      if(op.isPrivate()) {
        nodesToDelete.push_back(op);
      }
      else if(topModule) {
        modlist->emitError() << "multiple top module detected";
        return signalPassFailure();
      }
      else {
        topModule = op;
      }
    });
    auto & instGraph = getAnalysis<hw::InstanceGraph>();
    for(auto modNode: post_order(&instGraph)) {
      auto modop = modNode->getModule().getOperation();
      if(!modop) continue;
      if(auto mod = dyn_cast<hw::HWModuleOp>(modop)) {
        // TODO: parallel
        inlineChild(mod, instGraph);
      }
    }
    for(auto mod: nodesToDelete) {
      mod.erase();
    }
    resetNames(&getContext(), getOperation());
    modlist->walk([&](hw::HWModuleOp mod) {
      for(auto & op: mod.getOps()) {
        numEdges += op.getNumOperands();
        numOps += 1;
      }
    });
  }
};

std::unique_ptr<mlir::Pass> toucan::createFlattenPass() {
  return std::make_unique<FlattenPass>();
}