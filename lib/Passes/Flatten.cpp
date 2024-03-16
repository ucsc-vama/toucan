#include "circt/Dialect/Comb/CombOps.h"
#include "circt/Dialect/HW/HWOpInterfaces.h"
#include "circt/Dialect/HW/HWOps.h"
#include "circt/Dialect/HW/HWInstanceGraph.h"
#include "circt/Dialect/HW/HWTypes.h"
#include "circt/Dialect/Seq/SeqOps.h"
#include "circt/Dialect/Seq/SeqTypes.h"
#include "circt/Support/InstanceGraph.h"
#include "circt/Support/LLVM.h"
#include "circt/Support/Namespace.h"

#include "mlir/IR/Builders.h"
#include "mlir/IR/BuiltinAttributes.h"
#include "mlir/IR/BuiltinOps.h"
#include "mlir/IR/BuiltinTypes.h"
#include "mlir/IR/IRMapping.h"
#include "mlir/IR/MLIRContext.h"
#include "mlir/IR/PatternMatch.h"
#include "mlir/IR/Threading.h"
#include "mlir/Support/LLVM.h"
#include "mlir/Support/LogicalResult.h"
#include "toucan/ToucanOps.h"
#include "toucan/ToucanUtils.h"

#include "llvm/ADT/SmallVector.h"
#include "llvm/Support/Casting.h"
#include "llvm/Support/Debug.h"
#include "llvm/Support/GraphWriter.h"
#include "llvm/Support/raw_ostream.h"
#include "llvm/ADT/PostOrderIterator.h"

#define GEN_PASS_DEF_FLATTEN
#include "toucan/ToucanPassCommon.h"

#include <chrono>

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

  static Value createDummpyClockValue(OpBuilder &builder, Location loc) {
    auto constOneOp = builder.create<hw::ConstantOp>(loc, builder.getI1Type(), 1);
    auto asClockOp = builder.create<seq::ToClockOp>(loc, constOneOp.getResult());
    return asClockOp.getResult();
  }

  static Value createRegAndReturnSingleReadValue(size_t valueWidth, Location loc, OpBuilder &builder, StringAttr valueNewNameAttr) {
    SmallVector<Value> chunkValues;

    for (auto [chunkId, chunkSize]: split_signal_4B(valueWidth)) {
      auto regType = builder.getIntegerType(chunkSize);
      auto regOp = builder.create<toucan::DefRegOp>(loc, regType);

      auto regReadOp = builder.create<toucan::RegReadOp>(loc, regOp.getHandle());


      auto idAttr = builder.getI32IntegerAttr(chunkId);

      auto regOperation = regOp.getOperation();
      auto regReadOperation = regReadOp.getOperation();

      setSVNameHintAttr(regOperation, valueNewNameAttr);
      setSignalFragmentIDAttr(regOperation, idAttr);
      setIOSignalMarker(regOperation);

      setSVNameHintAttr(regReadOperation, valueNewNameAttr);
      setSignalFragmentIDAttr(regReadOperation, idAttr);
      setIOSignalMarker(regReadOperation);

      chunkValues.push_back(regReadOp.getResult());
    }
    if (chunkValues.size() > 1) {
      // merge segments
      auto concatOp = builder.create<comb::ConcatOp>(loc, chunkValues);
      return concatOp.getResult();
    }
    return chunkValues[0];
  }

  static void createRegAndWrite(Operation *op, Value &inputVal, RewriterBase &rewriter, StringAttr &valueNewNameAttr) {
    auto splitedVals = split_value_4B(op, inputVal, rewriter);
    for (size_t chunkId = 0; chunkId < splitedVals.size(); chunkId++) {
      auto inputChunk = splitedVals[splitedVals.size() - 1 - chunkId];
      auto inputChunkWidth = hw::getBitWidth(inputChunk.getType());
      auto regType = rewriter.getIntegerType(inputChunkWidth);

      auto regOp = rewriter.create<toucan::DefRegOp>(op->getLoc(), regType);

      rewriter.create<toucan::RegWriteOp>(op->getLoc(), inputChunk, regOp.getHandle());

      auto idAttr = rewriter.getI32IntegerAttr(chunkId);

      auto regOperation = regOp.getOperation();

      setSVNameHintAttr(regOperation, valueNewNameAttr);
      setSignalFragmentIDAttr(regOperation, idAttr);
      setIOSignalMarker(regOperation);

    }
  }


  static LogicalResult inlineExternalModule(hw::HWModuleOp parent, hw::InstanceOp inst, hw::HWModuleExternOp innerModule) {
    auto instName = inst.getInstanceName();

    auto builder = OpBuilder(parent.getBodyBlock(), inst->getIterator());
    IRRewriter rewriter(builder);
    auto context = inst->getContext();

    IRMapping mapping;


    // // Map all input signals
    // mapping.map(bodyBlock->getArguments(), inst->getOperands());
    for (auto [inputVal, valName]: zip(inst.getInputs(), inst.getArgNames()) ) {
      // input values are transformed to a register with writer only
      auto valueNewName = instName + "." + valName.cast<StringAttr>().getValue();
      auto valueNewNameAttr = StringAttr::get(context, valueNewName);

      if (isa<seq::ClockType>(inputVal.getType()) || hw::getBitWidth(inputVal.getType()) <= 0) {
        continue;
      }

      createRegAndWrite(inst, inputVal, rewriter, valueNewNameAttr);
    }


    // // Map all output signals
    for (auto [outputVal, valName]: zip(inst.getResults(), inst.getResultNames()) ) {

      // Dirty hack for seq.clock
      if (isa<seq::ClockType>(outputVal.getType())) {
        auto dummyClockVal = createDummpyClockValue(builder, inst->getLoc());

        mapping.map(outputVal, dummyClockVal);
      } else {
        // input values are transformed to a register with writer only
        auto valueWidth = hw::getBitWidth(outputVal.getType());
        auto valueNewName = instName + "." + valName.cast<StringAttr>().getValue();
        auto valueNewNameAttr = StringAttr::get(context, valueNewName);

        auto regVal = createRegAndReturnSingleReadValue(valueWidth, inst->getLoc(), rewriter, valueNewNameAttr);

        mapping.map(outputVal, regVal);
      }
    }


    auto getMap = [&](mlir::Value x) {
      while(mapping.contains(x)) {
        x = mapping.lookup(x);
      }
      return x;
    };

    // flatten
    for(auto outval: inst.getResults()) {
      outval.replaceAllUsesWith(getMap(outval));
    }
    assert(inst->getUses().empty());
    inst->erase();

    return success();
  }


  static LogicalResult inlineModule(hw::HWModuleOp parent, hw::InstanceOp inst, hw::HWModuleOp innerModule) {
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
      flattenName(context, op, instName);
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


  static LogicalResult inlineChild(hw::HWModuleOp mod, hw::InstanceGraph &graph) {
    auto context = mod->getContext();
    auto block = mod.getBodyBlock();
    auto insts = to_vector(block->getOps<hw::InstanceOp>());

    for(auto inst: insts) {
      auto innerNode = graph.lookup(StringAttr::get(context, inst.getModuleName()));
      auto modlike = innerNode->getModule();
      if (auto modop = dyn_cast<hw::HWModuleOp>(modlike.getOperation())) {
        if (!succeeded(inlineModule(mod, inst, modop))) return failure();
      } else if (auto extmodOp = dyn_cast<hw::HWModuleExternOp>(modlike.getOperation())) {
        if (!succeeded(inlineExternalModule(mod, inst, extmodOp))) return failure();
      }
    }
    return success();
  }

  static LogicalResult flattenTopModule(hw::HWModuleOp topMod) {
    // Remove module IO, redfine it as register
    auto bodyBlock = topMod.getBodyBlock();
    OpBuilder builder(bodyBlock, bodyBlock->begin());
    auto loc = bodyBlock->begin()->getLoc();

    SmallVector<unsigned int> inPortToErase;

    for (auto [argVal, portInfo]: zip(bodyBlock->getArguments(), topMod.getPortList())) {
      if (isa<seq::ClockType>(argVal.getType())) {
        auto dummyClockVal = createDummpyClockValue(builder, loc);

        argVal.replaceAllUsesWith(dummyClockVal);
      } else {
        // Not clock
        auto argName = portInfo.getName();
        auto valueWidth = hw::getBitWidth(argVal.getType());
        auto newNameAttr = builder.getStringAttr(argName);

        auto newVal = createRegAndReturnSingleReadValue(valueWidth, loc, builder, newNameAttr);

        argVal.replaceAllUsesWith(newVal);
      }
      inPortToErase.push_back(argVal.getArgNumber());
    }

    // Transform output to regwrite
    bodyBlock->walk([&](hw::OutputOp outputOp) {

      IRRewriter rewriter(builder);
      rewriter.setInsertionPoint(outputOp);

      for (auto [outVal, valName]: zip(outputOp.getOutputs(), topMod.getOutputNames())) {
        auto argName = valName.cast<StringAttr>().getValue();
        auto valueNewNameAttr = rewriter.getStringAttr(argName);

        createRegAndWrite(outputOp.getOperation(), outVal, rewriter, valueNewNameAttr);
      }
    });

    SmallVector<Operation*> cloneOps;
    cloneOps.reserve(bodyBlock->getOperations().size());

    IRMapping mapping;
    auto getMap = [&](mlir::Value x) {
      while(mapping.contains(x)) {
        x = mapping.lookup(x);
      }
      return x;
    };

    // clone all ops
    bodyBlock->walk([&](Operation *op) {
      if (!isa<hw::OutputOp>(op)) {
        // clone
        auto clone = op->clone();
        mapping.map(op->getResults(), clone->getResults());
        cloneOps.push_back(clone);
      }
    });

    // flatten
    builder.setInsertionPoint(topMod);

    auto context = topMod.getContext();
    auto modName = topMod.getSymName();
    for(auto op: cloneOps) {
      flattenName(context, op, modName);
      for(auto& opOperand: op->getOpOperands()) {
        opOperand.set(getMap(opOperand.get()));
      }
      builder.insert(op);
    }
    topMod.erase();


    return success();
  }

  void runOnOperation() final {
    auto modlist = getOperation();

    modlist->walk([&](Operation *op) {
        numEdgesBeforeFlatten += op->getNumOperands();
        numOpsBeforeFlatten += 1;
    });

    hw::HWModuleOp topModule = nullptr;
    SmallVector<hw::HWModuleExternOp> externModules;
    SmallVector<hw::HWModuleOp> nodesToDelete;

    modlist->walk([&](hw::HWModuleExternOp op) {
      externModules.push_back(op);
    });

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

    if (topModule == nullptr) {
      modlist->emitError() << "No top module detected. Is it already flatten?";
      return signalPassFailure();
    }

    auto & instGraph = getAnalysis<hw::InstanceGraph>();
    
    for(auto modNode: post_order(&instGraph)) {
      auto modop = modNode->getModule().getOperation();
      if(!modop) continue;
      if(auto mod = dyn_cast<hw::HWModuleOp>(modop)) {
        // TODO: parallel
        if (failed(inlineChild(mod, instGraph))) return signalPassFailure();
      }
    }
    for(auto mod: nodesToDelete) {
      mod.erase();
    }
    for (auto extmod: externModules) {
      extmod.erase();
    }

    if (failed(flattenTopModule(topModule))) return signalPassFailure();

    modlist->walk([&](Operation *op) {
        numEdgesAfterFlatten += op->getNumOperands();
        numOpsAfterFlatten += 1;
    });
  }
};

std::unique_ptr<mlir::Pass> toucan::createFlattenPass() {
  return std::make_unique<FlattenPass>();
}