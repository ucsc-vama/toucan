#pragma once

#include "mlir/Pass/Pass.h"
#include "mlir/Pass/PassRegistry.h"
#include <memory>
#include <optional>

#include "mlir/Transforms/GreedyPatternRewriteDriver.h"

namespace toucan {

#define GEN_PASS_DECL
#include "toucan/ToucanPasses.h.inc"

std::unique_ptr<mlir::Pass> createRemoveSVnOMPass();
std::unique_ptr<mlir::Pass> createExpandSVMacroPass();
std::unique_ptr<mlir::Pass> createExpandMemoryDelayPass();
// std::unique_ptr<mlir::Pass> createRemoveMemReadEnPass();
std::unique_ptr<mlir::Pass> createFactorArrayGetMuxPass();
std::unique_ptr<mlir::Pass> createFactorHWMiscPass();
std::unique_ptr<mlir::Pass> createSplitFirMemRWPortsPass();
std::unique_ptr<mlir::Pass> createSplitRegistersPass();
std::unique_ptr<mlir::Pass> createRemoveMemMaskPass();

std::unique_ptr<mlir::Pass> createToucanCanonicalizerPass();
std::unique_ptr<mlir::Pass> createToucanCanonicalizerPass(const mlir::GreedyRewriteConfig &config);

std::unique_ptr<mlir::Pass> createLowerRegMemTo4BPass();
std::unique_ptr<mlir::Pass> createEnsureNoClockOpPass();
std::unique_ptr<mlir::Pass> createFactorBinaryOpPass();
std::unique_ptr<mlir::Pass> createLowerHWVectorTo4BPass();
std::unique_ptr<mlir::Pass> createLowerCombTo4B_1Pass();
std::unique_ptr<mlir::Pass> createLowerCombTo4B_2Pass();
std::unique_ptr<mlir::Pass> createLowerCombTo4B_3Pass();

std::unique_ptr<mlir::Pass> createFlattenPass();
std::unique_ptr<mlir::Pass> createRemoveSeqPass();
std::unique_ptr<mlir::Pass> createFactorConcatExtractPass();
std::unique_ptr<mlir::Pass> createEnsureToucanOnlyPass();
std::unique_ptr<mlir::Pass> createDeduplicateRegistersPass();
std::unique_ptr<mlir::Pass> createRemoveConstRegsPass();
std::unique_ptr<mlir::Pass> createMergeConstPass();
std::unique_ptr<mlir::Pass> createBreakPinnedValueToOutputConnectionPass();

// std::unique_ptr<mlir::Pass> createCPUSingleThreadCodeGenPass(CPUSingleThreadCodeGenOptions options);
std::unique_ptr<mlir::Pass> createGPUCodeGenPass(GPUCodeGenOptions options);

#define GEN_PASS_REGISTRATION
#include "toucan/ToucanPasses.h.inc"

}
