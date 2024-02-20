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
std::unique_ptr<mlir::Pass> createFactorSVPass();
std::unique_ptr<mlir::Pass> createExpandMemoryDelayPass();
std::unique_ptr<mlir::Pass> createFactorArrayGetPass();
std::unique_ptr<mlir::Pass> createSplitFirMemRWPortsPass();
std::unique_ptr<mlir::Pass> createReplaceAsyncResetRegsPass();
std::unique_ptr<mlir::Pass> createSplitRegistersPass();
std::unique_ptr<mlir::Pass> createRemoveMemMaskPass();

std::unique_ptr<mlir::Pass> createParallelCanonicalizerPass();
std::unique_ptr<mlir::Pass> createParallelCanonicalizerPass(const mlir::GreedyRewriteConfig &config);

std::unique_ptr<mlir::Pass> createLowerRegMemTo4BPass();
std::unique_ptr<mlir::Pass> createEnsureNoClockOpPass();
std::unique_ptr<mlir::Pass> createLowerCombPreProcessPass();
std::unique_ptr<mlir::Pass> createLowerCombTo4B_1Pass();
std::unique_ptr<mlir::Pass> createLowerCombTo4B_2Pass();
std::unique_ptr<mlir::Pass> createLowerCombTo4B_3Pass();

std::unique_ptr<mlir::Pass> createFlattenPass();
std::unique_ptr<mlir::Pass> createRemoveSeqPass();
std::unique_ptr<mlir::Pass> createFactorConcatExtractPass();

#define GEN_PASS_REGISTRATION
#include "toucan/ToucanPasses.h.inc"

}
