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
std::unique_ptr<mlir::Pass> createFactorArrayGetMuxPass();
std::unique_ptr<mlir::Pass> createSplitFirMemRWPortsPass();
std::unique_ptr<mlir::Pass> createReplaceAsyncResetRegsPass();
std::unique_ptr<mlir::Pass> createSplitRegistersPass();
std::unique_ptr<mlir::Pass> createRemoveMemMaskPass();

std::unique_ptr<mlir::Pass> createCanonicalizerPass();
std::unique_ptr<mlir::Pass> createCanonicalizerPass(const mlir::GreedyRewriteConfig &config);

std::unique_ptr<mlir::Pass> createLowerRegMemTo4BPass();
std::unique_ptr<mlir::Pass> createEnsureNoClockOpPass();
std::unique_ptr<mlir::Pass> createLowerCombPreProcessPass();
std::unique_ptr<mlir::Pass> createLowerCombTo4B_ShortReplicateOpPass();
std::unique_ptr<mlir::Pass> createLowerCombTo4B_1Pass();
std::unique_ptr<mlir::Pass> createLowerCombTo4B_2Pass();
std::unique_ptr<mlir::Pass> createLowerCombTo4B_3Pass();

std::unique_ptr<mlir::Pass> createFlattenPass();
std::unique_ptr<mlir::Pass> createRemoveSeqPass();
std::unique_ptr<mlir::Pass> createFactorConcatExtractPass();
std::unique_ptr<mlir::Pass> createFlatDCEPass();
std::unique_ptr<mlir::Pass> createEnsureToucanOnlyPass();

std::unique_ptr<mlir::Pass> createCPUSingleThreadCodeGenPass(CPUSingleThreadCodeGenOptions options);

#define GEN_PASS_REGISTRATION
#include "toucan/ToucanPasses.h.inc"

}
