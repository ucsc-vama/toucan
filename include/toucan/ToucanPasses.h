#pragma once

#include "mlir/Pass/Pass.h"
#include "mlir/Pass/PassRegistry.h"
#include <memory>
#include <optional>

namespace toucan {

#define GEN_PASS_DECL
#include "toucan/ToucanPasses.h.inc"

std::unique_ptr<mlir::Pass> createRemoveOMPass();

#define GEN_PASS_REGISTRATION
#include "toucan/ToucanPasses.h.inc"

}
