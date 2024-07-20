#include "ToucanGPUSim/ToucanGPUGenDataTypes.h"
#include "llvm/Support/ErrorHandling.h"

using namespace toucanGPUSim;

void SimDesignInfo::Init() {
  // copy const pool
  llvm_unreachable("This procedure should not call from toucan");
}

void SimDesignInfo::Randomize(uint32_t seed, SimDebugInfo &symbols) {
  // Randomize reg and mem
  llvm_unreachable("This procedure should not call from toucan");
}

