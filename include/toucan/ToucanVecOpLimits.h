#pragma once

// Note: For VecLogic and VecArith operation, input vector width is limited.
// Set to 128 as it's the largest datatype that CUDA supports (by __int128)
#define TOUCAN_VEC_OP_MAX_WIDTH 128