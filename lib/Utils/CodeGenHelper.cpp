
#include <bit>
#include <utility>

#include "toucan/CodeGenCommon.h"

#include "toucan/ToucanAttributes.h"
#include "toucan/ToucanUtils.h"
#include "toucan/ToucanOps.h"
#include "llvm/Support/Debug.h"


using namespace mlir;
using namespace toucan;

void CodeGenHelper::populateLUT_Nop() {
  SmallVector<uint8_t> lut;

  for (uint8_t i = 0; i <= 0xF; i++) {
    lut.push_back(i);
  }

  lutPos[static_cast<uint8_t>(LUTOpName::LUT_Nop)] = lutContent.size();
  lutContent.insert(lutContent.end(), lut.begin(), lut.end());
}


void CodeGenHelper::populateLUT_Rep1b() {
  // simple lut, 0 -> 0000, 1 -> 1111
  SmallVector<uint8_t> lut;
  lut.push_back(0);
  lut.push_back(0xF);
  
  lutPos[static_cast<uint8_t>(LUTOpName::LUT_Rep1b)] = lutContent.size();
  lutContent.insert(lutContent.end(), lut.begin(), lut.end());
}

void CodeGenHelper::populateLUT_XorR() {
  SmallVector<uint8_t> lut;

  for (uint8_t i = 0; i <= 0xF; i++) {
    uint8_t output = std::popcount(i) & 1;
    lut.push_back(output);
  }

  lutPos[static_cast<uint8_t>(LUTOpName::LUT_XorR)] = lutContent.size();
  lutContent.insert(lutContent.end(), lut.begin(), lut.end());
}

// 2 inputs
void CodeGenHelper::populateLUT_And() {
  SmallVector<uint8_t> lut;

  for (uint8_t i = 0; i <= 0xF; i++) {
    for (uint8_t j = 0; j <= 0xF; j++) {
      uint8_t result = i & j;
      lut.push_back(result);
    }
  }

  lutPos[static_cast<uint8_t>(LUTOpName::LUT_And)] = lutContent.size();
  lutContent.insert(lutContent.end(), lut.begin(), lut.end());
}

void CodeGenHelper::populateLUT_Or() {
  SmallVector<uint8_t> lut;

  for (uint8_t i = 0; i <= 0xF; i++) {
    for (uint8_t j = 0; j <= 0xF; j++) {
      uint8_t result = i | j;
      lut.push_back(result);
    }
  }

  lutPos[static_cast<uint8_t>(LUTOpName::LUT_Or)] = lutContent.size();
  lutContent.insert(lutContent.end(), lut.begin(), lut.end());
}

void CodeGenHelper::populateLUT_Xor() {
  SmallVector<uint8_t> lut;

  for (uint8_t i = 0; i <= 0xF; i++) {
    for (uint8_t j = 0; j <= 0xF; j++) {
      uint8_t result = i ^ j;
      lut.push_back(result);
    }
  }

  lutPos[static_cast<uint8_t>(LUTOpName::LUT_Xor)] = lutContent.size();
  lutContent.insert(lutContent.end(), lut.begin(), lut.end());
}

// void CodeGenHelper::populateLUT_Shl1() {
//   SmallVector<uint8_t> lut;

//   for (uint8_t i = 0; i <= 0xF; i++) {
//     for (uint8_t j = 0; j <= 0xF; j++) {
//       uint8_t result = ((i << 1) | (j >> 3)) & 0xF;
//       lut.push_back(result);
//     }
//   }

//   lutPos[static_cast<uint8_t>(LUTOpName::LUT_Shl1)] = lutContent.size();
//   lutContent.insert(lutContent.end(), lut.begin(), lut.end());
// }

// void CodeGenHelper::populateLUT_Shl2() {
//   SmallVector<uint8_t> lut;

//   for (uint8_t i = 0; i <= 0xF; i++) {
//     for (uint8_t j = 0; j <= 0xF; j++) {
//       uint8_t result = ((i << 2) | (j >> 2)) & 0xF;
//       lut.push_back(result);
//     }
//   }

//   lutContent[LUTOpName::LUT_Shl2] = std::move(lut);
// }

// void CodeGenHelper::populateLUT_Shl3() {
//   SmallVector<uint8_t> lut;

//   for (uint8_t i = 0; i <= 0xF; i++) {
//     for (uint8_t j = 0; j <= 0xF; j++) {
//       uint8_t result = ((i << 3) | (j >> 1)) & 0xF;
//       lut.push_back(result);
//     }
//   }

//   lutContent[LUTOpName::LUT_Shl3] = std::move(lut);
// }

// void CodeGenHelper::populateLUT_Shr1() {
//   // Note: Shr1 and Shl3 has same truth table
// }

// void CodeGenHelper::populateLUT_Shr2() {
//   // Note: Shr2 and Shl2 has same truth table
// }

// void CodeGenHelper::populateLUT_Shr3() {
//   // Note: Shr3 and Shl1 has same truth table
// }

void CodeGenHelper::populateLUT_Cmp_Eq() {
  SmallVector<uint8_t> lut;

  for (uint8_t i = 0; i <= 0xF; i++) {
    for (uint8_t j = 0; j <= 0xF; j++) {
      uint8_t result = i == j;
      lut.push_back(result);
    }
  }

  lutPos[static_cast<uint8_t>(LUTOpName::LUT_Cmp_Eq)] = lutContent.size();
  lutContent.insert(lutContent.end(), lut.begin(), lut.end());
}

void CodeGenHelper::populateLUT_Mul_Hi() {
  SmallVector<uint8_t> lut;

  for (uint8_t i = 0; i <= 0xF; i++) {
    for (uint8_t j = 0; j <= 0xF; j++) {
      uint8_t result = (i * j) >> 4;
      lut.push_back(result);
    }
  }

  lutPos[static_cast<uint8_t>(LUTOpName::LUT_Mul_Hi)] = lutContent.size();
  lutContent.insert(lutContent.end(), lut.begin(), lut.end());
}

void CodeGenHelper::populateLUT_Mul_Lo() {
  SmallVector<uint8_t> lut;

  for (uint8_t i = 0; i <= 0xF; i++) {
    for (uint8_t j = 0; j <= 0xF; j++) {
      uint8_t result = (i * j) & 0xF;
      lut.push_back(result);
    }
  }

  lutPos[static_cast<uint8_t>(LUTOpName::LUT_Mul_Lo)] = lutContent.size();
  lutContent.insert(lutContent.end(), lut.begin(), lut.end());
}

void CodeGenHelper::populateLUT_Carry() {
  SmallVector<uint8_t> lut;

  for (uint8_t i = 0; i < 0x2; i++) {
    for (uint8_t j = 0; j <= 0xF; j++) {
      for (uint8_t k = 0; k <= 0xF; k++) {
        uint8_t result = (i + j + k) >> 4;
        assert(result == 1 || result == 0);
        lut.push_back(result);
      }
    }
  }

  lutPos[static_cast<uint8_t>(LUTOpName::LUT_Carry)] = lutContent.size();
  lutContent.insert(lutContent.end(), lut.begin(), lut.end());
}

// 3 inputs
void CodeGenHelper::populateLUT_Add() {
  // add op: carry (1b), op1, op2
  SmallVector<uint8_t> lut;

  for (uint8_t i = 0; i < 0x2; i++) {
    for (uint8_t j = 0; j <= 0xF; j++) {
      for (uint8_t k = 0; k <= 0xF; k++) {
        uint8_t result = (i + j + k) & 0xF;
        lut.push_back(result);
      }
    }
  }

  lutPos[static_cast<uint8_t>(LUTOpName::LUT_Add)] = lutContent.size();
  lutContent.insert(lutContent.end(), lut.begin(), lut.end());
}

void CodeGenHelper::populateLUT_Mux() {
  // mux: en (1b), op1, op2
  SmallVector<uint8_t> lut;

  for (uint8_t i = 0; i < 0x2; i++) {
    for (uint8_t j = 0; j <= 0xF; j++) {
      for (uint8_t k = 0; k <= 0xF; k++) {
        uint8_t result = (i == 1) ? j : k;
        lut.push_back(result);
      }
    }
  }

  lutPos[static_cast<uint8_t>(LUTOpName::LUT_Mux)] = lutContent.size();
  lutContent.insert(lutContent.end(), lut.begin(), lut.end());
}

void CodeGenHelper::populateLUT_Shl() {
  // dshl: shamt, op1, filling
  SmallVector<uint8_t> lut;

  for (uint8_t i = 0; i < 0x4; i++) {
    for (uint8_t j = 0; j <= 0xF; j++) {
      for (uint8_t k = 0; k <= 0xF; k++) {
        uint8_t result = (((j << 4) | k) >> (4 - i)) & 0xF;
        lut.push_back(result);
      }
    }
  }

  lutPos[static_cast<uint8_t>(LUTOpName::LUT_DShl)] = lutContent.size();
  lutPos[static_cast<uint8_t>(LUTOpName::LUT_Shl1)] = lutContent.size() + 256;
  lutPos[static_cast<uint8_t>(LUTOpName::LUT_Shl2)] = lutContent.size() + 512;
  lutPos[static_cast<uint8_t>(LUTOpName::LUT_Shl3)] = lutContent.size() + 768;
  lutContent.insert(lutContent.end(), lut.begin(), lut.end());
}

void CodeGenHelper::populateLUT_DShr() {
  // dshr: shamt, op1, filling
  SmallVector<uint8_t> lut;

  for (uint8_t i = 0; i < 0x4; i++) {
    for (uint8_t j = 0; j <= 0xF; j++) {
      for (uint8_t k = 0; k <= 0xF; k++) {
        uint8_t result = (((j << 4) | k) >> i) & 0xF;
        lut.push_back(result);
      }
    }
  }

  lutPos[static_cast<uint8_t>(LUTOpName::LUT_DShr)] = lutContent.size();
  lutContent.insert(lutContent.end(), lut.begin(), lut.end());
}

void CodeGenHelper::populateLUT() {
  assert(lutContent.empty());
  lutPos.resize(toucan::getMaxEnumValForLUTOpName());

  populateLUT_Nop();
  populateLUT_Rep1b();
  populateLUT_XorR();

  // 2 inputs
  populateLUT_And();
  populateLUT_Or();
  populateLUT_Xor();
  populateLUT_Cmp_Eq();
  populateLUT_Mul_Hi();
  populateLUT_Mul_Lo();
  populateLUT_Carry();

  // 3 inputs
  populateLUT_Add();
  populateLUT_Mux();

  // point shlN lut to DShl
  // Replace shr op with shl op
  populateLUT_Shl();
  populateLUT_DShr();

  // for (auto elem: lutContent) {
  //   assert(elem <= 0xF);
  // }

  // for (size_t enumId = 0; enumId <= toucan::getMaxEnumValForLUTOpName(); enumId++) {
  //   llvm::dbgs() << "Pos for op " << stringifyLUTOpName(static_cast<LUTOpName>(enumId)) << ": " << lutPos[enumId] << "\n";
  // }
}