// Host-only regression for scheduleNOPMicroPart's lane/constant encoding.
// Run against a compiled design containing copy-only micro-partitions.
#include "ToucanGPUGenDataTypes.h"
#include <fstream>
#include <iostream>

int main(int argc, char **argv) {
  if (argc != 2) {
    std::cerr << "Usage: check_nop_bytecode GPUSimDesign.bin\n";
    return 2;
  }
  std::ifstream input(argv[1], std::ios::binary);
  if (!input) return 2;
  toucanGPUSim::SimDesignInfo design;
  toucanGPUSim::deserializeSimDesignInfo(input, design);
  if (!input) return 2;
  unsigned checked = 0;
  for (unsigned p = 0; p < design.parts.size(); ++p) {
    const auto &part = design.parts[p];
    for (unsigned l = 0; l < part.exec_mParts.size(); ++l) {
      for (unsigned m = 0; m < part.exec_mParts[l].size(); ++m) {
        const auto &mp = part.exec_mParts[l][m];
        if (!mp.isLUTPart || !mp.middleLevels.empty() || mp.topLevel.empty())
          continue;
        bool copies = true;
        for (const auto &op : mp.topLevel)
          copies &= op.lutIndex == 0 && op.op2 >= 16;
        if (!copies) continue;
        // Every source holds the same nibble. A pure copy must preserve it,
        // regardless of how outputs are ordered or which lane supplies them.
        for (unsigned value = 0; value < 16; ++value) {
          for (const auto &op : mp.lastLevel) {
            const unsigned selector = op.shuffleId;
            const unsigned actual = selector < 32 ? selector : value;
            if (actual != value || selector >= 32 + mp.topLevel.size()) {
              std::cerr << "FAIL: partition=" << p << " level=" << l
                        << " micro-part=" << m << " output=" << op.result
                        << " selector=" << selector << " input=" << value
                        << " actual=" << actual << '\n';
              return 1;
            }
          }
        }
        checked += mp.lastLevel.size();
      }
    }
  }
  if (!checked) {
    std::cerr << "No eligible copy outputs: regression was not exercised\n";
    return 2;
  }
  std::cout << "PASS: " << checked << " copy outputs preserve all 16 nibble values\n";
}
