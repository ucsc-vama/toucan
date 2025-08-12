#include "toucan/ToucanAnalysis.h"
#include "toucan/PartitioningGraph.h"
#include "llvm/Support/ErrorHandling.h"

using namespace toucan;


void CGOpStatistics::print() const {
  auto printMember = [=](uint32_t num, const char* name) {
    if (num != 0) {
      llvm::outs() << name << ": " << num << "\n";
    }
  };

  printMember(numRegReads, "RegRead");
  printMember(numMemReads, "MemRead");
  printMember(numVecReads, "VecRead");
  printMember(numLuts, "LUT");
  printMember(numLutNops, "LUT_nop");
  printMember(numRegWrites, "RegWrite");
  printMember(numMemWrites, "MemWrite");
  printMember(numPrints, "Print");
  printMember(numStops, "Stop");
  printMember(numExchangeReads, "ExchangeRead");
  printMember(numExchangeWrites, "ExchangeWrite");
}

size_t toucan::getExtraAlignmentSpace(size_t valSize, size_t alignment) {
  size_t extraSpace = alignment - (valSize % alignment);
  if (extraSpace == alignment) {
    return 0;
  }
  return extraSpace;
}

