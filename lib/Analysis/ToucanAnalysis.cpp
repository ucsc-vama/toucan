#include "toucan/ToucanAnalysis.h"
#include "toucan/PartitioningGraph.h"
#include "llvm/Support/ErrorHandling.h"

using namespace toucan;

std::string toucan::stringifyCGToucanOPName(CGToucanOPName val) {
  switch (val) {
    case CGToucanOPName::ConstDecl: return "ConstDecl";
    case CGToucanOPName::MPart_Regular: return "MPart_Regular";
    case CGToucanOPName::MPart_Special: return "MPart_Special";
    case CGToucanOPName::LUT : return "LUT";
    case CGToucanOPName::VecRead : return "VecRead";
    case CGToucanOPName::VecDecl : return "VecDecl";
    case CGToucanOPName::Print : return "Print";
    case CGToucanOPName::Stop : return "Stop";
    case CGToucanOPName::RegRead : return "RegRead";
    case CGToucanOPName::RegWrite : return "RegWrite";
    case CGToucanOPName::MemRead : return "MemRead";
    case CGToucanOPName::MemWrite : return "MemWrite";
    case CGToucanOPName::ExchangeRead : return "ExgRead";
    case CGToucanOPName::ExchangeWrite : return "ExgWrite";
    case CGToucanOPName::VecLogic: return "VecLogic";
    case CGToucanOPName::VecArith: return "VecArith";
    case CGToucanOPName::VecStaticRead: return "VecStaticRead";

    case CGToucanOPName::ShouldNotAppear: return "ShouldNotAppear";
    case CGToucanOPName::Dummy_DefReg: return "DefReg";
    case CGToucanOPName::Dummy_DefMem: return "DefMem";
      // Should not appear
      break;
    }
  llvm_unreachable("Every op name should be stringified!");
}

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

