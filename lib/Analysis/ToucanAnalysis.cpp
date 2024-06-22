#include "toucan/ToucanAnalysis.h"

using namespace toucan;

std::string toucan::stringifyCGToucanOPName(CGToucanOPName val) {
  switch (val) {
    case CGToucanOPName::ConstDecl: return "ConstDecl";
    case CGToucanOPName::LUT : return "LUT";
    case CGToucanOPName::VecRead : return "VecRead";
    case CGToucanOPName::VecDecl : return "VecDecl";
    case CGToucanOPName::Print : return "Print";
    case CGToucanOPName::Stop : return "Stop";
    case CGToucanOPName::RegRead : return "RegRead";
    case CGToucanOPName::RegWrite : return "RegWrite";
    case CGToucanOPName::MemRead : return "MemRead";
    case CGToucanOPName::MemWrite : return "MemWrite";
    case CGToucanOPName::ShouldNotAppear : return "ShouldNotAppear";
    case CGToucanOPName::ExchangeRead : return "ExgRead";
    case CGToucanOPName::ExchangeWrite : return "ExgWrite";
  }
  return "???";
}