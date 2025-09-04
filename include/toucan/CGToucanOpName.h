
#pragma once


#include <boost/graph/adjacency_list.hpp>
#include <boost/graph/copy.hpp>


namespace toucan {
  enum class CGToucanOPName {
    ConstDecl,
    MPart_Regular,
    MPart_Special,
    LUT,
    VecRead,
    VecDecl,
    VecLogic,
    VecArith,
    VecStaticRead,
    Print,
    Stop,
    RegRead,
    RegWrite,
    MemRead,
    MemWrite,
    ShouldNotAppear,
    // Note: Those nodes should be removed from graph. They do not emit code
    Dummy_DefReg,
    Dummy_DefMem,
    // Note: Exchange between regions in MultiRegionScheduler.
    ExchangeRead,
    ExchangeWrite
    // Constant,
    // ConstVec
  };

  constexpr int getMaxEnumValForCGToucanOPName() {return static_cast<int>(CGToucanOPName::ExchangeWrite);};

  std::string stringifyCGToucanOPName(CGToucanOPName val);


}
