
add_circt_dialect_library(
  ToucanPasses

  RemoveSVnOM.cpp
  ExpandSVMacro.cpp
  SplitFirMemRWPorts.cpp
  ExpandMemoryDelay.cpp
  ReplaceAsyncResetRegs.cpp
  SplitRegisters.cpp
  RemoveMemMask.cpp
  LowerRegMemTo4B.cpp
  EnsureNoClockOp.cpp

  DEPENDS
  ToucanPassesIncGen

  LINK_LIBS PUBLIC

  Toucan
  ToucanUtils

  CIRCTFIRRTL
  CIRCTHW
  CIRCTSeq
  CIRCTComb

)