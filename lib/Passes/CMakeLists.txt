
add_circt_dialect_library(
  ToucanPasses

  RemoveSVnOM.cpp
  ExpandSVMacro.cpp
  SplitFirMemRWPorts.cpp
  ExpandMemoryDelay.cpp
  ReplaceAsyncResetRegs.cpp
  SplitRegisters.cpp
  RemoveMemMask.cpp
  ParallelCanonicalizer.cpp
  LowerRegMemTo4B.cpp
  EnsureNoClockOp.cpp
  LowerCombPreProcess.cpp
  LowerCombTo4B_1.cpp
  LowerCombTo4B_2.cpp
  LowerCombTo4B_3.cpp
  Flatten.cpp

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