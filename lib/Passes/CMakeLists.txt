
add_circt_dialect_library(
  ToucanPasses

  RemoveSVnOM.cpp
  ExpandSVMacro.cpp
  FactorArrayGet.cpp
  SplitFirMemRWPorts.cpp
  ExpandMemoryDelay.cpp
  ReplaceAsyncResetRegs.cpp
  SplitRegisters.cpp
  RemoveMemMask.cpp
  Canonicalizer.cpp
  LowerRegMemTo4B.cpp
  EnsureNoClockOp.cpp
  LowerCombPreProcess.cpp
  LowerCombTo4B_ShortReplicateOp.cpp
  LowerCombTo4B_1.cpp
  LowerCombTo4B_2.cpp
  LowerCombTo4B_3.cpp
  Flatten.cpp
  FactorConcatExtract.cpp
  FlatDCE.cpp
  EnsureToucanOnly.cpp
  MergeConst.cpp
  CPUSingleThreadCodeGen.cpp

  DEPENDS
  ToucanPassesIncGen

  LINK_LIBS PUBLIC

  Toucan
  ToucanSim
  ToucanUtils
  ToucanAnalysis

  CIRCTFIRRTL
  CIRCTHW
  CIRCTSeq
  CIRCTComb

)