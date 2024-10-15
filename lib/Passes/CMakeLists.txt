
add_circt_dialect_library(
  ToucanPasses

  RemoveSVnOM.cpp
  ExpandSVMacro.cpp
  FactorArrayGet.cpp
  SplitFirMemRWPorts.cpp
  ExpandMemoryDelay.cpp
  SplitRegisters.cpp
  RemoveMemMask.cpp
  ToucanCanonicalizer.cpp
  LowerRegMemTo4B.cpp
  EnsureNoClockOp.cpp
  FactorBinaryOp.cpp
  LowerHWVectorTo4B.cpp
  LowerCombTo4B_1.cpp
  LowerCombTo4B_2.cpp
  LowerCombTo4B_3.cpp
  Flatten.cpp
  FactorConcatExtract.cpp
  EnsureToucanOnly.cpp
  DeduplicateRegisters.cpp
  MergeConst.cpp
  CPUSingleThreadCodeGen.cpp
  GPUCodeGen.cpp

  DEPENDS
  ToucanPassesIncGen

  LINK_LIBS PUBLIC

  Toucan
  ToucanSim
  ToucanGPUSim
  ToucanUtils
  ToucanAnalysis

  CIRCTFIRRTL
  CIRCTHW
  CIRCTSeq
  CIRCTComb

)