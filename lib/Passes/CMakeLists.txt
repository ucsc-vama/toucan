
add_circt_dialect_library(
  ToucanPasses

  RemoveSVnOM.cpp
  ExpandSVMacro.cpp

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