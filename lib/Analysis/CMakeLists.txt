
add_circt_dialect_library(
  ToucanAnalysis

  IsLegalToucan4B.cpp


  DEPENDS
  ToucanPassesIncGen

  LINK_LIBS PUBLIC

  Toucan

  CIRCTFIRRTL
  CIRCTHW
  CIRCTSeq
  CIRCTComb
)
