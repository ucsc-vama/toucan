
add_circt_dialect_library(
  ToucanAnalysis

  IsLegalToucan4B.cpp
  DesignGraph.cpp
  NaivePartitioner.cpp
  RepCutPartitioner.cpp


  DEPENDS
  ToucanPassesIncGen

  LINK_LIBS PUBLIC

  Toucan

  CIRCTFIRRTL
  CIRCTHW
  CIRCTSeq
  CIRCTComb
)
