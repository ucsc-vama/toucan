
add_circt_dialect_library(
  ToucanAnalysis

  IsLegalToucan4B.cpp
  DesignGraph.cpp
  PartitionerNCodeGenBase.cpp
  NaivePartitioner.cpp
  RepCutPartitioner.cpp


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
