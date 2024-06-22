
add_circt_dialect_library(
  ToucanAnalysis

  IsLegalToucan4B.cpp
  DesignGraph.cpp
  SchedulerBase.cpp
  SingleRegionScheduler.cpp
  TwoRegionScheduler.cpp
  NaivePartitioner.cpp
  RepCutPartitioner.cpp
  GraphUtils.cpp
  ToucanAnalysis.cpp


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
