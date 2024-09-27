
add_circt_dialect_library(
  ToucanAnalysis

  IsLegalToucan4B.cpp
  DesignGraph.cpp
  SchedulerBase.cpp
  SingleRegionScheduler.cpp
  MultiRegionScheduler.cpp
  NaivePartitioner.cpp
  RepCutPartitioner.cpp
  GraphUtils.cpp
  ToucanAnalysis.cpp
  LocalValueAllocator.cpp


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
