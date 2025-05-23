
add_circt_dialect_library(
  ToucanAnalysis

  IsLegalToucan4B.cpp
  DesignGraph.cpp
  SchedulerBase.cpp
  # SingleRegionScheduler.cpp
  MultiRegionPartitioner.cpp
  # NaivePartitioner.cpp
  RepCutPartitioner.cpp
  GraphUtils.cpp
  ToucanAnalysis.cpp
  # LocalValueAllocator.cpp
  MicroPart.cpp
  MicroPartitioner.cpp
  MicroPartLocalValueAllocator.cpp
  SingleRegionMicroPartScheduler.cpp


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
