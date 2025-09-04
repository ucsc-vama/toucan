
add_circt_dialect_library(
  ToucanAnalysis

  IsLegalToucan4B.cpp
  DesignGraph.cpp
  SchedulerBase.cpp
  # SingleRegionScheduler.cpp
  # NaivePartitioner.cpp
  RepCutPartitioner.cpp
  PartitioningGraph.cpp
  ToucanAnalysis.cpp
  # LocalValueAllocator.cpp
  MicroPart.cpp
  MicroPartitioner.cpp
  MicroPartLocalValueAllocator.cpp
  PartitioningManager.cpp
  MultiRegionMicroPartScheduler.cpp


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
