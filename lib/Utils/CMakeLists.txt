
add_circt_dialect_library(
  ToucanUtils

  Utils.cpp
  BoostSetting.cpp


  DEPENDS
  ToucanPassesIncGen

  LINK_LIBS PUBLIC

  Toucan

  CIRCTFIRRTL
  CIRCTHW
  CIRCTSeq
  CIRCTComb

)
