add_circt_dialect_library(
  Toucan

  ToucanDialect.cpp
  ToucanTypes.cpp

#   DEPENDS
#   MLIRToucanIncGen

  LINK_LIBS PUBLIC
  CIRCTSupport
  MLIRIR
  MLIRSupport
)
