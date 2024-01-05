set(LLVM_LINK_COMPONENTS
  Support
)

add_executable(
    toucan
    toucan.cpp
)

llvm_update_compile_flags(toucan)
target_link_libraries(toucan PRIVATE
  CIRCTHW
  CIRCTComb
  CIRCTSeq
  CIRCTOM

  # CIRCTImportFIRFile
  # CIRCTFIRRTLTransforms
  # CIRCTHWTransforms
  # CIRCTOMTransforms
  # CIRCTSVTransforms
  # CIRCTTransforms

  MLIRBytecodeReader
  MLIRBytecodeWriter
  MLIRParser
  MLIRSupport
  MLIRIR
  MLIROptLib
  MLIRFuncDialect
  MLIRTransforms
)
