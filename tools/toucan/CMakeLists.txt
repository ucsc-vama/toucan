set(LLVM_LINK_COMPONENTS
  Support
)

add_executable(
    toucan
    toucan.cpp
)

llvm_update_compile_flags(toucan)
target_link_libraries(toucan PRIVATE
  CIRCTExportVerilog
  CIRCTImportFIRFile
  CIRCTFIRRTLTransforms
  CIRCTHWTransforms
  CIRCTOMTransforms
  CIRCTSeq
  CIRCTSVTransforms
  CIRCTTransforms
  CIRCTFirtool
  CIRCTTargetDebugInfo

  MLIRBytecodeReader
  MLIRBytecodeWriter
  MLIRParser
  MLIRSupport
  MLIRIR
  MLIROptLib
  MLIRFuncDialect
  MLIRTransforms
)
