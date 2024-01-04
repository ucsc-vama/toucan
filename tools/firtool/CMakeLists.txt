set(LLVM_LINK_COMPONENTS
  Support
)

add_executable(
    firtool_test
    firtool_test.cpp
)

llvm_update_compile_flags(firtool_test)
target_link_libraries(firtool_test PRIVATE
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
