add_circt_dialect(Toucan toucan)
# add_circt_dialect_doc(Toucan toucan)

include_directories(${CMAKE_CURRENT_SOURCE_DIR}/include)


set(LLVM_TARGET_DEFINITIONS ToucanPasses.td)
mlir_tablegen(ToucanPasses.h.inc -gen-pass-decls)
add_public_tablegen_target(ToucanPassesIncGen)