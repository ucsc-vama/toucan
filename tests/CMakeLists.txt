add_executable(check-design-graph check_design_graph.cpp)
llvm_update_compile_flags(check-design-graph)
target_link_libraries(check-design-graph PRIVATE Toucan ToucanAnalysis ToucanUtils MLIRParser)
