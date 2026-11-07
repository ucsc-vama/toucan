add_executable(check-design-graph check_design_graph.cpp)
llvm_update_compile_flags(check-design-graph)
target_link_libraries(check-design-graph PRIVATE Toucan ToucanAnalysis ToucanUtils MLIRParser)

add_executable(check-local-value-allocator check_local_value_allocator.cpp)
llvm_update_compile_flags(check-local-value-allocator)
target_link_libraries(check-local-value-allocator PRIVATE Toucan ToucanAnalysis ToucanUtils MLIRParser)
