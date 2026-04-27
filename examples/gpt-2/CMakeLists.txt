#
# gpt-2

set(TEST_TARGET gpt-2-ctx)
add_executable(${TEST_TARGET} main-ctx.cpp)
target_link_libraries(${TEST_TARGET} PRIVATE ggml common common-ggml)

set_source_files_properties(../../modular_engine/KaHIP.c PROPERTIES LANGUAGE CXX)

set(TEST_TARGET gpt-2-alloc)
add_executable(${TEST_TARGET} main-alloc.cpp ../../modular_engine/parser.c ../../modular_engine/distrib.c ../../modular_engine/dummy_distrib.c ../../modular_engine/KaHIP.c)
target_link_libraries(${TEST_TARGET} PRIVATE ggml common common-ggml m)
target_include_directories(${TEST_TARGET} PRIVATE ../../modular_engine)

if (GGML_HAVE_KAHIP)
	target_include_directories(${TEST_TARGET} PRIVATE ${KAHIP_INCLUDE_DIR})
	target_link_libraries(${TEST_TARGET} PRIVATE ${KAHIP_LIBRARY})
	target_compile_definitions(${TEST_TARGET} PRIVATE GGML_USE_KAHIP=1)
endif()

set(TEST_TARGET gpt-2-backend)
add_executable(${TEST_TARGET} main-backend.cpp)
target_link_libraries(${TEST_TARGET} PRIVATE ggml common common-ggml)

set(TEST_TARGET gpt-2-sched)
add_executable(${TEST_TARGET} main-sched.cpp)
target_link_libraries(${TEST_TARGET} PRIVATE ggml common common-ggml)

#
# gpt-2-quantize

set(TEST_TARGET gpt-2-quantize)
add_executable(${TEST_TARGET} quantize.cpp)
target_link_libraries(${TEST_TARGET} PRIVATE ggml common common-ggml)

#
# gpt-2-batched

set(TEST_TARGET gpt-2-batched)
add_executable(${TEST_TARGET} main-batched.cpp)
target_link_libraries(${TEST_TARGET} PRIVATE ggml common common-ggml)
