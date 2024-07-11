cmake_minimum_required(VERSION 3.22)
set(CMAKE_CXX_COMPILER "icpx")
project(cos CXX)

add_executable(cos main.cpp)
target_compile_features(cos PRIVATE cxx_std_20)

# target_compile_definitions(cos PUBLIC
#   -DDIMENSION=${DIMENSION}
# )

set(AVX "AVX512")

target_compile_options(cos PUBLIC
  -Wall -Wextra

  -qopt-report=3

  $<$<CONFIG:Debug>: -g -O0 -Rno-debug-disables-optimization>
  $<$<CONFIG:Release>: -O3 -DNDEBUG>

  $<$<STREQUAL:${AVX},Host>: -xHost>
  $<$<STREQUAL:${AVX},AVX1>: -xCore-AVX-I>
  $<$<STREQUAL:${AVX},AVX2>: -xCore-AVX2>
  $<$<STREQUAL:${AVX},AVX512>: -xCommon-AVX512 -qopt-zmm-usage=high>
  $<$<STREQUAL:${AVX},NOVEC>: -no-vec>
)

find_package(Eigen3 3.4 REQUIRED NO_MODULES)
target_link_libraries(cos INTERFACE Eigen3::Eigen)
