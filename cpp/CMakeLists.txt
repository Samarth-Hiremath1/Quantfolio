cmake_minimum_required(VERSION 3.15)
project(quantfolio_cpp CXX)

set(CMAKE_CXX_STANDARD 17)
set(CMAKE_CXX_STANDARD_REQUIRED ON)

# -O3 (not -O2) is chosen deliberately: this is a compute-bound, branch-light
# hot loop over contiguous double arrays (the SMA rolling sums and
# mark-to-market). -O3 enables the extra loop optimizations and auto-
# vectorization that -O2 leaves off, and there is no shared library / code-size
# constraint here to argue for -O2's smaller footprint. The gain over -O2 is
# expected to be small: the bottleneck is queue/visit dispatch, not arithmetic
# the vectorizer can widen.
if(NOT CMAKE_BUILD_TYPE)
  set(CMAKE_BUILD_TYPE Release)
endif()
set(CMAKE_CXX_FLAGS_RELEASE "-O3 -DNDEBUG")

add_executable(backtest  main.cpp)
add_executable(benchmark benchmark.cpp)
