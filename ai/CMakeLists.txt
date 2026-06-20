cmake_minimum_required(VERSION 3.18)
project(goes_ai CXX)

set(CMAKE_CXX_STANDARD 17)
set(CMAKE_CXX_STANDARD_REQUIRED ON)

# LibTorch: set via -DCMAKE_PREFIX_PATH=/path/to/libtorch
find_package(Torch REQUIRED)
find_package(OpenMP)
set(CMAKE_CXX_FLAGS "${CMAKE_CXX_FLAGS} ${TORCH_CXX_FLAGS}")

# Linux: LibTorch has transitive dependencies on libraries that may not be on
# the default linker search path. Pass -DCUPTI_DIR and -DNVSHMEM_DIR if you
# get "not found" linker warnings for libcupti or libnvshmem (see ai/Readme.md).
foreach(_dir CUPTI_DIR NVSHMEM_DIR)
    if(DEFINED ${_dir})
        link_directories(${${_dir}})
        list(APPEND CMAKE_BUILD_RPATH "${${_dir}}")
    endif()
endforeach()

add_library(goes_core STATIC
  src/game/board_config.cpp
  src/game/board_state.cpp
  src/model/features.cpp
  src/model/gnn.cpp
  src/mcts/mcts.cpp
  src/training/replay_buffer.cpp
  src/training/self_play.cpp
)
target_include_directories(goes_core PUBLIC include third_party)
target_link_libraries(goes_core PUBLIC "${TORCH_LIBRARIES}")
if(OpenMP_CXX_FOUND)
    target_link_libraries(goes_core PUBLIC OpenMP::OpenMP_CXX)
endif()
target_compile_features(goes_core PUBLIC cxx_std_17)

add_executable(goes_train src/train.cpp)
target_link_libraries(goes_train goes_core)

add_executable(goes_server src/server.cpp)
target_link_libraries(goes_server goes_core)

# Linux: embed RPATH (not RUNPATH) so that transitive dependencies such as
# libcupti are found at runtime without setting LD_LIBRARY_PATH.
if(UNIX AND NOT APPLE)
    foreach(tgt goes_train goes_server)
        target_link_options(${tgt} PRIVATE -Wl,--disable-new-dtags)
    endforeach()
endif()

# Windows: copy Torch DLLs next to each binary so they can be run in-place
if(MSVC)
  file(GLOB TORCH_DLLS "${TORCH_INSTALL_PREFIX}/lib/*.dll")
  foreach(tgt goes_train goes_server)
    add_custom_command(TARGET ${tgt} POST_BUILD
      COMMAND ${CMAKE_COMMAND} -E copy_if_different
        ${TORCH_DLLS} $<TARGET_FILE_DIR:${tgt}>
    )
  endforeach()
endif()
