cmake_minimum_required(VERSION 3.18)
project(goes_ai CXX)

set(CMAKE_CXX_STANDARD 17)
set(CMAKE_CXX_STANDARD_REQUIRED ON)

# LibTorch: set via -DCMAKE_PREFIX_PATH=/path/to/libtorch
find_package(Torch REQUIRED)
find_package(OpenMP)
set(CMAKE_CXX_FLAGS "${CMAKE_CXX_FLAGS} ${TORCH_CXX_FLAGS}")

# TorchConfig.cmake only auto-links extra libraries it finds via its
# append_torchlib_if_found macro (e.g. kineto); it does NOT do this for
# cuSPARSELt, NCCL, or NVSHMEM, even when libtorch_cuda.so / libtorch_nvshmem.so
# were built against them. Without these, the final link fails with
# "undefined reference" to symbols like cusparseLtMatmul, ncclCommMemStats,
# nvshmemx_putmem_signal_on_stream. Find and link them manually.
#
# Ubuntu's NVIDIA cuda repo installs NVSHMEM (and sometimes cuSPARSELt/NCCL)
# under a versioned subdirectory such as /usr/lib/x86_64-linux-gnu/nvshmem/12/
# that isn't on the default linker search path. Auto-detect it so
# -DCUSPARSELT_DIR / -DNCCL_DIR / -DNVSHMEM_DIR don't have to be passed by
# hand; they still take priority if set explicitly.
macro(_goes_autodetect_lib_dir out_var lib_file_name)
    if(NOT DEFINED ${out_var})
        file(GLOB_RECURSE _goes_lib_candidates
            /usr/lib/*/${lib_file_name}*
            /usr/local/cuda*/lib64/${lib_file_name}*
        )
        if(_goes_lib_candidates)
            list(GET _goes_lib_candidates 0 _goes_lib_first)
            get_filename_component(${out_var} "${_goes_lib_first}" DIRECTORY)
        endif()
    endif()
endmacro()
_goes_autodetect_lib_dir(CUSPARSELT_DIR libcusparseLt.so)
_goes_autodetect_lib_dir(NCCL_DIR libnccl.so)
_goes_autodetect_lib_dir(NVSHMEM_DIR libnvshmem_host.so)

# Enable with: cmake -DSANITIZE=ON ...
option(SANITIZE "Enable -fsanitize=address,undefined" OFF)
if(SANITIZE AND NOT MSVC)
    add_compile_options(-fsanitize=address,undefined -fno-omit-frame-pointer -g)
    add_link_options(-fsanitize=address,undefined)
endif()

# Linux: LibTorch has transitive dependencies on libraries that may not be on
# the default linker search path. CUSPARSELT_DIR/NCCL_DIR/NVSHMEM_DIR are
# auto-detected above; CUPTI_DIR is not, so pass -DCUPTI_DIR if you get "not
# found" linker warnings for libcupti.
foreach(_dir CUPTI_DIR NVSHMEM_DIR CUSPARSELT_DIR NCCL_DIR)
    if(DEFINED ${_dir})
        link_directories(${${_dir}})
        list(APPEND CMAKE_BUILD_RPATH "${${_dir}}")
    endif()
endforeach()

add_library(goes_core STATIC
  src/game/board_config.cpp
  src/game/board_state.cpp
  src/model/features.cpp
  src/model/model_config.cpp
  src/model/gnn.cpp
  src/model/unet.cpp
  src/model/cnn.cpp
  src/model/transformer.cpp
  src/mcts/mcts.cpp
  src/training/replay_buffer.cpp
  src/training/self_play.cpp
  src/util/sha256.cpp
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
