cmake_minimum_required(VERSION 3.14)
project(dlrand VERSION 0.1.0 LANGUAGES C)

set(CMAKE_C_STANDARD 99)
set(CMAKE_C_STANDARD_REQUIRED ON)
set(CMAKE_C_EXTENSIONS OFF)
set(CMAKE_C_VISIBILITY_PRESET hidden)

option(RL_BUILD_TESTS "Build the C unit tests" ON)
option(RL_SANITIZE "Build tests with AddressSanitizer + UBSan (GCC/Clang)" OFF)

file(GLOB RL_SOURCES CONFIGURE_DEPENDS ${CMAKE_CURRENT_SOURCE_DIR}/src/*.c)

add_library(dlrand SHARED ${RL_SOURCES})
target_include_directories(dlrand PUBLIC  $<BUILD_INTERFACE:${CMAKE_CURRENT_SOURCE_DIR}/include>
                                           $<INSTALL_INTERFACE:include>
                                   PRIVATE ${CMAKE_CURRENT_SOURCE_DIR}/src)
target_compile_definitions(dlrand PRIVATE RL_BUILDING)
if(NOT MSVC)
  target_link_libraries(dlrand PRIVATE m)
  target_compile_options(dlrand PRIVATE -Wall -Wextra -Wpedantic -Wshadow)
else()
  target_compile_options(dlrand PRIVATE /W4)
endif()

if(RL_BUILD_TESTS)
  enable_testing()
  add_executable(test_core tests/c/test_core.c ${RL_SOURCES})
  target_include_directories(test_core PRIVATE include src)
  target_compile_definitions(test_core PRIVATE RL_STATIC)
  if(NOT MSVC)
    target_link_libraries(test_core PRIVATE m)
    if(RL_SANITIZE)
      target_compile_options(test_core PRIVATE -g -O1 -fsanitize=address,undefined -fno-sanitize-recover=undefined)
      target_link_options(test_core PRIVATE -fsanitize=address,undefined)
    endif()
  endif()
  add_test(NAME c_unit_tests COMMAND test_core)
endif()

install(TARGETS dlrand ARCHIVE DESTINATION lib LIBRARY DESTINATION lib RUNTIME DESTINATION bin)
install(DIRECTORY include/dlrand DESTINATION include)
