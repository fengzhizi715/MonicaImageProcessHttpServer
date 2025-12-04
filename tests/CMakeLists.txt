# CMakeLists.txt for tests
cmake_minimum_required(VERSION 3.10)

# Create test executable
add_executable(unit_tests unit_tests.cpp)

# Link required libraries
target_link_libraries(unit_tests 
    PRIVATE 
    MonicaImageProcessHttpServer
)

# Include directories
target_include_directories(unit_tests 
    PRIVATE 
    ${CMAKE_SOURCE_DIR}/include
    ${CMAKE_SOURCE_DIR}/src/utils
)

# Add test to CTest
enable_testing()
add_test(NAME unit_tests COMMAND unit_tests)