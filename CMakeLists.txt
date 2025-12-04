cmake_minimum_required(VERSION 3.10)
project(MonicaImageProcess LANGUAGES CXX)

set(CMAKE_CXX_STANDARD 17)
set(CMAKE_CXX_STANDARD_REQUIRED True)

add_subdirectory(src)
add_subdirectory(include)

# Enable testing
enable_testing()

# Add tests subdirectory
add_subdirectory(tests)