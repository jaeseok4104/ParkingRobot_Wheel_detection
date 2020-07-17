add_executable( main main.cpp)
add_executable( test houghcircle.cpp)

target_link_libraries( main PARKING)
target_link_libraries( test PARKING)