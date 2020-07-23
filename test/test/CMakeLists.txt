# add_executable( main main.cpp)
# add_executable( test houghcircle.cpp)
add_executable( depth depth_img.cpp)

target_link_libraries( depth PARKING)
# target_link_libraries( main PARKING)
# target_link_libraries( test PARKING)