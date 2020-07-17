add_library( PARKING SHARED
    realsense.cpp
    slamBase.cpp
)

target_link_libraries( PARKING
    ${THIRD_PARTY_LIBS}
)
