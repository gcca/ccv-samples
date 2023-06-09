add_executable(stereo_calibrate stereo_calibrate.cc)
target_link_libraries(stereo_calibrate PRIVATE Boost::program_options ${OpenCV_LIBS})
