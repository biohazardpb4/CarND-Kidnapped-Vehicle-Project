include_directories("${PROJECT_SOURCE_DIR}")

set(SOURCE_FILES ../src/particle_filter.cpp ../src/multiv_gauss.cpp ../src/helper_functions.h ../src/json.hpp ../src/map.h ../src/multiv_gauss.h ../src/particle_filter.h)

add_executable(particle_filter_test particle_filter_test.cpp ${SOURCE_FILES})
target_link_libraries(particle_filter_test gtest)
add_test(NAME particle_filter_test COMMAND particle_filter_test)
