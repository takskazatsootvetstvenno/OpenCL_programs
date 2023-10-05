cmake_minimum_required(VERSION 3.18)

SET(PROJECT_NAME OpenCL_programs)

get_filename_component(GENERAL_FOLDER ../ ABSOLUTE)
set(CMAKE_INSTALL_PREFIX "${GENERAL_FOLDER}/install" CACHE PATH "${GENERAL_FOLDER}/install" FORCE)

project(${PROJECT_NAME})
                                                                        
option(OPENCL_SDK_BUILD_SAMPLES OFF)
option(BUILD_DOCS OFF)
option(BUILD_EXAMPLES OFF)
option(BUILD_TESTING OFF)
add_subdirectory(Tester)
target_compile_features(${PROJECT_NAME} PUBLIC cxx_std_20)

if(MSVC)
	 set_property( DIRECTORY PROPERTY VS_STARTUP_PROJECT ${PROJECT_NAME})
	 add_compile_options(/MP)
endif()