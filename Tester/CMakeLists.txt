cmake_minimum_required(VERSION 3.18)
   
set(TESTER_INCLUDES
	includes/Application.hpp
	includes/TableResults.hpp
	includes/TestVector.hpp
	includes/hashpp.h
	includes/json.hpp
)

set(TESTER_SOURCES
	sources/Application.cpp
	sources/TableResults.cpp
	sources/TestVector.cpp
	sources/main.cpp
)
add_executable(${PROJECT_NAME}
	${TESTER_INCLUDES}
	${TESTER_SOURCES}
)
set_property( DIRECTORY PROPERTY VS_STARTUP_PROJECT ${PROJECT_NAME} )
target_compile_features(${PROJECT_NAME} PUBLIC cxx_std_20)
target_include_directories(${PROJECT_NAME} PUBLIC includes)
target_include_directories(${PROJECT_NAME} PRIVATE src)
                                                           
add_subdirectory(../external/OpenCL-SDK ${CMAKE_CURRENT_BINARY_DIR}/OpenCL-SDK)
target_link_libraries(${PROJECT_NAME} PRIVATE OpenCL::HeadersCpp)
target_link_libraries(${PROJECT_NAME} PRIVATE OpenCL)

install(TARGETS ${PROJECT_NAME} DESTINATION ${CMAKE_INSTALL_PREFIX})