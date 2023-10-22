#pragma once
#define CL_HPP_TARGET_OPENCL_VERSION 300
#define CL_HPP_ENABLE_EXCEPTIONS
#include <CL/opencl.hpp>
#include <filesystem>
#include <string_view>
#include <vector>
#include <tuple>
#include "TestVector.hpp"

namespace Tester {

class Application {
 public:
    Application();

    void parseTestFolder(std::filesystem::path pathToTests);
    void runTests();
   
 private:
    std::vector<uint8_t> run_host_gpu(const Test& test);

    Test::GPUVenderType m_vendor = Test::GPUVenderType::NVIDIA;
    cl::Program compileProgram(std::string_view kernal);
    cl::Platform m_platform;
    cl::Context m_context;
    cl::CommandQueue m_queue;
    std::vector<Test> m_tests;
};
}  // namespace Tester
