#pragma once
#define CL_HPP_TARGET_OPENCL_VERSION 300
#define CL_HPP_ENABLE_EXCEPTIONS
#include <CL/opencl.hpp>
#include <string_view>

namespace Tester {
    class Application {
    public:
        Application();
        void test();

    private:
        cl::Program compileProgram(std::string_view kernal);
        cl::Platform m_platform;
        cl::Context m_context;
        cl::CommandQueue m_queue;
    };
}  // namespace Tester