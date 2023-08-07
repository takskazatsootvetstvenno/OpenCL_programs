#pragma once
#define CL_HPP_TARGET_OPENCL_VERSION 300
#define CL_HPP_ENABLE_EXCEPTIONS
#include <CL/opencl.hpp>

#include <string_view>
#include <vector>

namespace Tester {

class Application {
 public:
    Application();
    void loadDataFromDisk(std::string_view folderWithBinaries);
    void testVertexShader();

    void testFragmentShader();

 private:
    cl::Program compileProgram(std::string_view kernal);

    cl::Platform m_platform;
    cl::Context m_context;
    cl::CommandQueue m_queue;

    // Input blobs
    std::vector<float> m_vertexBuffer;
    std::vector<uint32_t> m_indexBuffer;
    std::array<float, 16> m_MVP{};
    std::array<float, 16> m_screen{};

    // Blobs for comparation
    std::vector<float> m_VStoFSBuffer;
    std::vector<float> m_VStoFSBuffer_from_gpu;
    std::vector<uint32_t> m_resultScreenBuffer;
    std::vector<uint32_t> m_resultScreenBuffer_from_gpu;
};
}  // namespace Tester
