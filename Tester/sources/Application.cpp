#include "Application.hpp"
#include <exception>
#include <iostream>
#include <string>
#include <fstream>
#include <algorithm>
#include <sstream>
#include <filesystem>
#include <iomanip>

const char* vertexShaderKernal =
#include "VertexShader.cl"
    ;
namespace {
    cl::Platform get_platform() {
        std::vector<cl::Platform> platforms;
        cl::Platform::get(&platforms);
        for (auto& p : platforms) {
            cl_uint numDevices = 0;
            clGetDeviceIDs(p(), CL_DEVICE_TYPE_GPU, 0, nullptr, &numDevices);
            if (numDevices > 0) return cl::Platform(p);
        }
        throw std::runtime_error("Can't find suitable opencl platform");
        return {};
    }

    cl::Context get_context(cl_platform_id p_id) {
        cl_context_properties properties[] = {CL_CONTEXT_PLATFORM,
                                              reinterpret_cast<cl_context_properties>(p_id), 0};

        return cl::Context(CL_DEVICE_TYPE_GPU, properties);
    }

    const cl::QueueProperties getQueueProperties() noexcept {
        return cl::QueueProperties::Profiling | cl::QueueProperties::OutOfOrder;
    }

    void compareBlobs(
        const std::vector<float> goldResult,
        const std::vector<float> measureResult,
        const int max_size = INT_MAX) 
    {
        if (goldResult.size() != measureResult.size()) 
            throw std::runtime_error("compareBlobs: Blobs should have equal sizes!");
        std::cout << std::string(34, '-');
        std::cout << "\n    Golden | Measured | data type" << std::endl;
        std::cout << std::string(34, '-');
        bool isDiff = false;
        for (size_t i = 0; i < goldResult.size() && i < max_size; ++i) { 
            if (goldResult[i] - measureResult[i] > std::numeric_limits<float>::epsilon()) { 
                isDiff = true; 
            }
            if (i % 6 == 0) std::cout << std::endl;
            std::cout << std::setw(10) << std::setprecision(5) << goldResult[i] << " | " << std::setw(8)
                      << std::setprecision(5) << measureResult[i] << " |";
            if (i % 6 == 0) std::cout << " Vertex #" << i / 6;
            if (i % 6 == 3) std::cout << " Normal";

            std::cout << std::endl;
            
        }
        std::cout << std::string(34, '-');
        std::cout << std::endl;
        if (isDiff)
        { 
            std::cout << "Warning! Blobs does not equal!!!" << std::endl;
            std::cout << std::string(34, '-');
        }
    }
    
}

namespace Tester {
    Application::Application()
        : 
        m_platform(get_platform()),
        m_context(get_context(m_platform())),
        m_queue(m_context, getQueueProperties()) {

        auto name = m_platform.getInfo<CL_PLATFORM_NAME>();
        auto profile = m_platform.getInfo<CL_PLATFORM_PROFILE>();
        auto version = m_platform.getInfo<CL_PLATFORM_VERSION>();
        auto vendor = m_platform.getInfo<CL_PLATFORM_VENDOR>();

        std::cout << "Selected platform: " << name
            << "\nVersion: " << version
            << ", Profile: " << profile
            << "\nVendor:  " << vendor
            << std::endl << std::endl;
    
    }

    void Application::loadDataFromDisk(std::string_view folderWithBinaries) {
        std::cout << "Folder: \"" << folderWithBinaries << "\"" << std::endl;
        std::filesystem::path folder(folderWithBinaries);
        auto verticesPath = folder / "Vertices.bin";
        auto indicesPath = folder / "Indices.bin";
        auto MVPPath = folder / "MVP.bin";
        auto ScreenMatrixPath = folder / "ScreenMatrix.bin";
        auto VStoFSPath = folder / "VStoFSBuffer.bin";

        std::ifstream vertexFile(verticesPath, std::ios::binary);
        std::ifstream indexFile(indicesPath, std::ios::binary);
        std::ifstream MVPFile(MVPPath, std::ios::binary);
        std::ifstream ScreenMatrixFile(ScreenMatrixPath, std::ios::binary);
        std::ifstream VStoFSFile(VStoFSPath, std::ios::binary);

        if (!vertexFile.is_open()) throw std::runtime_error("Can't open Vertices.bin");
        if (!indexFile.is_open()) throw std::runtime_error("Can't open Indices.bin");
        if (!MVPFile.is_open()) throw std::runtime_error("Can't open MVP.bin");
        if (!ScreenMatrixFile.is_open()) throw std::runtime_error("Can't open ScreenMatrix.bin");
        if (!VStoFSFile.is_open()) throw std::runtime_error("Can't open VStoFSBuffer.bin");

        auto vertices_size = std::filesystem::file_size(verticesPath);
        auto indices_size = std::filesystem::file_size(indicesPath);
        auto MVP_size = std::filesystem::file_size(MVPPath);
        auto ScreenMatrix_size = std::filesystem::file_size(ScreenMatrixPath);
        auto VStoFS_size = std::filesystem::file_size(VStoFSPath);

        if (MVP_size != sizeof(float) * 16) throw std::runtime_error("Wrong MVP file size! Size: " + MVP_size);
        if (ScreenMatrix_size != sizeof(float) * 16) throw std::runtime_error("Wrong ScreenMatrix file size! Size: " + MVP_size);

        m_vertexBuffer.resize(vertices_size / sizeof(float));
        vertexFile.read(reinterpret_cast<char*>(m_vertexBuffer.data()), vertices_size);
        m_indexBuffer.resize(indices_size / sizeof(uint32_t));
        indexFile.read(reinterpret_cast<char*>(m_indexBuffer.data()), indices_size);
        m_VStoFSBuffer.resize(VStoFS_size / sizeof(float));
        VStoFSFile.read(reinterpret_cast<char*>(m_VStoFSBuffer.data()), VStoFS_size);

        MVPFile.read(reinterpret_cast<char*>(m_MVP.data()), sizeof(float) * 16);
        ScreenMatrixFile.read(reinterpret_cast<char*>(m_screenBuffer.data()), sizeof(float) * 16);
    }

    void Application::testVertexShader() {
        if (m_vertexBuffer.empty() || m_VStoFSBuffer.empty()) throw std::runtime_error("Load data from disk is required before tests!");

        m_VStoFSBuffer_from_gpu.resize(m_VStoFSBuffer.size());
        std::fill(m_VStoFSBuffer_from_gpu.begin(), m_VStoFSBuffer_from_gpu.end(), 0.f);

        cl::Buffer vertex(m_context, CL_MEM_READ_ONLY, m_vertexBuffer.size() * sizeof(float));
        cl::Buffer index(m_context, CL_MEM_READ_ONLY, m_indexBuffer.size() * sizeof(uint32_t));
        cl::Buffer MVP(m_context, CL_MEM_READ_ONLY, 16 * sizeof(float));
        cl::Buffer Screen(m_context, CL_MEM_READ_ONLY, 16 * sizeof(float));
        cl::Buffer output(m_context, CL_MEM_WRITE_ONLY, m_VStoFSBuffer.size() * sizeof(float));

        cl::copy(m_queue, m_vertexBuffer.data(), m_vertexBuffer.data() + m_vertexBuffer.size(), vertex);
        cl::copy(m_queue, m_indexBuffer.data(), m_indexBuffer.data() + m_indexBuffer.size(), index);
        cl::copy(m_queue, m_MVP.data(), m_MVP.data() + m_MVP.size(), MVP);
        cl::copy(m_queue, m_screenBuffer.data(), m_screenBuffer.data() + m_screenBuffer.size(), Screen);

        cl::Program program = compileProgram(vertexShaderKernal);

        using vecAdd = cl::KernelFunctor<cl::Buffer, cl::Buffer, cl::Buffer, cl::Buffer, cl::Buffer>;
        vecAdd add_vecs(program, "vertex_shader");

        cl::NDRange GlobalRange(m_indexBuffer.size());
        cl::NDRange LocalRange(1);
        cl::EnqueueArgs Args(m_queue, GlobalRange, LocalRange);

        cl::Event evt = add_vecs(Args, vertex, index, MVP, Screen, output);

        evt.wait();

        cl::copy(m_queue, output, m_VStoFSBuffer_from_gpu.data(),
                 m_VStoFSBuffer_from_gpu.data() + m_VStoFSBuffer_from_gpu.size());

        auto GPUTimeStart = evt.getProfilingInfo<CL_PROFILING_COMMAND_START>();  // in ns
        auto GPUTimeFin = evt.getProfilingInfo<CL_PROFILING_COMMAND_END>();
        auto GDur = (GPUTimeFin - GPUTimeStart) / 1000;  // ns -> µs
        std::cout << "GPU pure time measured: " << GDur << " Microseconds" << std::endl;

        compareBlobs(m_VStoFSBuffer, m_VStoFSBuffer_from_gpu, 36);
        for (auto el : m_VStoFSBuffer_from_gpu) std::cout << el << " ";
        std::cout << std::endl;
    }


    cl::Program Application::compileProgram(std::string_view kernel) { 
        cl::Program program(m_context, kernel.data());
        try {
            program.build("-Werror"); // see https://man.opencl.org/clBuildProgram.html
        } catch (const std::exception& e) {
            std::stringstream ss;
            ss << "\ncompileProgram(..) error: \n";
            ss << "Exception: \"" << e.what() << "\"" << std::endl;
            ss << "Reason:\n";
            auto buildInfo = program.getBuildInfo<CL_PROGRAM_BUILD_LOG>();
            for (auto& [device, error] : buildInfo) {
                ss << "Program build log for device \"" << device.getInfo<CL_DEVICE_NAME>()
                   << "\"\nwith compiler arguments: \""
                   << program.getBuildInfo<CL_PROGRAM_BUILD_OPTIONS>(device)
                   << "\"\nKernal: " << program.getInfo<CL_PROGRAM_SOURCE>() << std::endl
                   << "Compilation error: \n"
                   << error << std::endl;
            }
            throw std::runtime_error(ss.str());
        }
        return program; 
    }

    
}  // namespace Tester