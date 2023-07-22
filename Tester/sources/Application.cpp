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

    void compareBlobsAfterVS(
        const std::vector<float> goldResult,
        const std::vector<float> measureResult,
        const int max_size = INT_MAX) 
    {
        if (goldResult.size() != measureResult.size()) 
            throw std::runtime_error("compareBlobs: Blobs should have equal sizes!");
        std::stringstream ss;
        bool isDiff = false;
        for (size_t i = 0; i < goldResult.size(); ++i) {
            if (goldResult[i] - measureResult[i] > std::numeric_limits<float>::epsilon()) { isDiff = true; }
        }
        if (isDiff) {
            ss << std::string(34, '-');
            ss << "\n    Golden | Measured | data type" << std::endl;
            ss << std::string(34, '-');

            for (size_t i = 0; i < goldResult.size() && i < max_size; ++i) {
                if (i % 6 == 0) ss << std::endl;
                ss << std::setw(10) << std::setprecision(5) << goldResult[i] << " | " << std::setw(8)
                   << std::setprecision(5) << measureResult[i] << " |";
                if (i % 6 == 0) ss << " Vertex #" << i / 6;
                if (i % 6 == 3) ss << " Normal";
                ss << std::endl;
            }
            ss << std::string(34, '-') << std::endl;
            ss << "Vertex shader test: DIFF" << std::endl;
            ss << "Blobs are NOT equal!" << std::endl;
            ss << std::string(34, '-');
        }
        else {
            ss << std::string(34, '-') << std::endl;
            ss << "Vertex shader test: PASS" << std::endl;
            ss << std::string(34, '-');
        }
        std::cout << ss.str();
    }
    
     std::pair<std::ifstream, uint32_t> getFile(const std::filesystem::path path) {
        std::ifstream file(path, std::ios::binary);
        if (!file.is_open()) throw std::runtime_error("Can't open file: " + path.string());
        uint32_t sizeofFile = std::filesystem::file_size(path);
        return {std::move(file), sizeofFile};
     }

     template<typename T>
     void fillBufferFromFile(std::ifstream& ifs, std::vector<T>& buffer, const uint32_t size)
     {
        using buffer_type = std::remove_reference_t<decltype(buffer)>::value_type;
        buffer.resize(size / sizeof(buffer_type));
        ifs.read(reinterpret_cast<char*>(buffer.data()), size);
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

        auto [vertexFile, vertices_size] =           getFile(folder / "Vertices.bin");
        auto [indexFile, indices_size] =             getFile(folder / "Indices.bin");
        auto [MVPFile, MVP_size] =                   getFile(folder / "MVP.bin");
        auto [screenMatrixFile, screenMatrix_size] = getFile(folder / "ScreenMatrix.bin");
        auto [VStoFSFile, VStoFS_size] =             getFile(folder / "VStoFSBuffer.bin");
        auto [screenBufferFile, screenBuffer_size] = getFile(folder / "ResultScreenBuffer.bin");

        if (MVP_size != sizeof(float) * 16) 
            throw std::runtime_error("Wrong MVP file size! Size: " + MVP_size);
        if (screenMatrix_size != sizeof(float) * 16)
            throw std::runtime_error("Wrong ScreenMatrix file size! Size: " + MVP_size);

        fillBufferFromFile(vertexFile, m_vertexBuffer, vertices_size);
        fillBufferFromFile(indexFile, m_indexBuffer, indices_size);
        fillBufferFromFile(VStoFSFile, m_VStoFSBuffer, VStoFS_size);
        fillBufferFromFile(screenBufferFile, m_resultScreenBuffer, screenBuffer_size);

        MVPFile.read(reinterpret_cast<char*>(m_MVP.data()), sizeof(float) * 16);
        screenMatrixFile.read(reinterpret_cast<char*>(m_screen.data()), sizeof(float) * 16);
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
        cl::copy(m_queue, m_screen.data(), m_screen.data() + m_screen.size(), Screen);

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

        compareBlobsAfterVS(m_VStoFSBuffer, m_VStoFSBuffer_from_gpu, 36);
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