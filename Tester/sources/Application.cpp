#include "Application.hpp"
#include <exception>
#include <iostream>
#include <string>
#include <fstream>
#include <filesystem>
#include "TableResults.hpp"

const char* vertexShaderKernal =
#include "VertexShader.cl"
    ;
const char* fragmentShaderKernal =
#include "FragmentShader.cl"
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
    std::vector<float> getExpectedVertex(const std::vector<float>& vertex, const std::vector<uint32_t>& indices) {
        std::vector<float> result;
        for (const auto i : indices) {
            result.emplace_back(vertex.at(6 * i));
            result.emplace_back(vertex.at(6 * i + 1));
            result.emplace_back(vertex.at(6 * i + 2));

            result.emplace_back(vertex.at(6 * i + 3));
            result.emplace_back(vertex.at(6 * i + 4));
            result.emplace_back(vertex.at(6 * i + 5));
        }
        return result;
    }
    }

namespace Tester {
    Application::Application() : 
        m_platform(get_platform()),
        m_context(get_context(m_platform())),
        m_queue(m_context, getQueueProperties()) {

        const auto name = m_platform.getInfo<CL_PLATFORM_NAME>();
        const auto profile = m_platform.getInfo<CL_PLATFORM_PROFILE>();
        const auto version = m_platform.getInfo<CL_PLATFORM_VERSION>();
        const auto vendor = m_platform.getInfo<CL_PLATFORM_VENDOR>();
        const auto extentions = m_platform.getInfo<CL_PLATFORM_EXTENSIONS_WITH_VERSION>();
        
        std::cout << "Selected platform: " << name
            << "\nVersion: " << version
            << ", Profile: " << profile
            << "\nVendor:  " << vendor
            << std::endl << std::endl;

        for (const auto& ext : extentions) 
            if (std::string(ext.name) == "cl_khr_fp16") std::cout << "Supported fp16 extention" << std::endl;
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
        if (m_vertexBuffer.empty() || m_VStoFSBuffer.empty())
            throw std::runtime_error("Vertex shader test: Load data from disk is required before tests!");

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
        std::cout << "System GPU: Vertex shader pure time measured: " << GDur << " Microseconds" << std::endl;

        TableResults table("Vertex shader", 14, 6, 16);
        table.addDataColumn("SoftRender", m_VStoFSBuffer);
        table.addDataColumn("System GPU", m_VStoFSBuffer_from_gpu);
        table.addAdditionalInfoColumn("Input Vertex", getExpectedVertex(m_vertexBuffer, m_indexBuffer));
        table.addAdditionalInfoColumn("Input Index", m_indexBuffer);
        table.processAndShow();

        std::cout << std::endl;
    }

    void Application::testFragmentShader() {
        if (m_VStoFSBuffer.empty() || m_resultScreenBuffer.empty())
            throw std::runtime_error("Fragment shader test: Load data from disk is required before tests!");
        if (m_resultScreenBuffer.size() != 480 * 320)
            throw std::runtime_error("Fragment shader test: Wrong global size!");

        m_resultScreenBuffer_from_gpu.resize(m_resultScreenBuffer.size());
        std::fill(m_resultScreenBuffer_from_gpu.begin(), m_resultScreenBuffer_from_gpu.end(), 0);

        cl::Buffer VStoFSBuffer(m_context, CL_MEM_READ_ONLY, m_VStoFSBuffer_from_gpu.size() * sizeof(float));
        cl::Buffer output(m_context, CL_MEM_WRITE_ONLY, m_resultScreenBuffer_from_gpu.size() * sizeof(float));

        cl::copy(m_queue, m_VStoFSBuffer.data(), m_VStoFSBuffer.data() + m_VStoFSBuffer.size(), VStoFSBuffer);

        cl::Program program = compileProgram(fragmentShaderKernal);

        using vecAdd = cl::KernelFunctor<cl::Buffer, cl::Buffer>;
        vecAdd add_vecs(program, "fragment_shader");

        cl::NDRange GlobalRange(480 * 320);
        cl::NDRange LocalRange(1);
        cl::EnqueueArgs Args(m_queue, GlobalRange, LocalRange);

        cl::Event evt = add_vecs(Args, VStoFSBuffer, output);

        evt.wait();

        cl::copy(m_queue, output, m_resultScreenBuffer_from_gpu.data(),
                 m_resultScreenBuffer_from_gpu.data() + m_resultScreenBuffer_from_gpu.size());

        auto GPUTimeStart = evt.getProfilingInfo<CL_PROFILING_COMMAND_START>();  // in ns
        auto GPUTimeFin = evt.getProfilingInfo<CL_PROFILING_COMMAND_END>();
        auto GDur = (GPUTimeFin - GPUTimeStart) / 1000;  // ns -> µs
        std::cout << " System GPU: Fragment shader pure time measured: " << GDur << " Microseconds" << std::endl;

        TableResults table("Fragment shader", 14, 10, 10);
        table.addDataColumn("SoftRender", m_resultScreenBuffer);
        table.addDataColumn("System GPU", m_resultScreenBuffer_from_gpu);
        table.processAndShow();

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