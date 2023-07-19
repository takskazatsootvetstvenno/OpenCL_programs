#include "Application.hpp"
#include <exception>
#include <iostream>
#include <string>
#include <fstream>
#include <algorithm>

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

    const char* vecAddKernelSRC = R"(
    float4 mult(__constant float* mat, float4 vec)
    {
        float dot0 = dot(vload4(0, mat), vec);
        float dot1 = dot(vload4(1, mat), vec);
        float dot2 = dot(vload4(2, mat), vec);
        float dot3 = dot(vload4(3, mat), vec);

        return (float4)(dot0, dot1, dot2, dot3);
    }

    __kernel void vector_add(__constant float* B, __global float* C) {
        float4 pos = (float4)(2.0f, 7.0f, 1.0f, 4.0f);
        float4 result = mult(B, pos);
	    int i = get_global_id(0);
	    C[0] = result.r;
        C[1] = result.g;
        C[2] = result.b;
        C[3] = result.a;
    }

    )";
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
    void Application::test() {
        constexpr int N = 4;
        constexpr int M = 4;

        cl::vector<cl_float> A_vec(N * M), B_vec(N * M), C_vec(N * M);
        
        cl_float16 mat = {
            2, 6, 0, 0,
            0, 2, 5, 0,
            1, 2, 6, 4,
            0, 0, 0, -5
        };

        std::fill(A_vec.begin(), A_vec.end(), 3.f);
        memcpy(B_vec.data(), &mat, sizeof(cl_float16));

        for (auto el : A_vec) std::cout << el << " ";
        std::cout << std::endl;
        for (auto el : B_vec) std::cout << el << " ";
        std::cout << std::endl;

        size_t bufferSize = A_vec.size() * sizeof(cl_int);

        cl::Buffer A(m_context, CL_MEM_READ_ONLY, bufferSize);
        cl::Buffer B(m_context, CL_MEM_READ_ONLY, bufferSize);
        cl::Buffer C(m_context, CL_MEM_WRITE_ONLY, bufferSize);

        cl::copy(m_queue, A_vec.data(), A_vec.data() + A_vec.size(), A);
        cl::copy(m_queue, B_vec.data(), B_vec.data() + B_vec.size(), B);

        cl::Program program = compileProgram(vecAddKernelSRC);

        using vecAdd = cl::KernelFunctor<cl::Buffer, cl::Buffer>;
        vecAdd add_vecs(program, "vector_add");

        cl::NDRange GlobalRange(A_vec.size());
        cl::NDRange LocalRange(1);
        cl::EnqueueArgs Args(m_queue, GlobalRange, LocalRange);

        cl::Event evt = add_vecs(Args, B, C);

        evt.wait();

        cl::copy(m_queue, C, C_vec.data(), C_vec.data() + A_vec.size());
        
        for (auto el : C_vec) std::cout << el << " "; //should be 9
        std::cout << std::endl;

        auto GPUTimeStart = evt.getProfilingInfo<CL_PROFILING_COMMAND_START>(); // in ns
        auto GPUTimeFin = evt.getProfilingInfo<CL_PROFILING_COMMAND_END>();
        auto GDur = (GPUTimeFin - GPUTimeStart) / 1000;  // ns -> µs
        std::cout << "GPU pure time measured: " << GDur << " Microseconds" << std::endl;
    }

    void Application::loadDataFromDisk() {
        //TO DO - check filling buffers from disk!
        // m_vetexBuffer; m_indexBuffer; m_MVP; m_screenBuffer;
        std::ifstream vertexFile("Vertices.bin", std::ios::binary);
        std::ifstream indexFile("Indices.bin", std::ios::binary);
        std::ifstream MVPFile("MVP.bin", std::ios::binary);
        std::ifstream ScreenMatrixFile("ScreenMatrix.bin", std::ios::binary);

        if (!vertexFile.is_open()) throw std::runtime_error("Can't open Vertices.bin");
        if (!indexFile.is_open()) throw std::runtime_error("Can't open Indices.bin");
        if (!MVPFile.is_open()) throw std::runtime_error("Can't open MVP.bin");
        if (!ScreenMatrixFile.is_open()) throw std::runtime_error("Can't open ScreenMatrix.bin");
        
        std::copy(std::istream_iterator<float>(vertexFile), std::istream_iterator<float>(),
                  std::back_inserter(m_vetexBuffer));
        std::copy(std::istream_iterator<uint32_t>(indexFile), std::istream_iterator<uint32_t>(),
                  std::back_inserter(m_indexBuffer));
        std::copy(std::istream_iterator<float>(MVPFile), std::istream_iterator<float>(),
                  m_MVP.begin());
        std::copy(std::istream_iterator<float>(ScreenMatrixFile), std::istream_iterator<float>(),
                  m_screenBuffer.begin());
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