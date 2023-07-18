#include "Application.hpp"
#include <exception>
#include <iostream>
#include <string>

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

    __kernel void vector_add(__global int* A, __global int* B, __global int* C) {
	    int i = get_global_id(0);
	    C[i] = A[i] + B[i];
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
        constexpr int N = 3;
        constexpr int M = 3;

        cl::vector<cl_int> A_vec(N * M), B_vec(N * M), C_vec(N * M);

        std::fill(A_vec.begin(), A_vec.end(), 3);
        std::fill(B_vec.begin(), B_vec.end(), 6);

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

        using vecAdd = cl::KernelFunctor<cl::Buffer, cl::Buffer, cl::Buffer>;
        vecAdd add_vecs(program, "vector_add");

        cl::NDRange GlobalRange(A_vec.size());
        cl::NDRange LocalRange(1);
        cl::EnqueueArgs Args(m_queue, GlobalRange, LocalRange);

        cl::Event evt = add_vecs(Args, A, B, C);

        evt.wait();

        cl::copy(m_queue, C, C_vec.data(), C_vec.data() + A_vec.size());
        
        for (auto el : C_vec) std::cout << el << " "; //should be 9
        std::cout << std::endl;
    }

    cl::Program Application::compileProgram(std::string_view kernel) { 
        cl::Program program(m_context, kernel.data());
        try {
            cl_int buildErr = program.build();
        } catch (const std::exception& e) {
            std::stringstream ss;
            ss << "\ncompileProgram(..) error: \n";
            ss << "Exception: \"" << e.what() << "\"" << std::endl;
            ss << "Reason:\n";
            auto buildInfo = program.getBuildInfo<CL_PROGRAM_BUILD_LOG>();
            for (auto& [device, error] : buildInfo) {
                ss << "Program build log for device \"" << device.getInfo<CL_DEVICE_NAME>()
                   << "\"\nwith compiler arguments: " << program.getBuildInfo<CL_PROGRAM_BUILD_OPTIONS>(device)
                   << std::endl
                   << "Compilation error: \n"
                   << error << std::endl;
            }
            throw std::runtime_error(ss.str());
        }
        return program; 
    }

    
}  // namespace Tester