#include "Application.hpp"

#include "TableResults.hpp"
#include <exception>
#include <filesystem>
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
    cl_context_properties properties[] = {CL_CONTEXT_PLATFORM, reinterpret_cast<cl_context_properties>(p_id), 0};
    return cl::Context(CL_DEVICE_TYPE_GPU, properties);
}

const cl::QueueProperties getQueueProperties() noexcept {
    return cl::QueueProperties::Profiling | cl::QueueProperties::OutOfOrder;
}

template<typename T>
std::vector<T> convertBuffer(const std::vector<uint8_t>& buffer) {
    std::vector<T> convertedBuffer(buffer.size() / sizeof(T));
    std::memcpy(convertedBuffer.data(), buffer.data(), buffer.size());
    return convertedBuffer;
}
}  // namespace

namespace Tester {
Application::Application()
    : m_platform(get_platform()), m_context(get_context(m_platform())), m_queue(m_context, getQueueProperties()) {
    const auto name = m_platform.getInfo<CL_PLATFORM_NAME>();
    const auto profile = m_platform.getInfo<CL_PLATFORM_PROFILE>();
    const auto version = m_platform.getInfo<CL_PLATFORM_VERSION>();
    const auto vendor = m_platform.getInfo<CL_PLATFORM_VENDOR>();
    std::vector<cl_name_version> extentions;
    try {
        extentions = m_platform.getInfo<CL_PLATFORM_EXTENSIONS_WITH_VERSION>();
    } catch (const std::exception& e) { std::cerr << "OpenCL get extentions error: " << e.what() << std::endl; }

    auto end_npos = std::string::npos;
    if (vendor.find("NVIDIA") != end_npos || vendor.find("nvidia") != end_npos) {
        m_vendor = Test::GPUVenderType::NVIDIA;
    }
    if (vendor.find("AMD") != end_npos || vendor.find("amd") != end_npos ||
        vendor.find("Advanced Micro Devices") != end_npos) {
        m_vendor = Test::GPUVenderType::AMD;
    }
    if (vendor.find("INTEL") != end_npos || vendor.find("intel") != end_npos) {
        m_vendor = Test::GPUVenderType::INTEL;
    }
    std::cout << "Selected platform: " << name << "\nVersion: " << version << ", Profile: " << profile
              << "\nVendor:  " << vendor << std::endl
              << std::endl;

    for (const auto& ext : extentions) {
        if (std::string(ext.name) == "cl_khr_fp16") std::cout << "Supported fp16 extention" << std::endl;
    }
}

void Application::parseTestFolder(std::filesystem::path pathToTests) {
    pathToTests.make_preferred();
    if (pathToTests.empty()) { throw std::runtime_error("parseTests: path is empty!"); }
    if (!std::distance(fs::directory_iterator(pathToTests), fs::directory_iterator{})) {
        throw std::runtime_error("parseTests: Directory is empty!\n\tDirectory: " + pathToTests.string());
    }
    for (const auto& entry : fs::directory_iterator(pathToTests)) {
        if (!entry.is_directory()) {
            std::cout << "Warning! \"Tests\" directory contains file!: " << entry.path().filename() << std::endl;
            continue;
        }
        if (!std::distance(fs::directory_iterator(entry), fs::directory_iterator{})) {
            std::cout << "Warning!: Test Directory is empty!\n\tDirectory: " << entry << std::endl;
            continue;
        }
        m_tests.emplace_back(Test::parseTest(entry));
    }
}

std::vector<uint8_t> Application::run_host_gpu(const Test& test) {
    std::vector<cl::Buffer> input_buffers;
    cl::Buffer output_buffer;

    cl::Program program = compileProgram(test.getProgram());
    cl::Kernel kernel;
    try {
        kernel = cl::Kernel(program, test.getName().c_str());
    } catch (const std::exception& e) {
        std::cerr << "Error during kernel creation! Test: " << test.getName() << "\nError : " << e.what() << std::endl;
        return {};
    }

    for (auto& input_info = test.getInputs(); auto& input : input_info) {
        auto& buffer = std::get<2>(input);
        cl::Buffer buf(m_context, CL_MEM_READ_ONLY, buffer.size());
        cl::copy(m_queue, buffer.data(), buffer.data() + buffer.size(), buf);
        input_buffers.emplace_back(std::move(buf));
        kernel.setArg(input_buffers.size() - 1, input_buffers.back());
    }
    auto& output_info = test.getOutputs();
    if (output_info.empty()) {
        std::cout << "Warning: output blobs for test: \"" << test.getName() << "\" are empty !" << std::endl;
        return {};
    }
    auto& buffer = std::get<2>(output_info.front().second);
    output_buffer = cl::Buffer(m_context, CL_MEM_WRITE_ONLY, buffer.size());
    kernel.setArg(input_buffers.size(), output_buffer);

    const auto output_size = std::get<2>(output_info[0].second).size();
    const auto output_type = std::get<1>(output_info[0].second);

    cl::NDRange GlobalRange(output_size / Test::getTypeSize(output_type));
    cl::NDRange LocalRange(1);
    cl::EnqueueArgs Args(m_queue, GlobalRange, LocalRange);

    cl::KernelFunctor functor(kernel);
    cl::Event evt;
    try {
        evt = functor(Args);
        evt.wait();
    } catch (const std::exception& e) {
        std::cerr << "Error during dispatch! Test: " << test.getName() << "\nError : " << e.what() << std::endl;
        return {};
    }

    std::vector<uint8_t> host_result_buffer(output_size);

    cl::copy(m_queue, output_buffer, host_result_buffer.begin(), host_result_buffer.end());

    auto GPUTimeStart = evt.getProfilingInfo<CL_PROFILING_COMMAND_START>();  // in ns
    auto GPUTimeFin = evt.getProfilingInfo<CL_PROFILING_COMMAND_END>();
    auto GDur = (GPUTimeFin - GPUTimeStart) / 1000;  // ns -> µs
    std::cout << "\nTest: " << test.getName() << std::endl;
    std::cout << "System GPU: Vertex shader pure time measured: " << GDur << " Microseconds" << std::endl;

    return host_result_buffer;
}

void Application::runTests() {
    for (auto& test : m_tests) {
        TableResults table(test.getName(), 15, 6, 16);

        auto& outputs = test.getOutputs();
        const auto output_size = std::get<2>(outputs.front().second).size();
        const auto output_type = std::get<1>(outputs.front().second);

        auto addDataColumn = [&](const std::string& name, const std::vector<uint8_t>& buf) {
            switch (output_type) {
                case Test::blob_type::float32: table.addDataColumn(name, convertBuffer<float>(buf)); break;
                case Test::blob_type::uint32: table.addDataColumn(name, convertBuffer<uint32_t>(buf)); break;
                default: break;
            }
        };

        for (auto& output : outputs) { addDataColumn(output.first, std::get<2>(output.second)); }

        //Run test on host device
        if (m_vendor == test.getVenderType()) {
            auto host_result_buffer = run_host_gpu(test);
            if (!host_result_buffer.empty()) { addDataColumn("Host GPU", host_result_buffer); }
        }
        try {
            table.processAndShow();
        } catch (const std::exception& e) {
            std::cout << "TableException, Test: " << test.getName() << std::endl << "Error: "
            << e.what() << std::endl;
        }
    }
}

cl::Program Application::compileProgram(std::string_view kernel) {
    cl::Program program(m_context, kernel.data());
    try {
        program.build("-Werror");  // see https://man.opencl.org/clBuildProgram.html
    } catch (const std::exception& e) {
        std::stringstream ss;
        ss << "\ncompileProgram(..) error: \n";
        ss << "Exception: \"" << e.what() << "\"" << std::endl;
        ss << "Reason:\n";
        auto buildInfo = program.getBuildInfo<CL_PROGRAM_BUILD_LOG>();
        for (auto& [device, error] : buildInfo) {
            ss << "Program build log for device \"" << device.getInfo<CL_DEVICE_NAME>()
               << "\"\nwith compiler arguments: \"" << program.getBuildInfo<CL_PROGRAM_BUILD_OPTIONS>(device)
               << "\"\nKernal: " << program.getInfo<CL_PROGRAM_SOURCE>() << std::endl
               << "Compilation error: \n"
               << error << std::endl;
        }
        throw std::runtime_error(ss.str());
    }
    return program;
}

}  // namespace Tester
