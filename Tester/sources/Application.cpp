#include "Application.hpp"

#include "TableResults.hpp"

#include <json.hpp>

#include <exception>
#include <filesystem>
#include <fstream>
#include <iostream>
#include <string>

namespace fs = std::filesystem;
using json = nlohmann::json;

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

std::pair<std::ifstream, uint32_t> getFile(const std::filesystem::path path) {
    std::ifstream file(path, std::ios::binary);
    if (!file.is_open()) throw std::runtime_error("Can't open file: " + path.string());
    uint32_t sizeofFile = std::filesystem::file_size(path);
    return {std::move(file), sizeofFile};
}

template<typename T>
void fillBufferFromFile(std::ifstream& ifs, std::vector<T>& buffer, const uint32_t size) {
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
std::vector<uint32_t> getExpectedIndex(const std::vector<uint32_t>& index) {
    std::vector<uint32_t> result;
    for (const auto i : index) {
        for (int j = 0; j < 6; ++j) result.emplace_back(i);
    }
    return result;
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
    const auto extentions = m_platform.getInfo<CL_PLATFORM_EXTENSIONS_WITH_VERSION>();

    if (vendor.find("NVIDIA") != std::string::npos || vendor.find("nvidia") != std::string::npos) {
        m_vendor = GPUVenderType::NVIDIA;
    }
    if (vendor.find("AMD") != std::string::npos || vendor.find("amd") != std::string::npos ||
        vendor.find("Advanced micro devices") != std::string::npos) {
        m_vendor = GPUVenderType::AMD;
    }
    if (vendor.find("INTEL") != std::string::npos || vendor.find("intel") != std::string::npos) {
        m_vendor = GPUVenderType::INTEL;
    }
    std::cout << "Selected platform: " << name << "\nVersion: " << version << ", Profile: " << profile
              << "\nVendor:  " << vendor << std::endl
              << std::endl;

    for (const auto& ext : extentions)
        if (std::string(ext.name) == "cl_khr_fp16") std::cout << "Supported fp16 extention" << std::endl;
}

void Application::parseTest(std::filesystem::path pathToTest) {
    std::vector<fs::path> files;
    for (const auto& test_files : fs::directory_iterator(pathToTest)) {
        if (test_files.is_directory()) {
            std::cout << "Warning! Test directory contains folder!: " << test_files.path().filename() << std::endl;
            continue;
        }
        files.emplace_back(test_files);
    }
    auto contain_json = [](const fs::path& path) { return path.extension() == ".json"; };
    auto contain_opencl = [](const fs::path& path) { return path.extension() == ".cl"; };
    {
        auto json_files_in_folder = std::count_if(files.begin(), files.end(), contain_json);
        if (json_files_in_folder > 2) {
            throw std::runtime_error("Error: Only one json file should be in test folder!");
        }
        if (json_files_in_folder == 0) { throw std::runtime_error("Error: Can't find .json file in test folder!"); }
    }
    {
        auto cl_files_in_folder = std::count_if(files.begin(), files.end(), contain_opencl);
        if (cl_files_in_folder > 2) {
            throw std::runtime_error("Error: Only one OpenCl programm should be in test folder!");
        }
        if (cl_files_in_folder == 0) { throw std::runtime_error("Error: Can't find .cl file in test folder!"); }
    }
    auto cl_file_path = *(std::find_if(files.begin(), files.end(), contain_opencl));
    std::ifstream opencl_prog_file(cl_file_path);
    if (!opencl_prog_file) {
        throw std::runtime_error("Error: Can't open opencl programm!\nPath: " + cl_file_path.string());
    }
    std::string openclProgram;
    auto openClFileSize = fs::file_size(cl_file_path);
    openclProgram.resize(openClFileSize + 1);
    opencl_prog_file.read(reinterpret_cast<char*>(openclProgram.data()), openClFileSize);

    auto json_file_path = *(std::find_if(files.begin(), files.end(), contain_json));
    std::ifstream json_file(json_file_path);
    if (!json_file) { throw std::runtime_error("Error: Can't open json file!\nPath: " + json_file_path.string()); }

    json data = json::parse(json_file);
    std::vector<Test::input_type> inputs;
    std::vector<Test::output_type> outputs;

    for (json input = data["Inputs"]; auto& binary : input) {
        if (binary.empty()) { continue; }
        auto it = binary.cbegin();
        Test::input_type input = {it.key(), Test::getBlobType(it.value()), {}};
        inputs.emplace_back(std::move(input));
    }
    for (json output = data["Outputs"]; auto& from : output) {
        if (from.empty()) { continue; }
        auto it = from.cbegin();
        std::string from_name = it.key();
        auto it_bin = it.value().cbegin();
        Test::output_type output = {from_name, {it_bin.key(), Test::getBlobType(it_bin.value()), {}}};
        outputs.emplace_back(std::move(output));
    }
    GPUVenderType vender = GPUVenderType::NVIDIA;
    if (data.contains("Disasm")) {
        if (data["Disasm"] == "AMD") {
            vender = GPUVenderType::AMD;
        }
        if (data["Disasm"] == "NVIDIA") { vender = GPUVenderType::NVIDIA; }
        if (data["Disasm"] == "INTEL") { vender = GPUVenderType::INTEL; }
    }
    m_tests.emplace_back(std::move(pathToTest), std::move(inputs), std::move(outputs), std::move(openclProgram),
                         json_file_path.stem().string(), vender);
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
        parseTest(entry);
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

Application::Test::Test(std::filesystem::path&& to_test_path, std::vector<input_type>&& inputs,
                        std::vector<output_type>&& output, std::string&& prog, std::string&& name, GPUVenderType type)
    : m_inputs(std::move(inputs)), m_outputs(std::move(output)), m_to_test_path(std::move(to_test_path)),
      m_opencl_program(std::move(prog)), m_name(std::move(name)), m_vendor(type) {
    fillBlobs();
}

void Application::Test::fillBlobs() {
    for (auto& input : m_inputs) {
        auto [ifsteam, file_size] = getFile(m_to_test_path / std::get<0>(input));
        fillBufferFromFile(ifsteam, std::get<2>(input), file_size);
    }

    for (auto& output : m_outputs) {
        auto [ifsteam, file_size] = getFile(m_to_test_path / std::get<0>(output.second));
        fillBufferFromFile(ifsteam, std::get<2>(output.second), file_size);
    }

    if (m_outputs.empty()) return;

    auto first_blob_size = std::get<2>(m_outputs.front().second).size();
    bool equal_size = std::all_of(m_outputs.begin(), m_outputs.end(),
                                  [&](auto& output) { return first_blob_size == std::get<2>(output.second).size(); });
    if (!equal_size) { throw std::runtime_error("All output blobs should have equal sizes! Test:" + m_name); }
}

Application::Test::blob_type Application::Test::getBlobType(std::string_view type) {
    std::string string_type(type);
    static std::unordered_map<std::string, Test::blob_type> map = {{"float32", blob_type::float32},
                                                                   {"uint32", blob_type::uint32}};
    if (!map.contains(string_type)) {
        throw std::runtime_error("Wrong blob type! String type: \"" + std::string(type) + "\"");
    }
    return map[string_type];
}

uint32_t Application::Test::getTypeSize(const blob_type type) {
    switch (type) {
        case Test::blob_type::float32: return sizeof(float);
        case Test::blob_type::uint32: return sizeof(uint32_t);
        default: return 4;
    }
}

}  // namespace Tester
