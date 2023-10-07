#pragma once
#define CL_HPP_TARGET_OPENCL_VERSION 300
#define CL_HPP_ENABLE_EXCEPTIONS
#include <CL/opencl.hpp>
#include <filesystem>
#include <string_view>
#include <vector>
#include <tuple>

namespace Tester {

class Application {
 public:
    Application();

    void parseTestFolder(std::filesystem::path pathToTests);
    void runTests();

    class Test {
     public:
        enum class blob_type {
            float32,
            uint32
        };
        using input_type = std::tuple<std::string, blob_type, std::vector<uint8_t>>;
        using output_type = std::pair<std::string, std::tuple<std::string, blob_type, std::vector<uint8_t>>>;
        Test(std::filesystem::path&& to_test_path, std::vector<input_type>&& inputs, std::vector<output_type>&& output,
             std::string&& prog, std::string&& test_name);
        const std::vector<input_type>& getInputs() const noexcept { return m_inputs; };
        const std::vector<output_type>& getOutputs() const noexcept { return m_outputs; };
        const std::string& getProgram() const noexcept { return m_opencl_program; };
        const std::string& getName() const noexcept { return m_name; };
        static blob_type getBlobType(std::string_view type);
        static uint32_t getTypeSize(blob_type type);
     private:
        void fillBlobs();
        std::filesystem::path m_to_test_path;
        std::string m_opencl_program;
        std::string m_name;
        std::vector<input_type> m_inputs;
        std::vector<output_type> m_outputs;
    };

 private:
    bool m_supported_gpu = true;
    void parseTest(std::filesystem::path pathToTests);
    std::vector<uint8_t> run_host_gpu(const Test& test);

    cl::Program compileProgram(std::string_view kernal);
    cl::Platform m_platform;
    cl::Context m_context;
    cl::CommandQueue m_queue;

    std::vector<Test> m_tests;
};
}  // namespace Tester
