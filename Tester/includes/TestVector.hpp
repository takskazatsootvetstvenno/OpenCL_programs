#pragma once

#include <vector>
#include <string>
#include <filesystem>

namespace fs = std::filesystem;

namespace Tester {
class Test {
 public:
    enum class GPUVenderType { AMD, NVIDIA, INTEL };
    enum class blob_type { float32, uint32 };
    using input_type = std::tuple<std::string, blob_type, std::vector<uint8_t>>;
    using output_type = std::pair<std::string, std::tuple<std::string, blob_type, std::vector<uint8_t>>>;
    Test(std::filesystem::path&& to_test_path, std::vector<input_type>&& inputs, std::vector<output_type>&& output,
         std::string&& prog, std::string&& test_name, GPUVenderType type);
    const std::vector<input_type>& getInputs() const noexcept { return m_inputs; };
    const std::vector<output_type>& getOutputs() const noexcept { return m_outputs; };
    const std::string& getProgram() const noexcept { return m_opencl_program; };
    const std::string& getName() const noexcept { return m_name; };
    static blob_type getBlobType(std::string_view type);
    static uint32_t getTypeSize(blob_type type);
    GPUVenderType getVenderType() const { return m_vendor; };
    static Test parseTest(std::filesystem::path pathToTest);

 private:
    void fillBlobs();
    GPUVenderType m_vendor = GPUVenderType::NVIDIA;
    std::filesystem::path m_to_test_path;
    std::string m_opencl_program;
    std::string m_name;
    std::vector<input_type> m_inputs;
    std::vector<output_type> m_outputs;
};
}  // namespace Tester