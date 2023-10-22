#include <json.hpp>
#include <iostream>
#include <fstream>
#include <exception>

#include "TestVector.hpp"

using json = nlohmann::json;

static std::pair<std::ifstream, uint32_t> getFile(const std::filesystem::path path) {
    std::ifstream file(path, std::ios::binary);
    if (!file.is_open()) throw std::runtime_error("Can't open file: " + path.string());
    uint32_t sizeofFile = std::filesystem::file_size(path);
    return {std::move(file), sizeofFile};
}

template<typename T>
static void fillBufferFromFile(std::ifstream& ifs, std::vector<T>& buffer, const uint32_t size) {
    using buffer_type = std::remove_reference_t<decltype(buffer)>::value_type;
    buffer.resize(size / sizeof(buffer_type));
    ifs.read(reinterpret_cast<char*>(buffer.data()), size);
}

namespace Tester {
/*static*/ Test Test::parseTest(std::filesystem::path pathToTest) {
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
    Test::GPUVenderType vender = Test::GPUVenderType::NVIDIA;
    if (data.contains("Disasm")) {
        if (data["Disasm"] == "AMD") { vender = Test::GPUVenderType::AMD; }
        if (data["Disasm"] == "NVIDIA") { vender = Test::GPUVenderType::NVIDIA; }
        if (data["Disasm"] == "INTEL") { vender = Test::GPUVenderType::INTEL; }
    }
    return Test(std::move(pathToTest), std::move(inputs), std::move(outputs), std::move(openclProgram),
            json_file_path.stem().string(), vender);
}

Test::Test(std::filesystem::path&& to_test_path, std::vector<input_type>&& inputs,
                        std::vector<output_type>&& output, std::string&& prog, std::string&& name, GPUVenderType type)
    : m_inputs(std::move(inputs)), m_outputs(std::move(output)), m_to_test_path(std::move(to_test_path)),
      m_opencl_program(std::move(prog)), m_name(std::move(name)), m_vendor(type) {
    fillBlobs();
}

void Test::fillBlobs() {
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

Test::blob_type Test::getBlobType(std::string_view type) {
    std::string string_type(type);
    static std::unordered_map<std::string, Test::blob_type> map = {{"float32", blob_type::float32},
                                                                   {"uint32", blob_type::uint32}};
    if (!map.contains(string_type)) {
        throw std::runtime_error("Wrong blob type! String type: \"" + std::string(type) + "\"");
    }
    return map[string_type];
}

uint32_t Test::getTypeSize(const blob_type type) {
    switch (type) {
        case Test::blob_type::float32: return sizeof(float);
        case Test::blob_type::uint32: return sizeof(uint32_t);
        default: return 4;
    }
}

}