#include <iostream>
#include <exception>
#include <string>
#include <format>
#include <algorithm>

#include "TableResults.hpp"
  
namespace {
template<typename T>
void changeVectorType(std::vector<Tester::TableResults::Variant_types_vec>& columns) {
    for (auto& col : columns) {
        std::vector<T> new_vec;
        std::visit([&new_vec](const auto& arg) { std::copy(arg.cbegin(), arg.cend(), std::back_inserter(new_vec)); },
                   col);
        col = std::move(new_vec);
    }
}
}  // namespace

namespace Tester {

void TableResults::drawNextData(const std::vector<Variant_types_vec>& data, const size_t i) const {
    for (const auto& column : data) {
        m_ss << "|";
        std::visit([this, i](const auto& col) {
                using type_info = std::remove_reference_t<decltype(col)>::value_type;
                if constexpr (std::is_floating_point_v<type_info>) {
                    m_ss << std::format("{:^{}.4f}", col[i], m_cell_width);
                } else {
                    m_ss << std::format("{:^{}}", col[i], m_cell_width);
                }
            },
            column);
    }
}

void TableResults::show(const size_t data_size) const {
    std::optional<size_t> firstWrongIndex;
    firstWrongIndex = findFirstMismatch(data_size);

    const unsigned int indexSpaceWidth = std::max(static_cast<int>(std::floor(log10(data_size))) + 1, 4);
    const unsigned int line_size = m_cell_width * m_columns_names.size() + indexSpaceWidth + m_columns_names.size();

    drawLine(indexSpaceWidth);

    if (!firstWrongIndex.has_value()) {
        std::string str = std::format("{}: PASS", m_table_name);
        drawTextLine(std::move(str), line_size);
        drawLine(indexSpaceWidth);
        std::cout << m_ss.str();
        return;
    }

    {
        std::string str = std::format("TEST: {}", m_table_name);
        drawTextLine(std::move(str), line_size);
        drawLine(indexSpaceWidth);
    }

    m_ss << std::format("|{:^{}}", "Id", indexSpaceWidth);
    for (const auto& column_name : m_columns_names) { m_ss << std::format("|{:^{}}", column_name, m_cell_width); }
    for (const auto& column_name : m_info_columns_names) {
         m_ss << std::format("|{:^{}}", column_name, m_cell_width);
    }
    if (m_info_columns_names.empty()) { m_ss << "|"; }
    m_ss << "\n";
    drawRowLine(indexSpaceWidth);
    size_t max_size = std::max(m_table_height, 1u);
    if (firstWrongIndex.has_value() && firstWrongIndex.value() != 0) {
        max_size += firstWrongIndex.value();
        drawSkipLine(indexSpaceWidth);
    }


    for (size_t i = firstWrongIndex.value_or(0); i < data_size && i < max_size; ++i) {
        m_ss << std::format("|{:^{}}", i, indexSpaceWidth);
        drawNextData(m_columns_data, i);
        drawNextData(m_info_columns_data, i);
        if (m_info_columns_data.empty()) {
            m_ss << "|";
        }

        m_ss << "\n";

        if (i == max_size - 1 && i < data_size) { drawSkipLine(indexSpaceWidth); }

        if (m_packet_size == 0) continue;

        if ((i + 1) % m_packet_size == 0 && (i + 1) != data_size) { drawRowLine(indexSpaceWidth); }
    }
    drawLine(indexSpaceWidth);
    {
        std::string str = std::format("{}: DIFF", m_table_name);
        drawTextLine(std::move(str), line_size);
        drawTextLine("Blobs are NOT equal!", line_size);
    }
    drawLine(indexSpaceWidth);
    std::cout << m_ss.str() << std::endl;
    
    }
TableResults::TableResults(std::string table_name, unsigned int cell_width, unsigned int packetSize,
                                      unsigned int tableHeight)
    : m_cell_width(cell_width), m_packet_size(packetSize), m_table_name(std::move(table_name)),
      m_table_height(tableHeight) {}

void TableResults::processAndShow() {
    if (m_columns_data.empty()) { throw std::runtime_error("Table.processAndShow(): Empty m_columns_data"); }
    m_ss.clear();

    const size_t data_size = std::visit([](const auto& vec) { return vec.size(); }, m_columns_data[0]);
    if (!std::all_of(m_columns_data.cbegin(), m_columns_data.cend(),
        [data_size](const auto& col) { return std::visit([](const auto& vec) { return vec.size(); }, col) == data_size; })) {
            throw std::runtime_error("Table.show(): Vectors should have equal sizes!");
        }
    const bool is_float_present = std::any_of(m_columns_data.begin(), m_columns_data.end(), [](const auto& col) {
        return std::visit(
            [](const auto& arg) {
                using T = std::remove_reference_t<decltype(arg)>::value_type;
                return std::is_floating_point_v<T>;
            },
            col);
    });

    if (is_float_present) {   
        changeVectorType<double>(m_columns_data);
        std::cout << "\n* Table: all data types are converted to double! *\n";
    } else {
        changeVectorType<int64_t>(m_columns_data);
        std::cout << "\n* Table: all data types are converted to int64! *\n";
    }
    show(data_size);
}

std::optional<size_t> TableResults::findFirstMismatch(unsigned int dataSize) const {   
    std::optional<size_t> firstWrongIndex;
    for (size_t i = 0; i < dataSize && firstWrongIndex.has_value() == false; ++i) {
        for (const auto& column : m_columns_data) {
            std::visit([this, &firstWrongIndex, i](const auto& col) {
                using T = std::remove_reference_t<decltype(col)>::value_type;
                if constexpr (std::is_floating_point_v<T>) {
                     const auto value = std::get<std::vector<double>>(m_columns_data[0])[i];
                     if (std::fabs(value - col[i]) > std::numeric_limits<float>::epsilon()) {
                         firstWrongIndex = i;
                     }
                } else {
                     const auto value = std::get<std::vector<int64_t>>(m_columns_data[0])[i];
                     if (value - col[i] != 0) { firstWrongIndex = i; }
                }
                },
                column);
        }
    }
    return firstWrongIndex;
}

void TableResults::drawRowLine(unsigned int indexSpaceWidth) const {
    m_ss << std::format("|{:-^{}}", "", indexSpaceWidth);
    for (auto i = 0; i < m_columns_names.size(); ++i){
        m_ss << std::format("|{:-^{}}", "", m_cell_width); 
    }
    m_ss << "|\n";
}

void TableResults::drawSkipLine(unsigned int indexSpaceWidth) const {
    m_ss << std::format("|{:^{}}", "..", indexSpaceWidth);
    for (auto i = 0; i < m_columns_names.size(); ++i) { m_ss << std::format("|{:^{}}", "...", m_cell_width); }
    m_ss << "|\n";
}

void TableResults::drawLine(unsigned int indexSpaceWidth) const {
    m_ss << std::format("|{:-^{}}|\n", "",
                        m_cell_width * m_columns_names.size() + m_columns_names.size() + indexSpaceWidth);
}

void TableResults::drawTextLine(std::string&& str, unsigned int lineSize) const {
    m_ss << std::format("|{:^{}}|\n", std::move(str), lineSize);
}

}