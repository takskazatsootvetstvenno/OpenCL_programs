#include <iostream>
#include <exception>
#include <format>
#include <algorithm>
#include <string>

#include "TableResults.hpp"
#include "hashpp.h"

namespace {
template<typename T>
void changeVectorsType(std::vector<Tester::TableResults::Variant_types_vec>& columns) {
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
                using type_info = std::decay_t<decltype(col)>::value_type;
                if constexpr (std::is_floating_point_v<type_info>) {
                    m_ss << std::format("{:^{}.4f}", col[i], m_cell_width);
                } else {
                    m_ss << std::format("{:^{}}", col[i], m_cell_width);
                }
            },
            column);
    }
}
std::string TableResults::getHash(size_t index) {

    std::vector<double> data;
    std::visit([&data](const auto& arg) { std::copy(arg.cbegin(), arg.cend(), std::back_inserter(data)); },
               m_columns_data.front());
    std::visit([&data](const auto& arg) { std::copy(arg.cbegin(), arg.cend(), std::back_inserter(data)); },
               m_columns_data[index]);

    constexpr int multiplier = sizeof(double) / sizeof(char);
    const char* const data_begin = reinterpret_cast<char*>(data.data());
    std::string data_for_hash = {data_begin, data_begin + data.size() * 8};
    const auto hash = hashpp::get::getHash(hashpp::ALGORITHMS::MD5, data_for_hash);
    std::string result = hash.getString();
    return result;
}

TestStatistic TableResults::getStatistics(size_t data_size) {
    TestStatistic st{};
    st.first_wrong_index = findFirstMismatch(data_size);
    for (int column_id = 1; column_id < m_columns_data.size(); ++column_id) {
        TestStatistic::Difference diff;
        std::vector<std::pair<size_t, double>> mismatch;
        for (size_t i = 0; i < data_size; ++i) {
            std::visit(
                [this, &diff, &mismatch, i](const auto& col) {
                    using T = std::decay_t<decltype(col)>::value_type;
                    if constexpr (std::is_floating_point_v<T>) {
                        const auto golden = std::get<std::vector<T>>(m_columns_data.front())[i];
                        if (std::fabs(golden - col[i]) > std::numeric_limits<T>::epsilon()) {
                            diff.mismatch_count++;
                            mismatch.push_back({i, std::fabs(golden - col[i])});
                        }
                    } else {
                        const auto golden = std::get<std::vector<T>>(m_columns_data.front())[i];
                        if (golden - col[i] != 0) {
                            diff.mismatch_count++;
                            mismatch.push_back({i, golden - col[i]});
                        }
                    }
                },
                m_columns_data[column_id]);
        }
        diff.match_percent = 100.0 * (1.0 - float(diff.mismatch_count) / data_size);
        if (diff.mismatch_count != 0) { diff.diff_hash = getHash(column_id); }
        st.diffs.emplace_back(diff);
    }
    return st;
}

void TableResults::show(const size_t data_size, TestStatistic&& stats) const {
    const unsigned int index_space_width = std::max(static_cast<int>(std::floor(log10(data_size))) + 1, 4);
    const unsigned int line_size = m_cell_width * m_columns_names.size() + index_space_width + m_columns_names.size();

    drawLine(index_space_width);

    if (!stats.first_wrong_index.has_value()) {
        std::string str = std::format("{}: PASS", m_table_name);
        drawTextLine(std::move(str), line_size);
        drawLine(index_space_width);
        std::cout << m_ss.str();
        return;
    }

    {
        std::string str = std::format("TEST: {}", m_table_name);
        drawTextLine(std::move(str), line_size);
        drawLine(index_space_width);
    }

    m_ss << std::format("|{:^{}}", "Id", index_space_width);
    for (const auto& column_name : m_columns_names) { m_ss << std::format("|{:^{}}", column_name, m_cell_width); }
    for (const auto& column_name : m_info_columns_names) {
         m_ss << std::format("|{:^{}}", column_name, m_cell_width);
    }
    if (m_info_columns_names.empty()) { m_ss << "|"; }
    m_ss << "\n";
    drawRowLine(index_space_width);
    size_t max_size = std::max(m_table_height, 1u);
    if (stats.first_wrong_index.has_value() && stats.first_wrong_index.value() != 0) {
        max_size += stats.first_wrong_index.value();
        drawSkipLine(index_space_width);
    }

    for (size_t i = stats.first_wrong_index.value_or(0); i < data_size && i < max_size; ++i) {
        m_ss << std::format("|{:^{}}", i, index_space_width);
        if (!m_columns_data_unconverted.empty()) {
            drawNextData(m_columns_data_unconverted, i);
        } else {
            drawNextData(m_columns_data, i);
        }
        drawNextData(m_info_columns_data, i);
        if (m_info_columns_data.empty()) {
            m_ss << "|";
        }

        m_ss << "\n";

        if (i == max_size - 1 && i < data_size) { drawSkipLine(index_space_width); }

        if (m_packet_size == 0) continue;

        if ((i + 1) % m_packet_size == 0 && (i + 1) != data_size) { drawRowLine(index_space_width); }
    }
    drawLine(index_space_width);
    {
        std::string str = std::format("{}: DIFF", m_table_name);
        drawTextLine(std::move(str), line_size);
        drawTextLine("Blobs are NOT equal!", line_size);
    }
    drawLine(index_space_width);
    m_ss << std::endl;

    m_ss <<  std::format("|{:-^{}}|\n", "", 49);
    for (int i = 1; i < m_columns_names.size(); ++i) {
        std::string str = std::format("Golden \"{}\" vs \"{}\"", m_columns_names.front(), m_columns_names[i]);
        drawTextLine(std::move(str), 49); 
        drawTextLine("", 49); 
        drawTextLineLeft(std::format(" Diff hash  :   {}", stats.diffs[i - 1].diff_hash), 49);
        drawTextLineLeft(std::format(" Data size  :   {}", data_size), 49);
        drawTextLineLeft(std::format(" Diff count :   {}", stats.diffs[i - 1].mismatch_count), 49);
        drawTextLineLeft(std::format(" Match :        {:.2f} %", stats.diffs[i - 1].match_percent), 49);
        if (i + 1 == m_columns_names.size()) break;
        drawTextLine("", 49); 
    }

    m_ss <<  std::format("|{:-^{}}|\n", "", 49);
    std::cout << m_ss.str() << std::endl;
    }
TableResults::TableResults(std::string table_name, unsigned int cell_width, unsigned int packetSize,
                                      unsigned int tableHeight)
    : m_cell_width(cell_width), m_packet_size(packetSize), m_table_name(std::move(table_name)),
      m_table_height(tableHeight) {}

void TableResults::processAndShow() {
    if (m_columns_data.empty()) { throw std::runtime_error("Table.processAndShow(): Empty columns data"); }
    if (m_columns_data.size() == 1) { throw std::runtime_error("Table.processAndShow(): Only one data row"); }
    reset();

    const size_t data_size = std::visit([](const auto& vec) { return vec.size(); }, m_columns_data.front());
    if (!std::all_of(m_columns_data.cbegin(), m_columns_data.cend(),
        [data_size](const auto& col) { return std::visit([](const auto& vec) { return vec.size(); }, col) == data_size; })) {
            throw std::runtime_error("Table.show(): Vectors should have equal sizes!");
        }
    if (!std::all_of(m_info_columns_data.cbegin(), m_info_columns_data.cend(), [data_size](const auto& col) {
            return std::visit([](const auto& vec) { return vec.size(); }, col) == data_size;
        })) {
            throw std::runtime_error("Table.show(): Additional Vectors should have equal sizes!");
    }
    const bool all_types_are_equal = std::all_of(
        m_columns_data.cbegin(), m_columns_data.cend(),
        [&](const auto& col) { return col.index() == (m_columns_data.front()).index(); });

    if (all_types_are_equal) {
        show(data_size, getStatistics(data_size));
        return;
    }

    m_columns_data_unconverted = m_columns_data;

    const bool is_float_present = std::any_of(m_columns_data.cbegin(), m_columns_data.cend(), [](const auto& col) {
        return std::visit(
            [](const auto& arg) {
                using T = std::decay_t<decltype(arg)>::value_type;
                return std::is_floating_point_v<T>;
            },
            col);
    });

    if (is_float_present) {   
        changeVectorsType<double>(m_columns_data);
        std::cout << "\n* Table: all data types are converted to double! *\n";
    } else {
        changeVectorsType<int64_t>(m_columns_data);
        std::cout << "\n* Table: all data types are converted to int64! *\n";
    }
    show(data_size, getStatistics(data_size));
}

std::optional<size_t> TableResults::findFirstMismatch(unsigned int dataSize) const {   
    std::optional<size_t> firstWrongIndex;
    for (size_t i = 0; i < dataSize && firstWrongIndex.has_value() == false; ++i) {
        for (const auto& column : m_columns_data) {
            std::visit([this, &firstWrongIndex, i](const auto& col) {
                using T = std::decay_t<decltype(col)>::value_type;
                if constexpr (std::is_floating_point_v<T>) {
                     const auto value = std::get<std::vector<T>>(m_columns_data[0])[i];
                     if (std::fabs(value - col[i]) > std::numeric_limits<T>::epsilon()) {
                         firstWrongIndex = i;
                     }
                } else {
                     const auto value = std::get<std::vector<T>>(m_columns_data[0])[i];
                     if (value - col[i] != 0) { firstWrongIndex = i; }
                }
                },
                column);
        }
    }
    return firstWrongIndex;
}

void TableResults::drawRowLine(unsigned int index_space_width) const {
    m_ss << std::format("|{:-^{}}", "", index_space_width);
    for (auto i = 0; i < m_columns_names.size(); ++i){
        m_ss << std::format("|{:-^{}}", "", m_cell_width); 
    }
    m_ss << "|\n";
}

void TableResults::drawSkipLine(unsigned int index_space_width) const {
    m_ss << std::format("|{:^{}}", "..", index_space_width);
    for (auto i = 0; i < m_columns_names.size(); ++i) { m_ss << std::format("|{:^{}}", "...", m_cell_width); }
    m_ss << "|\n";
}

void TableResults::drawLine(unsigned int index_space_width) const {
    m_ss << std::format("|{:-^{}}|\n", "",
                        m_cell_width * m_columns_names.size() + m_columns_names.size() + index_space_width);
}

void TableResults::drawTextLine(std::string&& str, unsigned int lineSize) const {
    m_ss << std::format("|{:^{}}|\n", std::move(str), lineSize);
}
void TableResults::drawTextLineLeft(std::string&& str, unsigned int lineSize) const {
    m_ss << std::format("|{:<{}}|\n", std::move(str), lineSize);
}

void TableResults::clear() {
    m_columns_names.clear();
    m_info_columns_names.clear();
    m_columns_data.clear();
    m_info_columns_data.clear();
    reset();
}

void TableResults::reset() {
    m_ss.clear(); 
    m_columns_data_unconverted.clear();
}
}