#pragma once
#include <string>
#include <vector>
#include <sstream>
#include <optional>
#include <variant>
#include <memory>

namespace Tester {


class TableResults final {
 public:
    TableResults(std::string column_name, unsigned int cell_width = 14,
                 unsigned int packetSize = 0,
                 unsigned int tableHeight = 12);
    template<typename data_type>
    void addDataColumn(std::string&& column_name, std::vector<data_type> data);
    template<typename data_type>
    void addAdditionalInfoColumn(std::string&& column_name, std::vector<data_type> data);

    void processAndShow();

    using Variant_types_vec = std::variant<std::vector<int64_t>, std::vector<double>, std::vector<uint32_t>, std::vector<float>>;
 private:

    std::optional<size_t> findFirstMismatch(unsigned int dataSize) const;
    void show(const size_t data_size) const;
    void drawRowLine(unsigned int indexSpaceWidth) const;
    void drawSkipLine(unsigned int indexSpaceWidth) const;
    void drawLine(unsigned int indexSpaceWidth) const;
    void drawTextLine(std::string&& str, unsigned int lineSize) const;
    void drawNextData(const std::vector<Variant_types_vec>& data, const size_t i) const;
    std::vector<std::string> m_columns_names;
    std::vector<std::string> m_info_columns_names;
    std::vector<Variant_types_vec> m_columns_data;
    std::vector<Variant_types_vec> m_info_columns_data;
    unsigned int m_cell_width;
    unsigned int m_packet_size;
    unsigned int m_table_height;
    std::string m_table_name;
    mutable std::stringstream m_ss;
};

  template<typename data_type>
inline void TableResults::addDataColumn(std::string&& column_name, std::vector<data_type> data) {
    if (column_name.empty() || data.empty()) throw std::runtime_error("addDataColumn: Empty arguments");
    static_assert(std::is_fundamental_v<data_type>, "TableResult suports only fundamental types!");

    m_columns_names.emplace_back(std::move(column_name));
    m_columns_data.emplace_back(data);
}

template<typename data_type>
inline void TableResults::addAdditionalInfoColumn(std::string&& column_name, std::vector<data_type> data) {
    if (column_name.empty() || data.empty()) throw std::runtime_error("addAdditionalInfoColumn: Empty arguments");
    static_assert(std::is_fundamental_v<data_type>, "TableResult suports only fundamental types!");

    m_info_columns_names.emplace_back(std::move(column_name));
    m_info_columns_data.emplace_back(data);
}

}  // namespace Tester