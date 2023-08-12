#pragma once
#include <string_view>
#include <vector>
#include <sstream>
#include <optional>
#include <variant>

namespace Tester {

struct TestStatistic {
    struct Difference {
        float match_percent = 0;
        size_t mismatch_count = 0;
        std::string diff_hash;
    };

    std::vector<Difference> diffs;
    std::optional<size_t> first_wrong_index;
};

class TableResults final {
 public:
    TableResults(std::string column_name, unsigned int cell_width = 14,
                 unsigned int packetSize = 0,
                 unsigned int tableHeight = 12);
    template<typename data_type>
    void addDataColumn(std::string_view column_name, std::vector<data_type> data);
    template<typename data_type>
    void addAdditionalInfoColumn(std::string_view column_name, std::vector<data_type> data);

    void processAndShow();
    void clear();
    using Variant_types_vec = std::variant<
        std::vector<uint64_t>,
        std::vector<int64_t>,
        std::vector<uint32_t>,
        std::vector<int32_t>,
        std::vector<uint16_t>,
        std::vector<int16_t>,
        std::vector<uint8_t>,
        std::vector<int8_t>,
        std::vector<double>,
        std::vector<float>
    >;

 private:

    std::optional<size_t> findFirstMismatch(unsigned int dataSize) const;
    void show(const size_t data_size, TestStatistic&& stats) const;
    void drawRowLine(unsigned int indexSpaceWidth) const;
    void drawSkipLine(unsigned int indexSpaceWidth) const;
    void drawLine(unsigned int indexSpaceWidth) const;
    void drawTextLine(std::string&& str, unsigned int lineSize) const;
    void drawTextLineLeft(std::string&& str, unsigned int lineSize) const;
    void drawNextData(const std::vector<Variant_types_vec>& data, const size_t i) const;
    TestStatistic getStatistics(size_t size);
    std::string getHash(size_t index);
    void reset();
    std::vector<std::string> m_columns_names;
    std::vector<std::string> m_info_columns_names;
    std::vector<Variant_types_vec> m_columns_data;
    std::vector<Variant_types_vec> m_columns_data_unconverted;
    std::vector<Variant_types_vec> m_info_columns_data;
    unsigned int m_cell_width;
    unsigned int m_packet_size;
    unsigned int m_table_height;
    std::string m_table_name;
    mutable std::stringstream m_ss;
};

  template<typename data_type>
inline void TableResults::addDataColumn(std::string_view column_name, std::vector<data_type> data) {
    if (column_name.empty() || data.empty()) throw std::runtime_error("addDataColumn: Empty arguments");
    static_assert(std::is_fundamental_v<data_type>, "TableResult suports only fundamental types!");

    m_columns_names.emplace_back(column_name);
    m_columns_data.emplace_back(data);
}

template<typename data_type>
inline void TableResults::addAdditionalInfoColumn(std::string_view column_name, std::vector<data_type> data) {
    if (column_name.empty() || data.empty()) throw std::runtime_error("addAdditionalInfoColumn: Empty arguments");
    static_assert(std::is_fundamental_v<data_type>, "TableResult suports only fundamental types!");

    m_info_columns_names.emplace_back(column_name);
    m_info_columns_data.emplace_back(data);
}

}  // namespace Tester