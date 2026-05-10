#include "npy_io.h++"

#include <cstdint>
#include <fstream>
#include <sstream>

static std::vector<char> &operator+=(std::vector<char> &v, const char *s)
{
    while (*s)
    {
        v.push_back(*s);
        ++s;
    }

    return v;
}

template void write_npy_file(const char file_path[], const std::vector<Real> &array);
template void write_npy_file(const char file_path[], const std::vector<Position> &array);
template void write_npy_file(const char file_path[], const std::vector<Vector3D> &array);
template void write_npy_file(const char file_path[], const std::vector<ComplexVector3D> &array);

template <typename T>
void write_npy_file(const char *file_path, const std::vector<T> &array)
{
    std::ofstream output(file_path, std::ios::binary);
    output.exceptions(std::ostream::badbit);

    // Start constructing the header in a memory buffer
    std::vector<uint8_t> header;
    header.reserve(64);

    // Magic number
    header.push_back(0x93);
    const char numpyString[] = "NUMPY";
    for (const char *pc = numpyString; *pc != 0; ++pc)
    {
        header.push_back(*pc);
    }

    // Version 3.0
    header.push_back(0x03);
    header.push_back(0x00);

    output.write(reinterpret_cast<const char *>(header.data()), header.size());

    // Array metadata dictionary
    std::vector<char> dictionary;
    dictionary.reserve(128);

    std::stringstream ss;
    ss << "<f";
    ss << sizeof(Real);

    dictionary += "{'descr':";
    dictionary.push_back('\'');
    dictionary += ss.str().data();
    dictionary.push_back('\'');

    dictionary.push_back(',');
    dictionary += "'fortran_order':False";

    dictionary.push_back(',');
    dictionary += "'shape':";

    ss.str("");
    ss.clear();

    ss << '(' << array.size() << ',' << (sizeof(T) / sizeof(Real)) << ')';

    dictionary += ss.str().data();

    dictionary += "}";

    const auto fixed_header_length = 6 + 2 + 4;
    auto dictionary_length = dictionary.size();
    do
    {
        dictionary.push_back(' ');
        ++dictionary_length;
    } while ((fixed_header_length + dictionary_length) % 64 != 0);

    dictionary.back() = '\n';

    // Header dictionary length
    uint32_t dict_length = dictionary.size();

    output.write(reinterpret_cast<const char *>(&dict_length), 4);

    // Header dictionary contents
    output.write(dictionary.data(), dictionary.size());

    output.write(reinterpret_cast<const char *>(array.data()), array.size() * sizeof(T));

    output.close();
}
