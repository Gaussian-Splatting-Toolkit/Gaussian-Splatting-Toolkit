// Single file header-only library for loading STL 3D models in either
// ASCII or binary formats.
//
// MIT License
//
// Copyright (c) 2019 David Cunningham
//
// Permission is hereby granted, free of charge, to any person obtaining a copy
// of this software and associated documentation files (the "Software"), to deal
// in the Software without restriction, including without limitation the rights
// to use, copy, modify, merge, publish, distribute, sublicense, and/or sell
// copies of the Software, and to permit persons to whom the Software is
// furnished to do so, subject to the following conditions:
//
// The above copyright notice and this permission notice shall be included in
// all copies or substantial portions of the Software.
//
// THE SOFTWARE IS PROVIDED "AS IS", WITHOUT WARRANTY OF ANY KIND, EXPRESS OR
// IMPLIED, INCLUDING BUT NOT LIMITED TO THE WARRANTIES OF MERCHANTABILITY,
// FITNESS FOR A PARTICULAR PURPOSE AND NONINFRINGEMENT. IN NO EVENT SHALL THE
// AUTHORS OR COPYRIGHT HOLDERS BE LIABLE FOR ANY CLAIM, DAMAGES OR OTHER
// LIABILITY, WHETHER IN AN ACTION OF CONTRACT, TORT OR OTHERWISE, ARISING FROM,
// OUT OF OR IN CONNECTION WITH THE SOFTWARE OR THE USE OR OTHER DEALINGS IN THE
// SOFTWARE.
#include <fstream>
#include <iostream>
#include <sstream>
#include <vector>

namespace stlloader {

////////////////////////////////////////////////////////////////////////
// External Interface
////////////////////////////////////////////////////////////////////////

struct Vertex {
  float x, y, z;
};
struct Normal {
  float x, y, z;
};

struct Facet {
  Vertex vertices[3];
  Normal normal;
};
void print(const Facet &facet);

struct Mesh {
  std::vector<Facet> facets;
  std::string name;
  std::string header;
};
void print(const Mesh &mesh);

void parse_stream(std::istream &is, Mesh &mesh);
void parse_stream(const char *filename, Mesh &mesh);

enum class Format { ascii, binary };

void write_stream(std::ostream &os, const Mesh &mesh,
                  Format format = Format::ascii);
void write_file(const char *filename, const Mesh &mesh,
                Format format = Format::ascii);

#ifdef STLLOADER_IMPLEMENTATION

////////////////////////////////////////////////////////////////////////
// Printing
////////////////////////////////////////////////////////////////////////

std::istream &operator>>(std::istream &is, Vertex &v) {
  return is >> v.x >> v.y >> v.z;
}
std::istream &operator>>(std::istream &is, Normal &n) {
  return is >> n.x >> n.y >> n.z;
}

void print(const Facet &facet) {
  auto &n = facet.normal;
  printf("n %f %f %f\n", n.x, n.y, n.z);
  for (int i = 0; i < 3; ++i) {
    auto &v = facet.vertices[i];
    printf(" v%d %f %f %f\n", i, v.x, v.y, v.z);
  }
}

void print(const Mesh &mesh) {
  std::cout << "name: " << mesh.name << '\n';
  std::cout << "header: " << mesh.header << '\n';
  std::cout << "num facets: " << mesh.facets.size() << '\n';
  for (auto &facet : mesh.facets) {
    print(facet);
  }
}

////////////////////////////////////////////////////////////////////////
// Reading
////////////////////////////////////////////////////////////////////////

void consume(std::istream &is, const std::string &expected) {
  std::string line, label;
  std::getline(is, line);
  std::istringstream ss(line);
  ss >> label;
  if (label != expected)
    throw;
}

std::istringstream getlinestream(std::istream &is) {
  std::string line;
  std::getline(is, line);
  return std::istringstream(line);
}

void parse_ascii_facet(std::istream &is, Facet &facet) {
  std::string label, sub;
  auto ss = getlinestream(is);
  ss >> label >> sub;
  if (label == "outer" && sub == "loop") {
    int vi = 0;
    do {
      auto ss = getlinestream(is);
      ss >> label;
      if (label == "vertex") {
        if (vi < 3) {
          ss >> facet.vertices[vi];
        }
        ++vi;
      }
    } while (label != "endloop");
    if (vi > 3)
      throw; // only support triangles
  } else {
    throw;
  }
  consume(is, "endfacet");
}

void parse_ascii_solid(std::istream &is, Mesh &mesh) {
  std::string line, kind, param;

  while (std::getline(is, line)) {
    std::istringstream ss(line);
    ss >> kind;
    if (kind == "endsolid") {
      break;
    } else if (kind == "facet") {
      Facet facet = {};
      ss >> param;
      if (param != "normal")
        throw;
      ss >> facet.normal;
      parse_ascii_facet(is, facet);
      mesh.facets.push_back(facet);
    } else {
      throw;
    }
  }
}

void parse_ascii_stream(std::istream &is, Mesh &mesh) {
  std::string line, kind;
  while (std::getline(is, line)) {
    std::istringstream ss(line);
    ss >> kind;
    if (kind == "solid") {
      ss.ignore(line.length(), ' ');
      std::getline(ss, mesh.name);
      parse_ascii_solid(is, mesh);
    } else {
      throw;
    }
  }
}

template <typename T> T little_endian_to_native(T v) {
  T vn = 0;
  for (unsigned int i = 0; i < sizeof(T); ++i) {
    vn += (T)((uint8_t *)&v)[i] << (8 * i);
  }
  return vn;
}

template <> float little_endian_to_native(float v) {
  unsigned int vntemp = little_endian_to_native(*(uint32_t *)&v);
  return *(float *)&vntemp;
}

template <typename T> T read_binary_value(std::istream &is);

template <> uint16_t read_binary_value(std::istream &is) {
  uint16_t value;
  is.read((char *)&value, 2);
  value = little_endian_to_native(value);
  return value;
}

template <> uint32_t read_binary_value(std::istream &is) {
  uint32_t value;
  is.read((char *)&value, 4);
  value = little_endian_to_native(value);
  return value;
}

template <> float read_binary_value(std::istream &is) {
  float value;
  is.read((char *)&value, 4);
  value = little_endian_to_native(value);
  return value;
}

template <> Normal read_binary_value(std::istream &is) {
  Normal n;
  n.x = read_binary_value<float>(is);
  n.y = read_binary_value<float>(is);
  n.z = read_binary_value<float>(is);
  return n;
}

template <> Vertex read_binary_value(std::istream &is) {
  Vertex v;
  v.x = read_binary_value<float>(is);
  v.y = read_binary_value<float>(is);
  v.z = read_binary_value<float>(is);
  return v;
}

const size_t STL_BINARY_HDR_SIZE = 80;

const size_t STL_BINARY_META_SIZE = sizeof(uint32_t); // number of triangles

const size_t STL_BINARY_TRIANGLE_SIZE = 3 * sizeof(float) +     // 1 normal
                                        3 * 3 * sizeof(float) + // 3 vertices
                                        sizeof(uint16_t);       // 1 attribute

void parse_binary_stream(std::istream &is, Mesh &mesh) {
  char header[STL_BINARY_HDR_SIZE + 1]; // header plus null byte
  is.read(header, STL_BINARY_HDR_SIZE);
  header[STL_BINARY_HDR_SIZE] = '\0'; // null terminate just in case
  mesh.header = header;

  auto num_triangles = read_binary_value<uint32_t>(is);

  for (uint32_t ti = 0; ti < num_triangles; ++ti) {
    Facet facet = {};
    facet.normal = read_binary_value<Normal>(is);
    for (int vi = 0; vi < 3; ++vi) {
      facet.vertices[vi] = read_binary_value<Vertex>(is);
    }
    // This field is unused, but must be present
    auto attrib_byte_count = read_binary_value<uint16_t>(is);
    mesh.facets.push_back(facet);
  }
}

void parse_stream(std::istream &is, Mesh &mesh) {
  // Read enough of file to determine its type.
  char header_start[6] = "";
  is.read(header_start, 5);
  header_start[5] = '\0';
  is.seekg(0, is.end);
  int file_size = is.tellg();

  // Rewind so parsers can start at the beginning.
  is.seekg(0, is.beg);

  // Ascii files start with "solid"
  bool is_ascii = (std::string(header_start) == "solid");

  // WAR: Some binary files appear to be ASCII, since they violate
  // the rule of not using "solid" in their metadata. Try to detect
  // this
  int geom_size = file_size - STL_BINARY_HDR_SIZE - STL_BINARY_META_SIZE;

  if (is_ascii && geom_size > 0 && geom_size % STL_BINARY_TRIANGLE_SIZE == 0) {
    // std::cout << "File looks suspiciously like a binary STL\n";
    is_ascii = false;
  }

  if (is_ascii) {
    parse_ascii_stream(is, mesh);
  } else {
    parse_binary_stream(is, mesh);
  }
}

void parse_file(const char *filename, Mesh &mesh) {
  std::ifstream ifs;
  ifs.exceptions(std::ifstream::failbit | std::ifstream::badbit);
  try {
    ifs.open(filename, std::ifstream::binary);
  } catch (...) {
    throw std::runtime_error("File not found: " + std::string(filename));
  }
  parse_stream(ifs, mesh);
  ifs.close();
}

////////////////////////////////////////////////////////////////////////
// Writing
////////////////////////////////////////////////////////////////////////

std::ostream &operator<<(std::ostream &os, const Vertex &v) {
  return os << v.x << ' ' << v.y << ' ' << v.z;
}
std::ostream &operator<<(std::ostream &os, const Normal &n) {
  return os << n.x << ' ' << n.y << ' ' << n.z;
}

template <typename T> T native_to_little_endian(T v) {
  T vn = 0;
  for (unsigned int i = 0; i < sizeof(T); ++i) {
    ((uint8_t *)&vn)[i] = (v >> (8 * i)) & 0xFF;
  }
  return vn;
}

template <> float native_to_little_endian(float v) {
  unsigned int vntemp = native_to_little_endian(*(uint32_t *)&v);
  return *(float *)&vntemp;
}

template <typename T> void write_binary_value(std::ostream &os, const T &value);

template <> void write_binary_value(std::ostream &os, const uint16_t &value) {
  uint16_t le_value = native_to_little_endian(value);
  os.write((char *)&le_value, 2);
}

template <> void write_binary_value(std::ostream &os, const uint32_t &value) {
  uint32_t le_value = native_to_little_endian(value);
  os.write((char *)&le_value, 4);
}

template <> void write_binary_value(std::ostream &os, const float &value) {
  float le_value = native_to_little_endian(value);
  os.write((char *)&le_value, 4);
}

template <> void write_binary_value(std::ostream &os, const Vertex &v) {
  write_binary_value(os, v.x);
  write_binary_value(os, v.y);
  write_binary_value(os, v.z);
}

template <> void write_binary_value(std::ostream &os, const Normal &n) {
  write_binary_value(os, n.x);
  write_binary_value(os, n.y);
  write_binary_value(os, n.z);
}

void write_stream(std::ostream &os, const Mesh &mesh, Format format) {
  switch (format) {
  case Format::ascii: {
    std::string solid_name = mesh.name;
    if (solid_name.size() == 0)
      solid_name = mesh.header;
    if (solid_name.size() == 0)
      solid_name = "stlloader";
    os << "solid " << solid_name << '\n';
    for (auto &facet : mesh.facets) {
      os << "facet "
         << "normal " << facet.normal << '\n';
      os << "outer loop\n";
      for (int vi = 0; vi < 3; ++vi) {
        os << "vertex " << facet.vertices[vi] << '\n';
      }
      os << "endloop\n";
      os << "endfacet\n";
    }
    os << "endsolid " << solid_name << '\n';
  } break;
  case Format::binary: {
    std::string padded_header = mesh.header;
    if (padded_header.size() == 0)
      padded_header = mesh.name;
    if (padded_header.size() == 0)
      padded_header = "stlloader";
    padded_header.resize(STL_BINARY_HDR_SIZE, '\0');
    os << padded_header;
    write_binary_value<uint32_t>(os, mesh.facets.size());
    for (auto &facet : mesh.facets) {
      write_binary_value(os, facet.normal);
      for (int vi = 0; vi < 3; ++vi) {
        write_binary_value(os, facet.vertices[vi]);
      }
      write_binary_value(os, uint16_t(0));
    }
  } break;
  }
}

void write_file(const char *filename, const Mesh &mesh, Format format) {
  std::ofstream ofs(filename, std::ifstream::binary);
  write_stream(ofs, mesh, format);
}

#endif // STLLOADER_IMPLEMENTATION

} // namespace stlloader
