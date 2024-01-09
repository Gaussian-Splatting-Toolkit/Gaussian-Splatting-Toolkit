#define STLLOADER_IMPLEMENTATION
#include "happly.h"
#include "stlloader.h"
#include "triangle_mesh_distance.h"
#include <iostream>
#include <set>
#include <vector>

int main(int argc, char **argv) {
  if (argc < 3) {
    std::cerr << "Usage: evaluate <STL filename> <generated filename>"
              << std::endl;
    return 1;
  }

  std::string stl_filename = argv[1];
  std::string generated_filename = argv[2];

  std::cout << "STL filename: " << stl_filename << std::endl;
  std::cout << "Generated filename: " << generated_filename << std::endl;

  // Read the ground truth
  stlloader::Mesh mesh;
  stlloader::parse_file(stl_filename.c_str(), mesh);
  // stlloader::print(mesh);

  // Read the generated mesh
  happly::PLYData ply_in(generated_filename);

  std::vector<float> x = ply_in.getElement("vertex").getProperty<float>("x");
  std::vector<float> y = ply_in.getElement("vertex").getProperty<float>("y");
  std::vector<float> z = ply_in.getElement("vertex").getProperty<float>("z");

  // Calculate the distance between the two meshes
  std::set<std::array<double, 3>> vertices_set;
  for (auto ms : mesh.facets) {
    for (auto v : ms.vertices) {
      vertices_set.insert(std::array<double, 3>{v.x, v.y, v.z});
    }
  }

  std::vector<std::array<double, 3>> vertices;
  std::vector<std::array<int, 3>> triangles;
  for (auto v : vertices_set) {
    vertices.push_back(v);
  }
  for (auto ms : mesh.facets) {
    std::array<int, 3> triangle;
    for (int i = 0; i < 3; i++) {
      triangle[i] =
          std::distance(vertices_set.begin(),
                        vertices_set.find({ms.vertices[i].x, ms.vertices[i].y,
                                           ms.vertices[i].z}));
    }
    triangles.push_back(triangle);
  }

  // Initialize TriangleMeshDistance
  tmd::TriangleMeshDistance mesh_distance(vertices, triangles);

  // Query TriangleMeshDistance
  double total_distance = 0;

  for (int i = 0; i < x.size(); i++) {
    tmd::Result result = mesh_distance.signed_distance({x[i], y[i], z[i]});
    total_distance += std::abs(result.distance);
  }

  double avg_distance = total_distance / x.size();
  std::cout << "Average Error: " << avg_distance << std::endl;

  return 0;
}
