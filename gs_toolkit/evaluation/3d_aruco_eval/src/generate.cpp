#include "aruco_samples_utility.hpp"
#include <iostream>
#include <opencv2/aruco/charuco.hpp>
#include <opencv2/highgui.hpp>

namespace {
const char *about = "Charuco board creation and detection of charuco board "
                    "with camera caliberation";
const char *keys =
    "{d        |       | dictionary: DICT_4X4_50=0, DICT_4X4_100=1, "
    "DICT_4X4_250=2,"
    "DICT_4X4_1000=3, DICT_5X5_50=4, DICT_5X5_100=5, DICT_5X5_250=6, "
    "DICT_5X5_1000=7, "
    "DICT_6X6_50=8, DICT_6X6_100=9, DICT_6X6_250=10, DICT_6X6_1000=11, "
    "DICT_7X7_50=12,"
    "DICT_7X7_100=13, DICT_7X7_250=14, DICT_7X7_1000=15, DICT_ARUCO_ORIGINAL = "
    "16}"
    "{c        |       | \nPut value of c=1 to create charuco board;\nc=2 to "
    "create the charuco markers;}";
} // namespace

static inline void createBoard(const std::string &outFile,
                               const cv::aruco::Dictionary &dictionary);
static inline void
createCharucoMarkers(const std::string &outFile,
                     const cv::aruco::Dictionary &dictionary);

int main(int argc, char *argv[]) {
  cv::CommandLineParser parser(argc, argv, keys);
  parser.about(about);
  if (argc < 2) {
    parser.printMessage();
    return 0;
  }

  cv::aruco::Dictionary dictionary = cv::aruco::getPredefinedDictionary(0);
  if (parser.has("d")) {
    int dictionaryId = parser.get<int>("d");
    dictionary = cv::aruco::getPredefinedDictionary(
        cv::aruco::PredefinedDictionaryType(dictionaryId));
  }

  std::string outFile = "/data/robot-studio/evaluation/output";
  int c = parser.get<int>("c");
  switch (c) {
  case 1:
    createBoard(outFile, dictionary);
    break;
  case 2:
    createCharucoMarkers(outFile, dictionary);
    break;
  default:
    break;
  }

  return 0;
}

static inline void createBoard(const std::string &outFile,
                               const cv::aruco::Dictionary &dictionary) {
  // Check the outFile
  if (outFile.empty()) {
    std::cerr << "Invalid output file path" << std::endl;
    return;
  }
  cv::aruco::CharucoBoard board(cv::Size(5, 7), 0.04f, 0.02f, dictionary);
  cv::Mat boardImage;
  board.generateImage(cv::Size(600, 500), boardImage, 10, 1);
  // Create board
  std::string save_path = outFile + "/charuco_board.png";
  std::cout << "Charuco board image saved to " << save_path << std::endl;
  cv::imwrite(save_path, boardImage);
}

static inline void
createCharucoMarkers(const std::string &outFile,
                     const cv::aruco::Dictionary &dictionary) {
  // Check the outFile
  if (outFile.empty()) {
    std::cerr << "Invalid output file path" << std::endl;
    return;
  }
  // Create the marker image
  cv::Mat markerImage;
  for (int i = 0; i < 50; i++) {
    cv::aruco::generateImageMarker(dictionary, i, 200, markerImage, 1);
    std::string save_path = outFile + "/marker_" + std::to_string(i) + ".png";
    cv::imwrite(save_path, markerImage);
  }
  std::cout << "Marker images saved to " << outFile << std::endl;
}
