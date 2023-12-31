#include "aruco_eval.hpp"
#include "aruco_utility.hpp"
#include <iostream>
#include <opencv2/aruco/charuco.hpp>
#include <opencv2/highgui.hpp>

namespace {
const char *about =
    "The toolkit to generate ArUco board and markers, calibrate camerd, "
    "detect markers and pose estimation, and evaluate the accuracy of "
    "the 3d generated mesh.";
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
    "create the charuco markers;\nc=3 to detect charuco board with "
    "camera calibration and Pose Estimation;\nc=4 to detect the charuco "
    "markers}"
    "{@file    |<none> | The file with calibrated camera parameters }"
    "{v        |       | Input from video file, if ommited, input comes from "
    "camera }"
    "{dp       |       | File of marker detector parameters }"
    "{rs       | false | Apply refind strategy }"
    "{zt       | false | Assume zero tangential distortion }"
    "{a        |       | Fix aspect ratio (fx/fy) to this value }"
    "{pc       | false | Fix the principal point at the center }"
    "{sc       | false | Show detected chessboard corners after calibration }";
} // namespace

int main(int argc, char *argv[]) {
  cv::CommandLineParser parser(argc, argv, keys);
  parser.about(about);
  if (argc < 4) {
    parser.printMessage();
    return 0;
  }

  // Video path
  cv::String video;
  if (parser.has("v")) {
    video = parser.get<cv::String>("v");
  }

  // Calibration file
  std::string filePath = parser.get<std::string>(0);
  int calibrationFlags = 0;
  float aspectRatio = 1;
  if (parser.has("a")) {
    calibrationFlags |= cv::CALIB_FIX_ASPECT_RATIO;
    aspectRatio = parser.get<float>("a");
  }
  if (parser.get<bool>("zt"))
    calibrationFlags |= cv::CALIB_ZERO_TANGENT_DIST;
  if (parser.get<bool>("pc"))
    calibrationFlags |= cv::CALIB_FIX_PRINCIPAL_POINT;

  bool showChessboardCorners = parser.get<bool>("sc");

  // Dictionary
  cv::aruco::Dictionary dictionary = cv::aruco::getPredefinedDictionary(0);
  if (parser.has("d")) {
    int dictionaryId = parser.get<int>("d");
    dictionary = cv::aruco::getPredefinedDictionary(
        cv::aruco::PredefinedDictionaryType(dictionaryId));
  }

  if (!parser.check()) {
    parser.printErrors();
    return 0;
  }

  int c = parser.get<int>("c");
  switch (c) {
  case 1:
    createBoard(filePath, dictionary);
    break;
  case 2:
    createCharucoMarkers(filePath, dictionary);
    break;
  case 3:
    calibrateCamera(filePath, video, dictionary, calibrationFlags,
                    showChessboardCorners, aspectRatio);
    break;
  case 4:
    detectCharucoBoardWithCalibrationPose(filePath, video);
    break;
  case 5:
    detectCharucoMarkers(filePath, video);
    break;
  default:
    break;
  }

  return 0;
}
