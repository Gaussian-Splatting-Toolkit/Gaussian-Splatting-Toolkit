#include "aruco_samples_utility.hpp"
#include <iostream>
#include <opencv2/aruco/charuco.hpp>
#include <opencv2/highgui.hpp>

namespace {
const char *about = "Charuco board creation and detection of charuco board "
                    "with camera caliberation";
const char *keys =
    "{c        |       | \nPut value of c=1 to detect charuco board with "
    "camera calibration and Pose Estimation;\nc=2 to detect the charuco "
    "markers}";
} // namespace

static inline void detectCharucoBoardWithCalibrationPose();
static inline void detectCharucoMarkers();

int main(int argc, char *argv[]) {
  cv::CommandLineParser parser(argc, argv, keys);
  parser.about(about);
  if (argc < 2) {
    parser.printMessage();
    return 0;
  }

  int c = parser.get<int>("c");
  switch (c) {
  case 1:
    detectCharucoBoardWithCalibrationPose();
    break;
  case 3:
    detectCharucoMarkers();
    break;
  default:
    break;
  }

  return 0;
}

static inline void detectCharucoBoardWithCalibrationPose() {
  cv::VideoCapture inputVideo;
  inputVideo.open(0);
  cv::Mat cameraMatrix, distCoeffs;
  std::string filename = "calib.txt";
  bool readOk = readCameraParameters(filename, cameraMatrix, distCoeffs);
  if (!readOk) {
    std::cerr << "Invalid camera file" << std::endl;
  } else {
    // Detect charuco board
    cv::aruco::Dictionary dictionary =
        cv::aruco::getPredefinedDictionary(cv::aruco::DICT_6X6_250);
    cv::Ptr<cv::aruco::CharucoBoard> board =
        new cv::aruco::CharucoBoard(cv::Size(5, 7), 0.04f, 0.02f, dictionary);
    cv::Ptr<cv::aruco::DetectorParameters> params =
        cv::makePtr<cv::aruco::DetectorParameters>();

    while (inputVideo.grab()) {
      cv::Mat image, imageCopy;
      inputVideo.retrieve(image);
      image.copyTo(imageCopy);
      std::vector<int> markerIds;
      std::vector<std::vector<cv::Point2f>> markerCorners;
      // Detect markers
      cv::aruco::detectMarkers(
          image, cv::makePtr<cv::aruco::Dictionary>(board->getDictionary()),
          markerCorners, markerIds, params);
      // If at least one marker detected
      if (markerIds.size() > 0) {
        // Draw detected markers
        cv::aruco::drawDetectedMarkers(imageCopy, markerCorners, markerIds);
        std::vector<cv::Point2f> charucoCorners;
        std::vector<int> charucoIds;
        cv::aruco::interpolateCornersCharuco(markerCorners, markerIds, image,
                                             board, charucoCorners, charucoIds,
                                             cameraMatrix, distCoeffs);
        // If at least one charuco corner detected
        if (charucoIds.size() > 0) {
          cv::Scalar color = cv::Scalar(255, 0, 0);
          cv::aruco::drawDetectedCornersCharuco(imageCopy, charucoCorners,
                                                charucoIds, color);
          cv::Vec3d rvec, tvec;
          bool valid = cv::aruco::estimatePoseCharucoBoard(
              charucoCorners, charucoIds, board, cameraMatrix, distCoeffs, rvec,
              tvec);
          if (valid)
            cv::drawFrameAxes(imageCopy, cameraMatrix, distCoeffs, rvec, tvec,
                              0.1f);
        }
      }
      cv::imshow("out", imageCopy);
      char key = (char)cv::waitKey(30);
      if (key == 27)
        break;
    }
  }
}

static inline void detectCharucoMarkers() {
  cv::VideoCapture inputVideo;
  inputVideo.open(0);
  cv::Mat cameraMatrix, distCoeffs;
  std::string filename = "calib.txt";
  bool readOk = readCameraParameters(filename, cameraMatrix, distCoeffs);
  if (!readOk) {
    std::cerr << "Invalid camera file" << std::endl;
  } else {
    // Detect charuco board
    cv::aruco::Dictionary dictionary =
        cv::aruco::getPredefinedDictionary(cv::aruco::DICT_6X6_250);
    cv::Ptr<cv::aruco::CharucoBoard> board =
        new cv::aruco::CharucoBoard(cv::Size(5, 7), 0.04f, 0.02f, dictionary);
    cv::Ptr<cv::aruco::DetectorParameters> params =
        cv::makePtr<cv::aruco::DetectorParameters>();

    while (inputVideo.grab()) {
      cv::Mat image, imageCopy;
      inputVideo.retrieve(image);
      image.copyTo(imageCopy);
      std::vector<int> markerIds;
      std::vector<std::vector<cv::Point2f>> markerCorners;
      // Detect markers
      cv::aruco::detectMarkers(
          image, cv::makePtr<cv::aruco::Dictionary>(board->getDictionary()),
          markerCorners, markerIds, params);
      // If at least one marker detected
      if (markerIds.size() > 0) {
        // Draw detected markers
        cv::aruco::drawDetectedMarkers(imageCopy, markerCorners, markerIds);
        std::vector<cv::Point2f> charucoCorners;
        std::vector<int> charucoIds;
        cv::aruco::interpolateCornersCharuco(markerCorners, markerIds, image,
                                             board, charucoCorners, charucoIds,
                                             cameraMatrix, distCoeffs);
        // If at least one charuco corner detected
        if (charucoIds.size() > 0) {
          cv::Scalar color = cv::Scalar(255, 0, 0);
          cv::aruco::drawDetectedCornersCharuco(imageCopy, charucoCorners,
                                                charucoIds, color);
        }
      }
      cv::imshow("out", imageCopy);
      char key = (char)cv::waitKey(30);
      if (key == 27)
        break;
    }
  }
}
