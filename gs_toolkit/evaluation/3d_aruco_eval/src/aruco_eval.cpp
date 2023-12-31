#include "aruco_eval.hpp"
#include "aruco_utility.hpp"
#include <iostream>
#include <opencv2/calib3d.hpp>
#include <opencv2/highgui.hpp>
#include <opencv2/imgproc.hpp>
#include <opencv2/objdetect.hpp>
#include <vector>

void createBoard(const std::string &outFile,
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

void createCharucoMarkers(const std::string &outFile,
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

void calibrateCamera(const std::string &caliFile, const std::string &video,
                     const cv::aruco::Dictionary &dictionary,
                     const int calibrationFlags,
                     const bool showChessboardCorners,
                     const float aspectRatio) {

  cv::aruco::DetectorParameters detectorParams =
      cv::aruco::DetectorParameters();

  cv::VideoCapture inputVideo;
  int waitTime;
  if (!video.empty()) {
    inputVideo.open(video);
    waitTime = 0;
  }

  // Create charuco board object
  cv::aruco::CharucoBoard board(cv::Size(5, 7), 0.04f, 0.02f, dictionary);
  cv::aruco::CharucoParameters charucoParams;

  cv::aruco::CharucoDetector detector(board, charucoParams, detectorParams);

  // Collect data from each frame
  std::vector<cv::Mat> allCharucoCorners;
  std::vector<cv::Mat> allCharucoIds;

  std::vector<std::vector<cv::Point2f>> allImagePoints;
  std::vector<std::vector<cv::Point3f>> allObjectPoints;

  std::vector<cv::Mat> allImages;
  cv::Size imageSize;

  while (inputVideo.grab()) {
    cv::Mat image, imageCopy;
    inputVideo.retrieve(image);

    std::vector<int> markerIds;
    std::vector<std::vector<cv::Point2f>> markerCorners, rejectedMarkers;
    cv::Mat currentCharucoCorners;
    cv::Mat currentCharucoIds;
    std::vector<cv::Point3f> currentObjectPoints;
    std::vector<cv::Point2f> currentImagePoints;

    // Detect ChArUco board
    detector.detectBoard(image, currentCharucoCorners, currentCharucoIds);

    // Draw results
    image.copyTo(imageCopy);
    if (!markerIds.empty()) {
      cv::aruco::drawDetectedMarkers(imageCopy, markerCorners);
    }

    // visualization
    if (currentCharucoCorners.total() > 3) {
      cv::aruco::drawDetectedCornersCharuco(imageCopy, currentCharucoCorners,
                                            currentCharucoIds);
    }

    putText(imageCopy,
            "Press 'c' to add current frame. 'ESC' to finish and calibrate",
            cv::Point(10, 20), cv::FONT_HERSHEY_SIMPLEX, 0.5,
            cv::Scalar(255, 0, 0), 2);

    imshow("out", imageCopy);

    // Wait for key pressed
    char key = (char)cv::waitKey(waitTime);

    if (key == 27) {
      std::cout << "ESC key pressed, calculating the camera intrinsics"
                << std::endl;
      break;
    }

    if (key == 'c' && currentCharucoCorners.total() > 3) {
      // Match image points
      board.matchImagePoints(currentCharucoCorners, currentCharucoIds,
                             currentObjectPoints, currentImagePoints);

      if (currentImagePoints.empty() || currentObjectPoints.empty()) {
        std::cout << "Point matching failed, try again." << std::endl;
        continue;
      }

      std::cout << "Frame captured" << std::endl;

      allCharucoCorners.push_back(currentCharucoCorners);
      allCharucoIds.push_back(currentCharucoIds);
      allImagePoints.push_back(currentImagePoints);
      allObjectPoints.push_back(currentObjectPoints);
      allImages.push_back(image);

      imageSize = image.size();
    }
  }

  if (allCharucoCorners.size() < 4) {
    std::cerr << "Not enough corners for calibration" << std::endl;
  }

  cv::Mat cameraMatrix, distCoeffs;

  // Calibrate camera using ChArUco
  double repError =
      calibrateCamera(allObjectPoints, allImagePoints, imageSize, cameraMatrix,
                      distCoeffs, cv::noArray(), cv::noArray(), cv::noArray(),
                      cv::noArray(), cv::noArray(), calibrationFlags);

  bool saveOk =
      saveCameraParams(caliFile, imageSize, aspectRatio, calibrationFlags,
                       cameraMatrix, distCoeffs, repError);

  if (!saveOk) {
    std::cerr << "Cannot save output file" << std::endl;
  }

  std::cout << "Rep Error: " << repError << std::endl;
  std::cout << "Calibration saved to " << caliFile << std::endl;

  // Show interpolated charuco corners for debugging
  if (showChessboardCorners) {
    for (size_t frame = 0; frame < allImages.size(); frame++) {
      cv::Mat imageCopy = allImages[frame].clone();

      if (allCharucoCorners[frame].total() > 0) {
        cv::aruco::drawDetectedCornersCharuco(
            imageCopy, allCharucoCorners[frame], allCharucoIds[frame]);
      }

      imshow("out", imageCopy);
      char key = (char)cv::waitKey(0);
      if (key == 27) {
        break;
      }
    }
  }
}

void detectCharucoBoardWithCalibrationPose(cv::String caliFile,
                                           cv::String video) {
  cv::VideoCapture inputVideo;
  inputVideo.open(video);
  cv::Mat cameraMatrix, distCoeffs;
  bool readOk = readCameraParameters(caliFile, cameraMatrix, distCoeffs);
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

void detectCharucoMarkers(cv::String caliFile, cv::String video) {
  cv::VideoCapture inputVideo;
  inputVideo.open(video);
  cv::Mat cameraMatrix, distCoeffs;
  bool readOk = readCameraParameters(caliFile, cameraMatrix, distCoeffs);
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
