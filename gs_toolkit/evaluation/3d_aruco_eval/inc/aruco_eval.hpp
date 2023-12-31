#ifndef ARUCO_EVAL_H
#define ARUCO_EVAL_H

#include <opencv2/aruco/charuco.hpp>
#include <string>

void createBoard(const std::string &outFile,
                 const cv::aruco::Dictionary &dictionary);
void createCharucoMarkers(const std::string &outFile,
                          const cv::aruco::Dictionary &dictionary);
void detectCharucoBoardWithCalibrationPose(cv::String inputPath,
                                           cv::String video);
void detectCharucoMarkers(cv::String inputPath, cv::String video);
void calibrateCamera(const std::string &caliFile, const std::string &video,
                     const cv::aruco::Dictionary &dictionary,
                     const int calibrationFlags,
                     const bool showChessboardCorners, const float aspectRatio);

#endif // ARUCO_EVAL_H
