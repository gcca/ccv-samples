#include <filesystem>
#include <iostream>
#include <sstream>

#include <boost/program_options.hpp>
#include <opencv4/opencv2/opencv.hpp>

namespace po = boost::program_options;

int main(int argc, char* argv[]) {
  po::options_description dsc("stereo_calibrate <imgdir> <output>");
  dsc.add_options()
    ("imgdir", po::value<std::string>()->required(), "Directory with images")
    ("output", po::value<std::string>()->required(), "Output file name");

  po::positional_options_description pos_dsc;
  pos_dsc.add("imgdir", 1);
  pos_dsc.add("output", 1);

  po::variables_map vm;

  try {
    po::store(po::command_line_parser(argc, argv)
                  .options(dsc)
                  .positional(pos_dsc)
                  .run(),
              vm);
    boost::program_options::notify(vm);

  } catch (boost::program_options::error& e) {
    std::cerr << e.what() << std::endl;
    std::cerr << dsc << std::endl;
    return 1;
  }

  std ::string imgdir(vm["imgdir"].as<std::string>());
  std ::string output(vm["output"].as<std::string>());

  const std::filesystem::path imagedirPath{imgdir};

  if (!std::filesystem::exists(imagedirPath)) {
    std::ostringstream oss;
    oss << "Invalid image dir: " << imagedirPath;
    throw std::runtime_error{oss.str()};
  }

  const cv::Size checkerBoardSize{7, 5};
  const double squareSize = .02875;

  std::vector<std::vector<cv::Point3f>> imagesCorners;
  std::vector<std::vector<cv::Point2f>> leftImagesCorners, rightImagesCorners,
      leftAccImagesCorners, rightAccImagesCorners;
  std::vector<cv::Size> imagesSizes;
  const std::size_t imagedirLength =
      std::distance(std::filesystem::directory_iterator{imagedirPath},
                    std::filesystem::directory_iterator{});
  leftImagesCorners.reserve(imagedirLength);
  rightImagesCorners.reserve(imagedirLength);
  imagesSizes.reserve(imagedirLength);

  std::cout << "Loading " << imagedirLength << " files from " << imagedirPath
            << std::endl;
  for (const std::filesystem::directory_entry& entry :
       std::filesystem::directory_iterator{imagedirPath}) {
    std::cout << entry << "\t...";

    cv::Mat image = cv::imread(entry.path().string(), cv::IMREAD_COLOR);
    cv::Rect leftBounds(0, 0, image.cols / 2, image.rows);
    cv::Rect rightBounds(image.cols / 2, 0, image.cols / 2, image.rows);
    cv::Mat leftImage = image(leftBounds);
    cv::Mat rightImage = image(rightBounds);
    imagesSizes.push_back(leftImage.size());

    std::vector<cv::Point2f> leftCorners, rightCorners;
    bool foundLeft = cv::findChessboardCorners(
             leftImage, checkerBoardSize, leftCorners,
             cv::CALIB_CB_ADAPTIVE_THRESH | cv::CALIB_CB_FILTER_QUADS),
         foundRight = cv::findChessboardCorners(
             rightImage, checkerBoardSize, rightCorners,
             cv::CALIB_CB_ADAPTIVE_THRESH | cv::CALIB_CB_FILTER_QUADS);

    if (foundLeft && foundRight) {
      std::cout << "found both" << std::endl;
    } else {
      std::cout << "Unmanaged left=" << foundLeft << ", right=" << foundRight
                << std::endl;
      continue;
    }

    cv::Mat grayLeftImage, grayRightImage;
    cv::cvtColor(leftImage, grayLeftImage, cv::COLOR_BGR2GRAY);
    cv::cvtColor(rightImage, grayRightImage, cv::COLOR_BGR2GRAY);

    cv::cornerSubPix(
        grayLeftImage, leftCorners, checkerBoardSize, cv::Size{-1, -1},
        cv::TermCriteria(cv::TermCriteria::MAX_ITER + cv::TermCriteria::EPS, 60,
                         1e-6));
    cv::cornerSubPix(
        grayRightImage, rightCorners, checkerBoardSize, cv::Size{-1, -1},
        cv::TermCriteria(cv::TermCriteria::MAX_ITER + cv::TermCriteria::EPS, 60,
                         1e-6));

    std::vector<cv::Point3f> corners;
    for (std::size_t i = 0; i < checkerBoardSize.height; i++) {
      for (std::size_t j = 0; j < checkerBoardSize.width; j++) {
        corners.emplace_back(cv::Point3f(static_cast<float>(j) * squareSize,
                                         static_cast<float>(i) * squareSize,
                                         0));
      }
    }
    imagesCorners.emplace_back(std::move(corners));
    leftAccImagesCorners.emplace_back(std::move(leftCorners));
    rightAccImagesCorners.emplace_back(std::move(rightCorners));
  }
  // TODO: compute <lr>K, <lr>D with cv::undistort, <lr>corners

  for (std::size_t i = 0; i < leftAccImagesCorners.size(); i++) {
    std::vector<cv::Point2f> left, right;
    for (std::size_t j = 0; j < leftAccImagesCorners[i].size(); j++) {
      left.push_back(
          cv::Point2f(static_cast<double>(leftAccImagesCorners[i][j].x),
                      static_cast<double>(leftAccImagesCorners[i][j].y)));
      right.push_back(
          cv::Point2f(static_cast<double>(rightAccImagesCorners[i][j].x),
                      static_cast<double>(rightAccImagesCorners[i][j].y)));
    }
    leftImagesCorners.push_back(left);
    rightImagesCorners.push_back(right);
  }

  cv::Mat leftK, leftD, rightK, rightD, R, E, F;
  cv::Vec3d T;

  cv::stereoCalibrate(imagesCorners, leftImagesCorners, rightImagesCorners,
                      leftK, leftD, rightK, rightD, imagesSizes.back(), R, T, E,
                      F);

  cv::FileStorage fs(output, cv::FileStorage::WRITE);
  fs << "LEFT_K" << leftK << "LEFT_D" << leftD << "RIGTH_K" << rightK
     << "RIGTH_D" << rightD << "R" << R << "T" << T << "E" << E << "F" << F;

  return 0;
}
