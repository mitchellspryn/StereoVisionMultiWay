#include <algorithm>
#include <iostream>
#include <limits>
#include <memory>
#include <stdexcept>

#include <opencv2/core.hpp>
#include <opencv2/core/utility.hpp>
#include <opencv2/imgcodecs.hpp>

#include "../include/DisparityMapAlgorithmParameters.hpp"
#include "../include/DisparityMapGenerator.hpp"
#include "../include/DisparityMapGeneratorFactory.hpp"

int main(int argc, char** argv) {

    const cv::String commandLineKeys = 
        "{help h usage ? |                          | This program takes in two images and generates a disparity map.}"
        "{leftImage       |                  <none> | The left image to process.}"
        "{rightImage      |                  <none> | The right image to proces.}"
        "{algorithmName   |                  <none> | The algorithm name to use.}"
        "{outputPath      |           disparity.png | The output path to which to write the image.}"
        "{blockSize       |                       7 | The maximum block size to use for matching.}"
        "{leftScanSteps   |                      50 | The number of blocks to scan to the left.}"
        "{rightScanSteps  |                      50 | The number of blocks to scan to the right.}";

    cv::CommandLineParser parser(argc, argv, commandLineKeys);

    if (!parser.check()) {
        parser.printMessage();
        parser.printErrors();
        return 1;
    }

    if (parser.has("help")
        || (!parser.has("leftImage"))
        || (!parser.has("rightImage"))
        || (!parser.has("algorithmName"))) {
        parser.printMessage();
        return 1;
    }

    DisparityMapAlgorithmParameters_t parameters;
    parameters.blockSize = parser.get<int>("blockSize");
    parameters.leftScanSteps = parser.get<int>("leftScanSteps");
    parameters.rightScanSteps = parser.get<int>("rightScanSteps");
    parameters.leftImageFilePath = std::string(parser.get<cv::String>("leftImage"));
    parameters.rightImageFilePath = std::string(parser.get<cv::String>("rightImage"));
    parameters.outputPath = std::string(parser.get<cv::String>("outputPath"));
    parameters.algorithmName = std::string(parser.get<cv::String>("algorithmName"));

    std::cout << "Creating disparity generator..." << std::endl;

    DisparityMapGeneratorFactory factory;
    std::unique_ptr<DisparityMapGenerator> generator = factory.create(parameters);

    std::cout << "Initializing disparity generator..." << std::endl;
    generator->setParameters(parameters);

    std::cout 
        << "Reading in left image from '" 
        << parameters.leftImageFilePath 
        << "'..." 
        << std::endl;

    cv::Mat leftImage = cv::imread(parameters.leftImageFilePath, cv::IMREAD_GRAYSCALE);
    
    std::cout 
        << "Reading in right image from '" 
        << parameters.rightImageFilePath 
        << "'..." 
        << std::endl;

    cv::Mat rightImage = cv::imread(parameters.rightImageFilePath, cv::IMREAD_GRAYSCALE);

    if ((leftImage.rows == 0)
            ||
        (leftImage.cols == 0)) {
        throw std::runtime_error("Error. Left image is empty.");
    }

    if ((rightImage.rows == 0) 
            ||
        (rightImage.cols == 0)) {
        throw std::runtime_error("Error. Right image is empty.");
    }

    if ((leftImage.rows != rightImage.rows)
            ||
        (leftImage.cols != rightImage.cols)) {
        throw std::runtime_error(std::string("Error. Dimensions of input images are not the same.\n")
            + std::string("Left image: (") + std::to_string(leftImage.rows) + std::string("x") + std::to_string(leftImage.cols) + std::string(")\n")
            + std::string("Right iamge: (") + std::to_string(rightImage.rows) + std::string("x") + std::to_string(rightImage.cols) + std::string(")"));
    }

    std::cout << "Generating disparity image with the following parameters:" << std::endl;
    std::cout << "\tAlgorithm Name: " << parameters.algorithmName << "." << std::endl;
    std::cout << "\tBlock Size: " << parameters.blockSize << "." << std::endl;
    std::cout << "\tLeft Scan Steps: " << parameters.leftScanSteps << "." << std::endl;
    std::cout << "\tRight Scan Steps: " << parameters.rightScanSteps << "." << std::endl;
    std::cout << "\tDisparity Metric: " << "SUM_ABSOLUTE_DIFFERENCE" << "." << std::endl;
    std::cout << "\tLeft Image: " << parameters.leftImageFilePath << "." << std::endl;
    std::cout << "\tRight Image: " << parameters.rightImageFilePath << "." << std::endl;
    std::cout << "\tImage Size: (" << leftImage.rows << "x" << leftImage.cols << ")." << std::endl;
    std::cout << "\tOutput Path: " << parameters.outputPath << std::endl;

    cv::Mat disparityImage(leftImage.rows, leftImage.cols, CV_32FC1);

    generator->computeDisparity(leftImage, rightImage, disparityImage);

    std::cout << "Computation complete. Generating output image..." << std::endl;
    
    float maxDisparity = std::numeric_limits<float>::min();
    float minDisparity = std::numeric_limits<float>::max();

    for (int y = 0; y < disparityImage.rows; y++) {
        for (int x = 0; x < disparityImage.cols; x++) {
            float value = disparityImage.at<float>(y, x);
            maxDisparity = std::max(value, maxDisparity);
            minDisparity = std::min(value, minDisparity);
        }
    }

    float range = maxDisparity - minDisparity;

    // This should not happen for any sane input.
    if (range == 0) {
        throw std::runtime_error("Hmm...the max and min disparities are the same.");
    }

    cv::Mat outputImage(leftImage.rows, leftImage.cols, CV_8UC3);
    for (int y = 0; y < disparityImage.rows; y++) {
        for (int x = 0; x < disparityImage.cols; x++) {
            float value = disparityImage.at<float>(y, x);
            uint8_t rgbValue = static_cast<uint8_t>(255.0f * (value - minDisparity) / range);
            cv::Vec3b color;
            color[0] = rgbValue;
            color[1] = rgbValue;
            color[2] = rgbValue;
            outputImage.at<cv::Vec3b>(y, x) = color;
        }
    }

    std::cout << "Writing to " << parameters.outputPath << "..." << std::endl;

    cv::imwrite(parameters.outputPath, outputImage);

    std::cout << "Graceful termination" << std::endl;

    return 0;
}
