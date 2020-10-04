#include "../include/OpenMpThreadedDisparityMapGenerator.hpp"

#include <iostream>

OpenMpThreadedDisparityMapGenerator::OpenMpThreadedDisparityMapGenerator(
        const DisparityMapAlgorithmParameters_t& parameters)
        : parameters_(parameters) {
    this->ensureParametersValid();
}

void OpenMpThreadedDisparityMapGenerator::setParameters(
        const DisparityMapAlgorithmParameters_t& parameters) {
    this->parameters_ = parameters;
    this->ensureParametersValid();
}

const DisparityMapAlgorithmParameters_t& OpenMpThreadedDisparityMapGenerator::getParameters() const {
    return this->parameters_;
}

void OpenMpThreadedDisparityMapGenerator::computeDisparity(
        const cv::Mat& leftImage,
        const cv::Mat& rightImage,
        cv::Mat& disparity) {
    #pragma omp parallel for collapse(2) default(none) shared(leftImage, rightImage, disparity)
    for (int y = 0; y < disparity.rows; y++) {
        for (int x = 0; x < disparity.cols; x++) {
            disparity.at<float>(y, x) = computeDisparityForPixel(
                y,
                x,
                leftImage,
                rightImage);
        }
    }
}

void OpenMpThreadedDisparityMapGenerator::ensureParametersValid() {
    if (this->parameters_.blockSize < 0) {
        throw std::runtime_error("Error: block size is less than zero.");
    }

    if (this->parameters_.blockSize % 2 == 0) {
        throw std::runtime_error("Error: block size is not odd.");
    }

    if (this->parameters_.leftScanSteps < 0) {
        throw std::runtime_error("Error: left scan steps is negative.");
    }

    if (this->parameters_.rightScanSteps < 0) {
        throw std::runtime_error("Error: right scan steps is negative.");
    }
}

float OpenMpThreadedDisparityMapGenerator::computeDisparityForPixel(
        int y, 
        int x,
        const cv::Mat& leftImage,
        const cv::Mat& rightImage) {

    float localDisparityBuf[512];
    int maxBlockStep = (this->parameters_.blockSize - 1) / 2;

    int templateLeftHalfWidth = std::min(x, maxBlockStep);
    int templateRightHalfWidth = std::min(leftImage.cols - x - 1, maxBlockStep);
    int templateTopHalfHeight = std::min(y, maxBlockStep);
    int templateBottomHalfHeight = std::min(leftImage.rows - y - 1, maxBlockStep);

    int templateWidth = templateLeftHalfWidth + templateRightHalfWidth + 1;
    int templateHeight = templateTopHalfHeight + templateBottomHalfHeight + 1;

    int leftMinY = y - templateTopHalfHeight;
    int leftMinX = x - templateLeftHalfWidth;

    int rightMinStartX = std::max(0, x - this->parameters_.leftScanSteps - templateLeftHalfWidth);
    int rightMaxStartX = std::min(leftImage.cols - templateWidth /*- 1*/, x + this->parameters_.rightScanSteps - templateLeftHalfWidth);

    int numSteps = rightMaxStartX - rightMinStartX;

    int bestIndex = 0;
    int bestSadValue = std::numeric_limits<int>::max();
    int zeroDisparityIndex = x - rightMinStartX - templateLeftHalfWidth;

    // Enabling parallelization here is faster than no parallelization at all,
    // but is slower than parallelizing on the center pixel level
    // #pragma omp parallel for
    for (int xx = rightMinStartX; xx <= rightMaxStartX; xx++) {
        int sad = computeSadOverBlock(
            leftMinY,
            leftMinX,
            leftMinY, // Ys are aligned for the two images
            xx,
            templateWidth,
            templateHeight,
            leftImage, 
            rightImage);

        localDisparityBuf[xx - rightMinStartX] = sad;

        if (sad < bestSadValue) {
            bestSadValue = sad;
            bestIndex = xx - rightMinStartX;
        }
    }

    float disparity = static_cast<float>(std::abs(bestIndex - zeroDisparityIndex));
    if ((bestIndex == 0)
        ||
        (bestIndex == numSteps)
        ||
        (bestSadValue == 0)) {
        return disparity;
    }

    float c3 = localDisparityBuf[bestIndex+1];
    float c2 = localDisparityBuf[bestIndex];
    float c1 = localDisparityBuf[bestIndex-1];

    return disparity - (0.5 * ((c3 - c1) / (c1 - (2*c2) + c3)));
}

int OpenMpThreadedDisparityMapGenerator::computeSadOverBlock(
        int minYL,
        int minXL,
        int minYR,
        int minXR,
        int width,
        int height,
        const cv::Mat& leftImage,
        const cv::Mat& rightImage) {

    int sum = 0;

    // Parallelizing here is super slow for two reasons:
    // 1) It's doing a reduction, not a straight parallel workload
    // 2) There are a relatively small number of tasks to perform
    // 3) Each iteration of the loop is quick. More time is spent waiting for threads to start.
    // #pragma omp parallel for collapse(2) reduction(+:sum)
    for (int y = 0; y < height; y++) {
        for (int x = 0; x < width; x++) {
            sum += std::abs(
                leftImage.at<unsigned char>(y + minYL, x + minXL)
                - rightImage.at<unsigned char>(y + minYR, x + minXR));
        }
    }

    return sum;
}
