#include "../include/SingleThreadedSimdDisparityMapGenerator.hpp"

#include <iostream>

SingleThreadedSimdDisparityMapGenerator::SingleThreadedSimdDisparityMapGenerator(
        const DisparityMapAlgorithmParameters_t& parameters)
        : parameters_(parameters) {
    this->ensureParametersValid();
    this->disparityBuf_.resize(this->parameters_.rightScanSteps + this->parameters_.leftScanSteps + 1, 0);
}

void SingleThreadedSimdDisparityMapGenerator::setParameters(
        const DisparityMapAlgorithmParameters_t& parameters) {
    this->parameters_ = parameters;
    this->ensureParametersValid();
    this->disparityBuf_.resize(this->parameters_.rightScanSteps + this->parameters_.leftScanSteps + 1, 0);
}

const DisparityMapAlgorithmParameters_t& SingleThreadedSimdDisparityMapGenerator::getParameters() const {
    return this->parameters_;
}

void SingleThreadedSimdDisparityMapGenerator::computeDisparity(
        const cv::Mat& leftImage,
        const cv::Mat& rightImage,
        cv::Mat& disparity) {
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

void SingleThreadedSimdDisparityMapGenerator::ensureParametersValid() {
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

float SingleThreadedSimdDisparityMapGenerator::computeDisparityForPixel(
        int y, 
        int x,
        const cv::Mat& leftImage,
        const cv::Mat& rightImage) {

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

    for (int xx = rightMinStartX; xx <= rightMaxStartX; xx++) {
        int sad = computeSadOverBlockSimd(
            leftMinY,
            leftMinX,
            leftMinY, // Ys are aligned for the two images
            xx,
            templateWidth,
            templateHeight,
            leftImage, 
            rightImage);

        this->disparityBuf_[xx - rightMinStartX] = sad;

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

    float c3 = disparityBuf_[bestIndex+1];
    float c2 = disparityBuf_[bestIndex];
    float c1 = disparityBuf_[bestIndex-1];

    return disparity - (0.5 * ((c3 - c1) / (c1 - (2*c2) + c3)));
}

int SingleThreadedSimdDisparityMapGenerator::computeSadOverBlockSimd(
        int minYL,
        int minXL,
        int minYR,
        int minXR,
        int width,
        int height,
        const cv::Mat& leftImage,
        const cv::Mat& rightImage) {

    asm("# Start SIMD loop");
    union { __m256i accumulator; uint64_t accumulatorValues[4]; };
    union { __m256i workRegA; uint8_t workRegABytes[32]; };
    union { __m256i workRegB; uint8_t workRegBBytes[32]; };
    union { __m256i maskReg; uint8_t maskRegBytes[32]; };
    union { __m256i sadReg; uint8_t sadRegBytes[32]; };

    __m256i zeros;
    accumulator = _mm256_setzero_si256();
    maskReg = _mm256_setzero_si256();
    zeros = _mm256_setzero_si256();

    uint8_t* leftImageData = leftImage.data;
    uint8_t* rightImageData = rightImage.data;

    for (int i = 0; i < width; i++) {
        maskRegBytes[i] = 0xFF;
    }

    for (int y = 0; y < height; y++) {
        int leftLoadIdx = (leftImage.cols * (y+minYL)) + minXL;
        int rightLoadIdx = (rightImage.cols * (y+minYR)) + minXR;
        workRegA = _mm256_loadu_si256(
            reinterpret_cast<__m256i const*>(leftImageData + leftLoadIdx));
        workRegB = _mm256_loadu_si256(
            reinterpret_cast<__m256i const*>(rightImageData + rightLoadIdx));
        sadReg = _mm256_sad_epu8(
                _mm256_blendv_epi8(zeros, workRegA, maskReg),
                _mm256_blendv_epi8(zeros, workRegB, maskReg)
                );
        accumulator = _mm256_add_epi64(sadReg, accumulator);
    }
    asm("# End manual unroll");
    
    int result = 0;
    for (int i = 0; i < 4; i++) {
        result += static_cast<int>(accumulatorValues[i]);
    }

    asm("# End SIMD loop");

    return result;
}
