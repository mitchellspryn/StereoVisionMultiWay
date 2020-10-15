#include "../include/CudaSimdDisparityMapGenerator.hpp"
#include "../include/CudaSimdFunctions.h"

#include <iostream>

CudaSimdDisparityMapGenerator::CudaSimdDisparityMapGenerator(
        const DisparityMapAlgorithmParameters_t& parameters)
        : parameters_(parameters) {
    this->ensureParametersValid();
    this->disparityBuf_.resize(this->parameters_.rightScanSteps + this->parameters_.leftScanSteps + 1, 0);
}

CudaSimdDisparityMapGenerator::~CudaSimdDisparityMapGenerator() {
    destroyCudaMemoryBuffersSimd();
}

void CudaSimdDisparityMapGenerator::setParameters(
        const DisparityMapAlgorithmParameters_t& parameters) {
    this->parameters_ = parameters;
    this->ensureParametersValid();
    this->disparityBuf_.resize(this->parameters_.rightScanSteps + this->parameters_.leftScanSteps + 1, 0);
}

const DisparityMapAlgorithmParameters_t& CudaSimdDisparityMapGenerator::getParameters() const {
    return this->parameters_;
}

void CudaSimdDisparityMapGenerator::computeDisparity(
        const cv::Mat& leftImage,
        const cv::Mat& rightImage,
        cv::Mat& disparity) {

    computeDisparityCudaSimd(
        leftImage.rows,
        leftImage.cols,
        this->parameters_.blockSize,
        this->parameters_.leftScanSteps,
        this->parameters_.rightScanSteps,
        leftImage.data,
        rightImage.data,
        reinterpret_cast<float*>(disparity.data));
}

void CudaSimdDisparityMapGenerator::ensureParametersValid() {
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

