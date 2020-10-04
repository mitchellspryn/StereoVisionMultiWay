#include "../include/CudaDisparityMapGenerator.hpp"
#include "../include/CudaFunctions.h"

#include <iostream>

CudaDisparityMapGenerator::CudaDisparityMapGenerator(
        const DisparityMapAlgorithmParameters_t& parameters)
        : parameters_(parameters) {
    this->ensureParametersValid();
    this->disparityBuf_.resize(this->parameters_.rightScanSteps + this->parameters_.leftScanSteps + 1, 0);
}

CudaDisparityMapGenerator::~CudaDisparityMapGenerator() {
    destroyCudaMemoryBuffers();
}

void CudaDisparityMapGenerator::setParameters(
        const DisparityMapAlgorithmParameters_t& parameters) {
    this->parameters_ = parameters;
    this->ensureParametersValid();
    this->disparityBuf_.resize(this->parameters_.rightScanSteps + this->parameters_.leftScanSteps + 1, 0);
}

const DisparityMapAlgorithmParameters_t& CudaDisparityMapGenerator::getParameters() const {
    return this->parameters_;
}

void CudaDisparityMapGenerator::computeDisparity(
        const cv::Mat& leftImage,
        const cv::Mat& rightImage,
        cv::Mat& disparity) {

    computeDisparityCuda(
        leftImage.rows,
        leftImage.cols,
        this->parameters_.blockSize,
        this->parameters_.leftScanSteps,
        this->parameters_.rightScanSteps,
        leftImage.data,
        rightImage.data,
        reinterpret_cast<float*>(disparity.data));
}

void CudaDisparityMapGenerator::ensureParametersValid() {
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

