#include "../include/SingleThreadedDisparityMapGenerator.hpp"

SingleThreadedDisparityMapGenerator::SingleThreadedDisparityMapGenerator(
        const DisparityMapAlgorithmParameters_t& parameters)
        : parameters_(parameters) {
    this->ensureParametersValid();
}

void SingleThreadedDisparityMapGenerator::setParameters(
        const DisparityMapAlgorithmParameters_t& parameters) {
    this->parameters_ = parameters;
    this->ensureParametersValid();
}

const DisparityMapAlgorithmParameters_t& SingleThreadedDisparityMapGenerator::getParameters() const {
    return this->parameters_;
}

void SingleThreadedDisparityMapGenerator::computeDisparity(
        const cv::Mat& leftImage,
        const cv::Mat& rightImage,
        cv::Mat& disparity) {
    
    // TODO: write an actual implementation
    // For now, just togggle the colors.
    for (int y = 0; y < disparity.rows; y++) {
        for (int x = 0; x < disparity.cols; x++) {
            disparity.at<float>(y, x) = (y+x) % 256;
        }
    }
}

void SingleThreadedDisparityMapGenerator::ensureParametersValid() {
    if (this->parameters_.disparityMetric != SUM_ABSOLUTE_DIFFERENCE) {
        throw std::runtime_error("Unsupported disparity metric.");
    }   

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

