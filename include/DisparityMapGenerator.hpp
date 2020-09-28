#pragma once

#include <opencv2/core.hpp>

#include "DisparityMapAlgorithmParameters.hpp"

class DisparityMapGenerator {
    public:
        virtual ~DisparityMapGenerator() {};

        virtual void setParameters(
            const DisparityMapAlgorithmParameters_t& parameters) = 0;

        virtual const DisparityMapAlgorithmParameters_t& getParameters() const = 0;

        virtual void computeDisparity(
            const cv::Mat& leftImage, 
            const cv::Mat& rightImage, 
            cv::Mat& disparity) = 0;
};
