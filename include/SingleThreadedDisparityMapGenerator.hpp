#pragma once

#include <stdexcept>

#include <opencv2/core.hpp>

#include "DisparityMapAlgorithmParameters.hpp"
#include "DisparityMapGenerator.hpp"

class SingleThreadedDisparityMapGenerator : public DisparityMapGenerator {
    public:
        SingleThreadedDisparityMapGenerator(
            const DisparityMapAlgorithmParameters_t& parameters);

        virtual void setParameters(
            const DisparityMapAlgorithmParameters_t& parameters) override;

        virtual const DisparityMapAlgorithmParameters_t& getParameters() const override;
        
        virtual void computeDisparity(
            const cv::Mat& leftImage, 
            const cv::Mat& rightImage, 
            cv::Mat& disparity) override;

    private:
        DisparityMapAlgorithmParameters_t parameters_;

        void ensureParametersValid();
};
