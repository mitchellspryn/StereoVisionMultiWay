#pragma once

#include <omp.h>
#include <stdexcept>

#include <opencv2/core.hpp>

#include "DisparityMapAlgorithmParameters.hpp"
#include "DisparityMapGenerator.hpp"

class OpenMpThreadedDisparityMapGenerator : public DisparityMapGenerator {
    public:
        OpenMpThreadedDisparityMapGenerator(
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
        float computeDisparityForPixel(
                int y, 
                int x, 
                const cv::Mat& leftImage, 
                const cv::Mat& rightImage);

        int computeSadOverBlock(
                int minYL,
                int minXL,
                int minYR,
                int minXR,
                int width,
                int height,
                const cv::Mat& leftImage,
                const cv::Mat& rightImage);
};
