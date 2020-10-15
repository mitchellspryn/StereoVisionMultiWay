#pragma once

#include <stdexcept>

#include <opencv2/core.hpp>

#include "DisparityMapAlgorithmParameters.hpp"
#include "DisparityMapGenerator.hpp"

class CudaSimdDisparityMapGenerator : public DisparityMapGenerator {
    public:
        CudaSimdDisparityMapGenerator(
            const DisparityMapAlgorithmParameters_t& parameters);

        virtual ~CudaSimdDisparityMapGenerator() override;

        virtual void setParameters(
            const DisparityMapAlgorithmParameters_t& parameters) override;

        virtual const DisparityMapAlgorithmParameters_t& getParameters() const override;
        
        virtual void computeDisparity(
            const cv::Mat& leftImage, 
            const cv::Mat& rightImage, 
            cv::Mat& disparity) override;

    private:
        DisparityMapAlgorithmParameters_t parameters_;
        std::vector<int> disparityBuf_;

        uint8_t* leftCudaData = nullptr;
        uint8_t* rightCudaData = nullptr;
        float* disparityCudaData = nullptr;

        void ensureParametersValid();
};
