#pragma once

#include <fstream>
#include <sstream>
#include <stdexcept>

#include <CL/cl.h>
#include <opencv2/core.hpp>

#include "DisparityMapAlgorithmParameters.hpp"
#include "DisparityMapGenerator.hpp"

class OpenClDisparityMapGenerator : public DisparityMapGenerator {
    public:
        OpenClDisparityMapGenerator(
            const DisparityMapAlgorithmParameters_t& parameters);

        virtual ~OpenClDisparityMapGenerator() override;

        virtual void setParameters(
            const DisparityMapAlgorithmParameters_t& parameters) override;

        virtual const DisparityMapAlgorithmParameters_t& getParameters() const override;
        
        virtual void computeDisparity(
            const cv::Mat& leftImage, 
            const cv::Mat& rightImage, 
            cv::Mat& disparity) override;

    private:
        DisparityMapAlgorithmParameters_t parameters_;

        bool openClKernelCreated_ = false;
        int imageWidth_;
        int imageHeight_;

        cl_platform_id oclPlatformId_;
        cl_device_id oclDeviceId_;
        cl_uint oclNumDevices_;
        cl_uint oclNumPlatforms_;

        cl_context oclContext_;
        cl_command_queue oclCommandQueue_;
        cl_mem oclLeftImageData_;
        cl_mem oclRightImageData_;
        cl_mem oclDisparityData_;
        cl_program oclProgram_;
        cl_kernel oclKernel_;

        void ensureParametersValid();
        void initializeOclKernel();
        void cleanOclKernel();
};
