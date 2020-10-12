#include "../include/OpenClDisparityMapGenerator.hpp"

#include <iostream>

OpenClDisparityMapGenerator::OpenClDisparityMapGenerator(
        const DisparityMapAlgorithmParameters_t& parameters)
        : parameters_(parameters) {
    this->ensureParametersValid();
}

OpenClDisparityMapGenerator::~OpenClDisparityMapGenerator() {
    if (this->openClKernelCreated_) {
        this->cleanOclKernel();
        this->openClKernelCreated_ = false;
    }
}

void OpenClDisparityMapGenerator::setParameters(
        const DisparityMapAlgorithmParameters_t& parameters) {
    this->parameters_ = parameters;
    this->ensureParametersValid();
}

const DisparityMapAlgorithmParameters_t& OpenClDisparityMapGenerator::getParameters() const {
    return this->parameters_;
}

void OpenClDisparityMapGenerator::computeDisparity(
        const cv::Mat& leftImage,
        const cv::Mat& rightImage,
        cv::Mat& disparity) {
    if (!this->openClKernelCreated_) {
        this->imageWidth_ = leftImage.cols;
        this->imageHeight_ = leftImage.rows;
        this->initializeOclKernel();
    }
    
    size_t numPixels = this->imageWidth_ * this->imageHeight_;
    size_t localItemSize = 100;

    cl_int ret = clEnqueueWriteBuffer(
        this->oclCommandQueue_,       
        this->oclLeftImageData_,
        CL_TRUE,                     // blocking write
        0,                           // offset
        numPixels * sizeof(uint8_t), // size
        leftImage.data,              // buffer
        0,                           // num_events_in_wait_list
        NULL,                        // event_wait_list
        NULL);                       // event

    ret = clEnqueueWriteBuffer(
        this->oclCommandQueue_,       
        this->oclRightImageData_,
        CL_TRUE,                     // blocking write
        0,                           // offset
        numPixels * sizeof(uint8_t), // size
        rightImage.data,             // buffer
        0,                           // num_events_in_wait_list
        NULL,                        // event_wait_list
        NULL);                       // event

    ret = clEnqueueNDRangeKernel(
        this->oclCommandQueue_,
        this->oclKernel_,
        1,              // dimensions
        NULL,           // global work offset
        &numPixels,     // global work size
        &localItemSize, // local work size
        0,              // num_events_in_wait_list
        NULL,           // event_wait_list (NULL == don't wait)
        NULL);          // event (could be used in another kernel's wait_list)

    ret = clEnqueueReadBuffer(
        this->oclCommandQueue_,
        this->oclDisparityData_,
        CL_TRUE,                    // blocking read
        0,                          // offset
        numPixels * sizeof(float),  // size
        disparity.data,             // output data
        0,                          // num_events_in_wait_list
        NULL,                       // event_wait_list
        NULL);                      // event
}

void OpenClDisparityMapGenerator::ensureParametersValid() {
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

void OpenClDisparityMapGenerator::initializeOclKernel() {
    int numPixels = this->imageWidth_ * this->imageHeight_;

    std::ifstream openclProgramFile("OpenClFunctions.cl");
    std::stringstream buf;
    buf << openclProgramFile.rdbuf();
    std::string openclProgramText = buf.str();

    cl_int ret = clGetPlatformIDs(
            1,                     // num_entries
            &this->oclPlatformId_,
            &this->oclNumPlatforms_);

    ret = clGetDeviceIDs(
            this->oclPlatformId_,
            CL_DEVICE_TYPE_DEFAULT, // CL_DEVICE_TYPE_GPU, CL_DEVICE_TYPE_CPU (CPU needs drivers)
            1,                      // num_entries
            &this->oclDeviceId_,
            &this->oclNumDevices_);

    this->oclContext_ = clCreateContext(
            NULL,             // context properties
            1,                // num devices
            &this->oclDeviceId_, 
            NULL,             // callback for reporting errors
            NULL,             // user_data for error callback
            &ret);

    this->oclCommandQueue_ = clCreateCommandQueue(
            this->oclContext_,
            this->oclDeviceId_,
            0,                 // properties (no profiling, no out-of-order)
            &ret);

    this->oclLeftImageData_ = clCreateBuffer(
            this->oclContext_,
            CL_MEM_READ_ONLY,
            numPixels * sizeof(uint8_t),
            NULL,              // buffer preallocated by the host
            &ret);

    this->oclRightImageData_ = clCreateBuffer(
            this->oclContext_,
            CL_MEM_READ_ONLY,
            numPixels * sizeof(uint8_t),
            NULL, 
            &ret);

    this->oclDisparityData_ = clCreateBuffer(
            this->oclContext_,
            CL_MEM_WRITE_ONLY,
            numPixels * sizeof(float),
            NULL,
            &ret);

    const char* programTxt = openclProgramText.c_str();
    size_t sz = openclProgramText.size();
    this->oclProgram_ = clCreateProgramWithSource(
            this->oclContext_,
            1, // count (e.g. number of programs)
            static_cast<const char**>(&programTxt),
            static_cast<const size_t*>(&sz),
            &ret);

    ret = clBuildProgram(
            this->oclProgram_, 
            1, 
            &this->oclDeviceId_, 
            NULL,   // program build options
            NULL,   // error callback
            NULL);  // user data for error callback

    if (ret != CL_SUCCESS) {
        size_t length;
        char buffer[8192];
        clGetProgramBuildInfo(
            this->oclProgram_,
            this->oclDeviceId_,
            CL_PROGRAM_BUILD_LOG,
            sizeof(buffer),
            buffer,
            &length);

        std::string error(buffer);
        throw std::runtime_error(error);
    }
    
    this->oclKernel_ = clCreateKernel(this->oclProgram_, "computeDisparityOpenClKernel", &ret);

    ret = clSetKernelArg(
            this->oclKernel_, 
            0, 
            sizeof(int), 
            &(this->imageHeight_));
    ret = clSetKernelArg(
            this->oclKernel_, 
            1, 
            sizeof(int), 
            &(this->imageWidth_));
    ret = clSetKernelArg(
            this->oclKernel_, 
            2, 
            sizeof(int), 
            &(this->parameters_.blockSize));
    ret = clSetKernelArg(
            this->oclKernel_, 
            3, 
            sizeof(int), 
            &(this->parameters_.leftScanSteps));
    ret = clSetKernelArg(
            this->oclKernel_, 
            4, 
            sizeof(int), 
            &(this->parameters_.rightScanSteps));
    ret = clSetKernelArg(
            this->oclKernel_, 
            5, 
            sizeof(cl_mem), 
            &(this->oclLeftImageData_));
    ret = clSetKernelArg(
            this->oclKernel_, 
            6, 
            sizeof(cl_mem), 
            &(this->oclRightImageData_));
    ret = clSetKernelArg(
            this->oclKernel_, 
            7, 
            sizeof(cl_mem), 
            &(this->oclDisparityData_));

    this->openClKernelCreated_ = true;
}

void OpenClDisparityMapGenerator::cleanOclKernel() {
    cl_int ret = clFlush(this->oclCommandQueue_);
    ret = clFinish(this->oclCommandQueue_);
    ret = clReleaseKernel(this->oclKernel_);
    ret = clReleaseProgram(this->oclProgram_);
    ret = clReleaseMemObject(this->oclLeftImageData_);
    ret = clReleaseMemObject(this->oclRightImageData_);
    ret = clReleaseMemObject(this->oclDisparityData_);
    ret = clReleaseCommandQueue(this->oclCommandQueue_);
    ret = clReleaseContext(this->oclContext_);
}
