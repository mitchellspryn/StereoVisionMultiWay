#include "../include/CudaSimdFunctions.h"

__device__
void computeSadOverBlockCudaSimd(
        int minYL,
        int minXL,
        int minYR,
        int minXR,
        int width,
        int height,
        int imageWidth,
        const uint8_t* leftImageData,
        const uint8_t* rightImageData,
        int* sum) {
    *sum = 0;
    int numStrides = width / 4;
    for (int y = 0; y < height; y++) {
        for (int n = 0; n < numStrides; n++) {
            int  leftBaseIdx = ((y+minYL)*imageWidth) + minXL + (n*4);
            int rightBaseIdx = ((y+minYR)*imageWidth) + minXR + (n*4);

            uint32_t leftVal = 
                (leftImageData[leftBaseIdx+3] << 24)
                |
                (leftImageData[leftBaseIdx+2] << 16)
                |
                (leftImageData[leftBaseIdx+1] << 8)
                |
                (leftImageData[leftBaseIdx+0]);

            uint32_t rightVal = 
                (rightImageData[rightBaseIdx+3] << 24)
                |
                (rightImageData[rightBaseIdx+2] << 16)
                |
                (rightImageData[rightBaseIdx+1] << 8)
                |
                (rightImageData[rightBaseIdx+0]);

            *sum += __vsadu4(leftVal, rightVal);
        }

        for (int x = numStrides*4; x < width; x++) {
             //__usad(a, b, c) = |a-b| + c
            *sum += __usad(
                    leftImageData[((y + minYL) * imageWidth) + (x + minXL)],
                    rightImageData[((y + minYR) * imageWidth) + (x + minXR)],
                    0);
        }
    }
}

__device__
void computeDisparityForPixelCudaSimd(
        int y, 
        int x,
        int imageWidth,
        int imageHeight,
        int blockSize,
        int leftScanSteps,
        int rightScanSteps,
        const uint8_t* leftImageData,
        const uint8_t* rightImageData,
        float* output) {

    float disparityBuf[512];
    int maxBlockStep = (blockSize - 1) / 2;

    int templateLeftHalfWidth = min(x, maxBlockStep);
    int templateRightHalfWidth = min(imageWidth - x - 1, maxBlockStep);
    int templateTopHalfHeight = min(y, maxBlockStep);
    int templateBottomHalfHeight = min(imageHeight - y - 1, maxBlockStep);

    int templateWidth = templateLeftHalfWidth + templateRightHalfWidth + 1;
    int templateHeight = templateTopHalfHeight + templateBottomHalfHeight + 1;

    int leftMinY = y - templateTopHalfHeight;
    int leftMinX = x - templateLeftHalfWidth;

    int rightMinStartX = max(0, x - leftScanSteps - templateLeftHalfWidth);
    int rightMaxStartX = min(imageWidth - templateWidth, x + rightScanSteps - templateLeftHalfWidth);

    int numSteps = rightMaxStartX - rightMinStartX;

    int bestIndex = 0;
    int bestSadValue = 2147483646; // value of std::numeric_limits<int>::max() - 1
    int zeroDisparityIndex = x - rightMinStartX - templateLeftHalfWidth;

    for (int xx = rightMinStartX; xx <= rightMaxStartX; xx++) {
        int sad = 0;
        computeSadOverBlockCudaSimd(
            leftMinY,
            leftMinX,
            leftMinY, // Ys are aligned for the two images
            xx,
            templateWidth,
            templateHeight,
            imageWidth,
            leftImageData, 
            rightImageData,
            &sad);

        disparityBuf[xx - rightMinStartX] = sad;

        if (sad < bestSadValue) {
            bestSadValue = sad;
            bestIndex = xx - rightMinStartX;
        }
    }

    float disparity = __int2float_rn(abs(bestIndex - zeroDisparityIndex));
    if ((bestIndex == 0)
        ||
        (bestIndex == numSteps)
        ||
        (bestSadValue == 0)) {
        *output = disparity;
    } else { 
        float c3 = disparityBuf[bestIndex+1];
        float c2 = disparityBuf[bestIndex];
        float c1 = disparityBuf[bestIndex-1];

        *output = disparity - (0.5 * ((c3 - c1) / (c1 - (2*c2) + c3)));
    }
}

__global__ 
void computeDisparityCudaInternalSimd(
        int height,
        int width,
        int blockSize,
        int leftScanSteps,
        int rightScanSteps,
        uint8_t* leftImageData,
        uint8_t* rightImageData,
        float* disparityData) {
    int index = blockIdx.x * blockDim.x + threadIdx.x;
    int stride = blockDim.x * gridDim.x;

    int numElements = height * width;
    for (int i = index; i < numElements; i += stride) {
        int y = i / width;
        int x = i % width;

        computeDisparityForPixelCudaSimd(
            y,
            x,
            width,
            height,
            blockSize,
            leftScanSteps,
            rightScanSteps,
            leftImageData,
            rightImageData,
            disparityData + i);
    }
}

static uint8_t* leftCudaData = NULL;
static uint8_t* rightCudaData = NULL;
static float* disparityCudaData = NULL;

void destroyCudaMemoryBuffersSimd() {
    if (leftCudaData != NULL) {
        cudaFree(leftCudaData);
        leftCudaData = NULL;
    }

    if (rightCudaData != NULL) {
        cudaFree(rightCudaData);
        rightCudaData = NULL;
    }

    if (disparityCudaData != NULL) {
        cudaFree(disparityCudaData);
        disparityCudaData = NULL;
    }
}

void computeDisparityCudaSimd(
        int imageHeight,
        int imageWidth, 
        int blockSize,
        int leftScanSteps,
        int rightScanSteps,
        uint8_t* leftImageData,
        uint8_t* rightImageData,
        float* disparityData) {

    int numElements = imageHeight * imageWidth;

    if (leftCudaData == NULL) {
        cudaMallocManaged(&leftCudaData, numElements * sizeof(uint8_t));
        cudaMallocManaged(&rightCudaData, numElements * sizeof(uint8_t));
        cudaMallocManaged(&disparityCudaData, numElements * sizeof(float));
    }

    cudaMemcpy(leftCudaData, leftImageData, numElements * sizeof(uint8_t), cudaMemcpyHostToDevice);
    cudaMemcpy(rightCudaData, rightImageData, numElements * sizeof(uint8_t), cudaMemcpyHostToDevice);

    int numThreads = 256;
    int numBlocks = ceil(((float)numElements) / ((float)numThreads));
    computeDisparityCudaInternalSimd<<<numBlocks, numThreads>>>(
        imageHeight,
        imageWidth,
        blockSize,
        leftScanSteps,
        rightScanSteps,
        leftCudaData,
        rightCudaData,
        disparityCudaData);

    cudaDeviceSynchronize();

    cudaMemcpy(disparityData, disparityCudaData, numElements * sizeof(float), cudaMemcpyDeviceToHost);
}
