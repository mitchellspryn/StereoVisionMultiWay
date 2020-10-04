#ifndef CUDA_FUNCTIONS_H
#define CUDA_FUNCTIONS_H

#include <stdint.h>
#include <stdio.h>

extern "C" {
    void computeDisparityCuda(
        int imageHeight,
        int imageWidth,
        int blockSize,
        int leftScanSteps,
        int rightScanSteps,
        uint8_t* leftImageData,
        uint8_t* rightImageData,
        float* disparityData);
}

#endif
