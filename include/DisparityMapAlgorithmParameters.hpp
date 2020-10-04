#pragma once

#include <string>

typedef struct DisparityMapAlgorithmParameters {
    int blockSize = 7;
    int leftScanSteps = 50;
    int rightScanSteps = 50;
    std::string leftImageFilePath;
    std::string rightImageFilePath;
    std::string outputPath;
    std::string algorithmName;
} DisparityMapAlgorithmParameters_t;
