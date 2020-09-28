#pragma once

#include <string>

typedef enum DisparityMetric {
    SUM_ABSOLUTE_DIFFERENCE = 0
} DisparityMetric_t;

typedef struct DisparityMapAlgorithmParameters {
    int blockSize = 7;
    int leftScanSteps = 50;
    int rightScanSteps = 50;
    DisparityMetric_t disparityMetric = SUM_ABSOLUTE_DIFFERENCE;
    std::string leftImageFilePath;
    std::string rightImageFilePath;
    std::string outputPath;
    std::string algorithmName;
} DisparityMapAlgorithmParameters_t;
