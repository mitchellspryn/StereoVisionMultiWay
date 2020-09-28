#include "../include/CommonArgumentParser.hpp"

DisparityMapAlgorithmParameters_t CommonArgumentParser::parseFromCommandLine(
        int argc, 
        char** argv) {
    DisparityMapAlgorithmParameters_t parameters;

    int i = 0;
    while (i < argc-1) {
        std::string flag(argv[i]);

        // If parsing fails, std::stoi will throw exception
        // Will be more helpful upon failure than trying to catch and re-throw
        if (flag == "--block-size") {
            parameters.blockSize = std::stoi(std::string(argv[i+1]));
            i += 2;
        } else if (flag == "--left-scan-steps") {
            parameters.leftScanSteps = std::stoi(std::string(argv[i+1]));
            i += 2;
        } else if (flag == "--right-scan-steps") {
            parameters.rightScanSteps = std::stoi(std::string(argv[i+1]));
            i += 2;
        } else if (flag == "--disparity-metric") {
            std::string metricName = std::string(argv[i+1]);
            if (metricName == "SUM_ABSOLUTE_DIFFERENCE") {
                parameters.disparityMetric == SUM_ABSOLUTE_DIFFERENCE;
            } else {
                throw std::runtime_error("Unrecognized disparity metric '"
                        + metricName
                        + "'.\n"
                        + "Valid options are: SUM_ABSOLUTE_DIFFERENCE.");
            }

            i += 2;
        } else if (flag == "--left-image-path") {
            parameters.leftImageFilePath = std::string(argv[i+1]);
            i += 2;
        } else if (flag == "--right-image-path") {
            parameters.rightImageFilePath = std::string(argv[i+1]);
            i += 2;
        } else if (flag == "--output-path") {
            parameters.outputPath = std::string(argv[i+1]);
            i += 2;
        } else if (flag == "--algorithm-name") {
            parameters.algorithmName = std::string(argv[i+1]);
            i += 2;
        } else {
            i++;
        }
    }

    return parameters;
}

