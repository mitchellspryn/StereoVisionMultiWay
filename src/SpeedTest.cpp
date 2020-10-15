#include <algorithm>
#include <chrono>
#include <fstream>
#include <iostream>
#include <iomanip>
#include <limits>
#include <memory>
#include <sstream>
#include <stdexcept>
#include <time.h>
#include <unistd.h>
#include <unordered_map>
#include <vector>

#include <opencv2/core.hpp>
#include <opencv2/core/utility.hpp>
#include <opencv2/imgcodecs.hpp>

#include "../include/DisparityMapAlgorithmParameters.hpp"
#include "../include/DisparityMapGenerator.hpp"
#include "../include/DisparityMapGeneratorFactory.hpp"

int main(int argc, char** argv) {

    const cv::String commandLineKeys = 
        "{help h usage ?         |          | This program runs a speed test on selected algorithms.}"
        "{leftImage              |   <none> | The left image to process.}"
        "{rightImage             |   <none> | The right image to proces.}"
        "{algorithmNames         |   <none> | The algorithms to benchmark, comma-separated.}"
        "{outputPath             | data.csv | The output directory to which to write the results.}"
        "{blockSize              |        7 | The maximum block size to use for matching.}"
        "{leftScanSteps          |       50 | The number of blocks to scan to the left.}"
        "{rightScanSteps         |       50 | The number of blocks to scan to the right.}"
        "{numIterations          |     1000 | The number of production iterations to run.}"
        "{warmUpIterations       |       50 | The number of iterations to perform before saving data. Used to warm up caches}"
        "{progressReportInterval |       20 | The number of iterations to perform before saving data. Used to warm up caches}";

    cv::CommandLineParser parser(argc, argv, commandLineKeys);

    if (!parser.check()) {
        parser.printMessage();
        parser.printErrors();
        return 1;
    }

    if (parser.has("help")
        || (!parser.has("leftImage"))
        || (!parser.has("rightImage"))
        || (!parser.has("algorithmNames"))) {
        parser.printMessage();
        return 1;
    }

    // Set all printouts to 3 decimal places
    std::cout << std::fixed << std::setprecision(3);

    DisparityMapAlgorithmParameters_t templateParameters;
    templateParameters.blockSize = parser.get<int>("blockSize");
    templateParameters.leftScanSteps = parser.get<int>("leftScanSteps");
    templateParameters.rightScanSteps = parser.get<int>("rightScanSteps");
    templateParameters.leftImageFilePath = std::string(parser.get<cv::String>("leftImage"));
    templateParameters.rightImageFilePath = std::string(parser.get<cv::String>("rightImage"));
    templateParameters.outputPath = std::string(parser.get<cv::String>("outputPath"));
    std::string algorithmNamesStr = std::string(parser.get<cv::String>("algorithmNames"));
    int numIterations = parser.get<int>("numIterations");
    int numWarmUpIterations = parser.get<int>("warmUpIterations");
    int progressReportInterval = parser.get<int>("progressReportInterval");

    std::cout 
        << "Reading in left image from '" 
        << templateParameters.leftImageFilePath 
        << "'..." 
        << std::endl;

    cv::Mat leftImage = cv::imread(templateParameters.leftImageFilePath, cv::IMREAD_GRAYSCALE);
    
    std::cout 
        << "Reading in right image from '" 
        << templateParameters.rightImageFilePath 
        << "'..." 
        << std::endl;

    cv::Mat rightImage = cv::imread(templateParameters.rightImageFilePath, cv::IMREAD_GRAYSCALE);

    if ((leftImage.rows == 0)
            ||
        (leftImage.cols == 0)) {
        throw std::runtime_error("Error. Left image is empty.");
    }

    if ((rightImage.rows == 0) 
            ||
        (rightImage.cols == 0)) {
        throw std::runtime_error("Error. Right image is empty.");
    }

    if ((leftImage.rows != rightImage.rows)
            ||
        (leftImage.cols != rightImage.cols)) {
        throw std::runtime_error(std::string("Error. Dimensions of input images are not the same.\n")
            + std::string("Left image: (") + std::to_string(leftImage.rows) + std::string("x") + std::to_string(leftImage.cols) + std::string(")\n")
            + std::string("Right iamge: (") + std::to_string(rightImage.rows) + std::string("x") + std::to_string(rightImage.cols) + std::string(")"));
    }

    std::cout << "Running benchmark with the following parameters:" << std::endl;
    std::cout << "\tAlgorithm Names: " << algorithmNamesStr << "." << std::endl;
    std::cout << "\tBlock Size: " << templateParameters.blockSize << "." << std::endl;
    std::cout << "\tLeft Scan Steps: " << templateParameters.leftScanSteps << "." << std::endl;
    std::cout << "\tRight Scan Steps: " << templateParameters.rightScanSteps << "." << std::endl;
    std::cout << "\tDisparity Metric: " << "SUM_ABSOLUTE_DIFFERENCE" << "." << std::endl;
    std::cout << "\tLeft Image: " << templateParameters.leftImageFilePath << "." << std::endl;
    std::cout << "\tRight Image: " << templateParameters.rightImageFilePath << "." << std::endl;
    std::cout << "\tImage Size: (" << leftImage.rows << "x" << leftImage.cols << ")." << std::endl;
    std::cout << "\tOutput Path: " << templateParameters.outputPath << std::endl;
    std::cout << "\tNumber of iterations: " << numIterations << std::endl;
    std::cout << "\tNumber of warm-up iterations: " << numWarmUpIterations << std::endl;
    std::cout << "\tProgress Report Interval: " << progressReportInterval << std::endl;

    std::stringstream stream(algorithmNamesStr);
    std::vector<std::string> algorithmNames;
    while (stream.good()) {
        std::string algorithmName;
        std::getline(stream, algorithmName, ',');
        algorithmNames.emplace_back(algorithmName);
    }

    std::unordered_map<std::string, std::vector<double>> wallClockProcessingTimes;
    std::unordered_map<std::string, std::vector<double>> cpuProcessingTimes;
    cv::Mat disparityImage(leftImage.rows, leftImage.cols, CV_32FC1);
    std::chrono::high_resolution_clock clk;
    clock_t t;

    for (size_t algorithmIdx = 0; algorithmIdx < algorithmNames.size(); algorithmIdx++) {
        const std::string& algorithmName = algorithmNames[algorithmIdx];
        wallClockProcessingTimes[algorithmName].resize(numIterations, 0);
        cpuProcessingTimes[algorithmName].resize(numIterations, 0);

        DisparityMapAlgorithmParameters_t localParameters(templateParameters);
        localParameters.algorithmName = algorithmName;
        std::cout << "Creating disparity generator for " << algorithmName << "..." << std::endl;

        DisparityMapGeneratorFactory factory;
        std::unique_ptr<DisparityMapGenerator> generator = factory.create(localParameters);

        std::cout << "Initializing disparity generator..." << std::endl;
        generator->setParameters(localParameters);

        std::cout << "Running warm-up iterations..." << std::endl;
        for (int i = 0; i < numWarmUpIterations; i++) {
            generator->computeDisparity(leftImage, rightImage, disparityImage);

            if (((i+1) % progressReportInterval == 0)) {
                std::cout << "\tProcessed " << (i+1) << " / " << numWarmUpIterations << " warm up iterations (" 
                    << static_cast<float>(i+1) * 100.0f / static_cast<float>(numWarmUpIterations) << "%)" << std::endl;
            }
        }
        
        std::cout << "Running production iterations..." << std::endl;
        for (int i = 0; i < numIterations; i++) {
            std::chrono::high_resolution_clock::time_point start = clk.now();
            t = clock();
            generator->computeDisparity(leftImage, rightImage, disparityImage);
            t = clock() - t;
            std::chrono::high_resolution_clock::time_point end = clk.now();

            cpuProcessingTimes[algorithmName][i] =  
                static_cast<float>(t) * 1000000.0f / static_cast<float>(CLOCKS_PER_SEC);
            wallClockProcessingTimes[algorithmName][i] =
                std::chrono::duration_cast<std::chrono::microseconds>(end-start).count();

            if (((i+1) % progressReportInterval == 0)) {
                std::cout << "\tProcessed " << (i+1) << " / " << numIterations << " production iterations ("
                    << static_cast<float>(i+1) * 100.0f / static_cast<float>(numIterations) << "%)" << std::endl;
            }
        }

        std::cout << "Data for " << algorithmName << " generated." << std::endl;

        if (algorithmIdx < algorithmNames.size() - 1) {
            std::cout << "Sleeping for 10 seconds to allow conditions to return to normal." << std::endl;
            usleep(10*1000000);
        }
    }

    std::cout << "Writing csv to " << templateParameters.outputPath << " ..." << std::endl;
    
    std::ofstream outputStream(templateParameters.outputPath, std::ios::out);
    for (size_t i = 0; i < algorithmNames.size(); i++) {
        outputStream << algorithmNames[i] + "_cpu";
        outputStream << ",";
        outputStream << algorithmNames[i] + "_wall";
        if (i != (algorithmNames.size() - 1)) {
            outputStream << ",";
        } else {
            outputStream << "\n";
        }
    }

    for (int exampleIndex = 0; exampleIndex < numIterations; exampleIndex++) {
        for (size_t i = 0; i < algorithmNames.size(); i++) {
            outputStream << cpuProcessingTimes[algorithmNames[i]][exampleIndex];
            outputStream << ",";
            outputStream << wallClockProcessingTimes[algorithmNames[i]][exampleIndex];
            if (i != (algorithmNames.size() - 1)) {
                outputStream << ",";
            } else {
                outputStream << "\n";
            }
        }
    }

    outputStream.flush();
    outputStream.close();

    std::cout << "Graceful termination" << std::endl;

    return 0;
}
