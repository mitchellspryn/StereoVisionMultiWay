#include <algorithm>
#include <chrono>
#include <fstream>
#include <iostream>
#include <limits>
#include <random>
#include <time.h>
#include <stdexcept>
#include <unordered_map>
#include <vector>

#include <opencv2/core/utility.hpp>

#include <immintrin.h>

typedef struct quickStatistics {
    double mean = 0;
    double stdev = 0;
} quickStatistics_t;

static constexpr int ALIGNMENT_SIZE = 32;

// Even forcing no avx usage by the compiler doesn't change perf!
__attribute__((optimize("no-tree-vectorize")))
long computeSad(
        const uint8_t* a, 
        const uint8_t* b,
        const int n) {
    long result = 0;

    asm("# Start non-SIMD loop");
    for (size_t i = 0; i < n; i++) {
        result += static_cast<long>(std::abs(a[i] - b[i]));
    }
    asm("# End non-SIMD loop");

    return result;
}

long computeSadSimd(
        const uint8_t* a, 
        const uint8_t* b,
        const int n) {
    asm("# Start SIMD loop");
    union { __m256i accumulator; uint64_t accumulatorValues[4]; };
    union { __m256i workRegA; uint8_t workRegABytes[32]; };
    union { __m256i workRegB; uint8_t workRegBBytes[32]; };
    union { __m256i sadReg; uint8_t sadRegBytes[32]; };
    accumulator = _mm256_setzero_si256();
    constexpr int numBytesPerSimd = 32;
    size_t numSimdIter = n / numBytesPerSimd;
    for (size_t i = 0 ; i < numSimdIter; i++) {
        workRegA = _mm256_load_si256(
            reinterpret_cast<__m256i const*>(a + (i *numBytesPerSimd)));
        workRegB = _mm256_load_si256(
            reinterpret_cast<__m256i const*>(b + (i * numBytesPerSimd)));
        sadReg = _mm256_sad_epu8(workRegA, workRegB);
        accumulator = _mm256_add_epi64(sadReg, accumulator);
    }
    asm("# End manual unroll");
    
    long result = 0;
    for (int i = 0; i < 4; i++) {
        result += static_cast<long>(accumulatorValues[i]);
    }

    for (size_t i = (numSimdIter*numBytesPerSimd); i < n; i++) {
        result += static_cast<long>(std::abs(a[i] - b[i]));
    }

    asm("# End SIMD loop");

    return result;
}

uint8_t* generateRandomNumbers(int num) {
    std::default_random_engine generator;
    std::uniform_int_distribution<uint8_t> distribution(
            std::numeric_limits<uint8_t>::min(),
            std::numeric_limits<uint8_t>::max());

    generator.seed(std::chrono::system_clock::now().time_since_epoch().count());

    uint8_t* result = static_cast<uint8_t*>(_mm_malloc(num, ALIGNMENT_SIZE));
    for (int i = 0; i < num; i++) {
        result[i] = distribution(generator);
    }

    return result;
}

std::vector<double> benchmark(
        int numDataPoints,
        int numWarmupTrials,
        int numTrials,
        long checkNumber,
        long (*benchmarkFunc)(const uint8_t* a, const uint8_t* b, const int n),
        const uint8_t* a,
        const uint8_t* b) {
    std::vector<double> result;
    result.reserve(numTrials);

    clock_t t;

    for (size_t i = 0; i < numWarmupTrials; i++) {
        long sad = benchmarkFunc(a, b, numDataPoints);
        if (sad != checkNumber) {
            throw std::runtime_error("Check values did not match. Expected " 
                    + std::to_string(checkNumber)
                    + ", but got " 
                    + std::to_string(sad)
                    + ".");
        }
    }

    for (size_t i = 0; i < numTrials; i++) {
        t = clock();
        long sad = benchmarkFunc(a, b, numDataPoints);
        t = clock() - t;
        if (sad != checkNumber) {
            throw std::runtime_error("Check values did not match. Expected " 
                    + std::to_string(checkNumber)
                    + ", but got " 
                    + std::to_string(sad)
                    + ".");
        }

        result.emplace_back(
                static_cast<float>(t) * 100000.0f / static_cast<float>(CLOCKS_PER_SEC));
    }

    return result;
}

quickStatistics_t computeRuntimeStatistics(const std::vector<double>& runtimes) {
    quickStatistics_t result;
    double numDataPoints = static_cast<double>(runtimes.size());

    result.mean = std::accumulate(
            runtimes.begin(),
            runtimes.end(),
            0) / numDataPoints;

    result.stdev = 0;
    for (size_t i = 0; i < runtimes.size(); i++) {
        result.stdev += (runtimes[i] - result.mean) * (runtimes[i] - result.mean);
    }
    result.stdev /= numDataPoints;
    result.stdev = std::sqrt(result.stdev);

    return result;
}


int main(int argc, char** argv) {
    const cv::String commandLineKeys = 
        "{help h usage ?  |                         | A quick program to generate runtimes for SAD algorithms using SIMD computation.}"
        "{numDataPoints   |                     100 | The number of data points to process on each iteration.}"
        "{numWarmupTrials |                      50 | The number of iterations to perform before beginning to record results.}"
        "{numTrials       |                     100 | The number of data points to collect.}"
        "{outputFileName  |                 out.csv | The output file to which to write data.}";

    cv::CommandLineParser parser(argc, argv, commandLineKeys);

    if (!parser.check()) {
        parser.printMessage();
        parser.printErrors();
        return 1;
    }

    if (parser.has("help")) {
        parser.printMessage();
        return 1;
    }

    int numDataPoints = parser.get<int>("numDataPoints");
    int numWarmupTrials = parser.get<int>("numWarmupTrials");
    int numTrials = parser.get<int>("numTrials");
    std::string outputFileName = std::string(parser.get<cv::String>("outputFileName"));

    std::cout << "Performing a trial with the following parameters:\n";
    std::cout << "\tNumber of data points: " << numDataPoints << "\n";
    std::cout << "\tNumber of warmup trials: " << numWarmupTrials << "\n";
    std::cout << "\tNumber of trials: " << numTrials << std::endl;

    std::cout << "Generating data..." << std::endl;
    
    uint8_t* a = generateRandomNumbers(numDataPoints);
    uint8_t* b = generateRandomNumbers(numDataPoints);

    std::cout << "Generated " << numDataPoints << " data points." << std::endl;

    std::cout << "Computing check number using single threaded implementation..." << std::endl;
    long checkValue = computeSad(a, b, numDataPoints);
    std::cout << "Check value is " << checkValue << std::endl;

    std::cout << "Benchmarking nonSimd SAD implementation..." << std::endl;
    long (*benchmarkFunc)(const uint8_t* a, const uint8_t* b, const int n) = computeSad;
    std::vector<double> nonSimdRuntimes = benchmark(
            numDataPoints,
            numWarmupTrials,
            numTrials,
            checkValue,
            benchmarkFunc,
            a,
            b);

    if (nonSimdRuntimes.size() != numTrials) {
        throw std::runtime_error("Test harness is broken. Expected " 
                + std::to_string(numTrials) 
                + " trials to be performed, but actually performed "
                + std::to_string(nonSimdRuntimes.size())
                + " trials.");
    }

    std::cout << "Completed.\n";
    std::cout << "Benchmarking SIMD implementation..." << std::endl;
    benchmarkFunc = computeSadSimd;
    std::vector<double> simdRuntimes = benchmark(
            numDataPoints,
            numWarmupTrials,
            numTrials,
            checkValue,
            benchmarkFunc,
            a,
            b);

    if (simdRuntimes.size() != numTrials) {
        throw std::runtime_error("Test harness is broken. Expected " 
                + std::to_string(numTrials) 
                + " trials to be performed, but actually performed "
                + std::to_string(simdRuntimes.size())
                + " trials.");
    }

    std::cout << "Completed.\n";
    std::cout << "Writing to " << outputFileName << "..." << std::endl;

    if (simdRuntimes.size() != nonSimdRuntimes.size()) {
        throw std::runtime_error("Cannot write output file. Simd recorded "
                + std::to_string(simdRuntimes.size())
                + " samples, while non-simd recorded "
                + std::to_string(nonSimdRuntimes.size())
                + " samples.");
    }

    std::ofstream outputStream;
    outputStream.open(outputFileName, std::ios::out);

    outputStream << "non_simd, simd\n";
    for (size_t i = 0; i < simdRuntimes.size(); i++) {
        outputStream << nonSimdRuntimes[i] << "," << simdRuntimes[i] << "\n";
    }

    outputStream.flush();
    outputStream.close();

    std::cout << "File writen. Computing quick statistics..." << std::endl;
    quickStatistics_t simdRuntimeStatistics = computeRuntimeStatistics(simdRuntimes);
    quickStatistics_t nonSimdRuntimeStatistics = computeRuntimeStatistics(nonSimdRuntimes);

    std::cout << "Runtime statistics:\n";
    std::cout << "\tnon-simd: " 
        << nonSimdRuntimeStatistics.mean 
        << " ± " 
        << nonSimdRuntimeStatistics.stdev
        << ".\n";
    std::cout << "\tsimd: "
        << simdRuntimeStatistics.mean
        << " ± " 
        << simdRuntimeStatistics.stdev
        << "."
        << std::endl;

    std::cout << "Cleaning up..." << std::endl;

    free(a);
    free(b);

    std::cout << "Graceful termination." << std::endl;

    return 0;
}

