#pragma once

#include <memory>
#include <string>

#include "DisparityMapAlgorithmParameters.hpp"
#include "DisparityMapGenerator.hpp"

class DisparityMapGeneratorFactory {
    public:
        std::unique_ptr<DisparityMapGenerator> create(
                const DisparityMapAlgorithmParameters_t& parameters);

    private:
        bool caseInsensitiveStringsEqual(
            const std::string& s1,
            const std::string& s2);
};
