#pragma once

#include <stdexcept>

#include "DisparityMapAlgorithmParameters.hpp"

class CommonArgumentParser {
    public:
        DisparityMapAlgorithmParameters_t parseFromCommandLine(int argc, char** argv);
};
