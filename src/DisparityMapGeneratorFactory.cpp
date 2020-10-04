#include "../include/CudaDisparityMapGenerator.hpp"
#include "../include/DisparityMapGeneratorFactory.hpp"
#include "../include/SingleThreadedDisparityMapGenerator.hpp"
#include "../include/OpenMpThreadedDisparityMapGenerator.hpp"

std::unique_ptr<DisparityMapGenerator> DisparityMapGeneratorFactory::create(
        const DisparityMapAlgorithmParameters_t& parameters) {
    
    if (this->caseInsensitiveStringsEqual(parameters.algorithmName, "SingleThreaded")) {
        return std::make_unique<SingleThreadedDisparityMapGenerator>(parameters);
    } else if (this->caseInsensitiveStringsEqual(parameters.algorithmName, "OpenMP")) {
        return std::make_unique<OpenMpThreadedDisparityMapGenerator>(parameters);
    } else if (this->caseInsensitiveStringsEqual(parameters.algorithmName, "CUDA")) {
        return std::make_unique<CudaDisparityMapGenerator>(parameters);
    }else {
        throw std::runtime_error("Unrecognized algorithmName '" 
            + parameters.algorithmName
            + "'.\n"
            + "Valid Options are 'SingleThreaded','OpenMP','CUDA'.");
    }
}

bool DisparityMapGeneratorFactory::caseInsensitiveStringsEqual(
        const std::string& s1,
        const std::string& s2) {
    if (s1.size() != s2.size()) {
        return false;
    }

    for (size_t i = 0; i < s1.size(); i++) {
        if (toupper(s1[i]) != toupper(s2[i])) {
            return false;
        }
    }

    return true;
}
