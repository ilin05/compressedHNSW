#pragma once

#include "algorithm.h"
#include "dexor/dexor.h"
#include "gorilla/gorilla.h"
#include "elf/elf.h"
#include "camel/camel.h"
// Include other algorithm headers here, e.g., #include "gorilla/gorilla.h"
#include <string>
#include <map>
#include <memory>
#include <functional>
#include <stdexcept>

namespace encoding_algorithm {

    // This would be where the enums from Java are defined, perhaps as const strings
    namespace EncodingAlgorithmEnums {
    const std::string DEXOR = "DeXOR";
    const std::string GORILLA = "Gorilla";
    const std::string ELF = "Elf";
    const std::string CAMEL = "Camel";
        // ... other algorithms
    }

    class AlgorithmsManager {
    private:
        using AlgorithmFactory = std::function<std::unique_ptr<Algorithm>()>;
        static std::map<std::string, AlgorithmFactory> algorithmFactoryMap;

        static void initialize() {
            if (algorithmFactoryMap.empty()) {
                algorithmFactoryMap[EncodingAlgorithmEnums::DEXOR] = []() {
                    return std::make_unique<dexor::DeXOR>();
                };
                algorithmFactoryMap[EncodingAlgorithmEnums::GORILLA] = []() {
                    return std::make_unique<gorilla::Gorilla>();
                };
                algorithmFactoryMap[EncodingAlgorithmEnums::ELF] = []() {
                    return std::make_unique<elf::Elf>();
                };
                algorithmFactoryMap[EncodingAlgorithmEnums::CAMEL] = []() {
                    return std::make_unique<camel::Camel>();
                };
                // Register other algorithms here
                // algorithmFactoryMap[EncodingAlgorithmEnums::GORILLA] = []() {
                //     return std::make_unique<gorilla::Gorilla>();
                // };
            }
        }

    public:
        static std::unique_ptr<Algorithm> getAlgorithm(const std::string& algorithm_name) {
            initialize();
            auto it = algorithmFactoryMap.find(algorithm_name);
            if (it != algorithmFactoryMap.end()) {
                return it->second();
            }
            throw std::runtime_error("No Such Algorithm: " + algorithm_name);
        }

        static std::unique_ptr<Encoder> getEncoder(const std::string& data_type, const std::string& algorithm_name, const std::string& output_path) {
            auto instance = getAlgorithm(algorithm_name);
            return instance->getEncoder(data_type, output_path);
        }

        static std::unique_ptr<Decoder> getDecoder(const std::string& data_type, const std::string& algorithm_name, const std::string& input_path) {
            auto instance = getAlgorithm(algorithm_name);
            return instance->getDecoder(data_type, input_path);
        }

        static std::unique_ptr<Encoder> getEncoder(const std::string& data_type, const std::string& algorithm_name, const std::string& output_path, const std::string& config) {
            auto instance = getAlgorithm(algorithm_name);
            return instance->getEncoder(data_type, output_path, config);
        }

        static std::unique_ptr<Decoder> getDecoder(const std::string& data_type, const std::string& algorithm_name, const std::string& input_path, const std::string& config) {
            auto instance = getAlgorithm(algorithm_name);
            return instance->getDecoder(data_type, input_path, config);
        }

        static std::unique_ptr<Encoder> getEncoder(const std::string& data_type, const std::string& algorithm_name, std::shared_ptr<utils::StreamWriter> sharedOut, const std::string& config) {
            auto instance = getAlgorithm(algorithm_name);
            return instance->getEncoder(data_type, std::move(sharedOut), config);
        }

        static std::unique_ptr<Encoder> getEncoder(const std::string& data_type, const std::string& algorithm_name, std::shared_ptr<utils::StreamWriter> sharedOut) {
            auto instance = getAlgorithm(algorithm_name);
            return instance->getEncoder(data_type, std::move(sharedOut));
        }

        static std::unique_ptr<Decoder> getDecoder(const std::string& data_type, const std::string& algorithm_name, std::shared_ptr<utils::BlockStreamReader> sharedIn, const std::string& config) {
            auto instance = getAlgorithm(algorithm_name);
            return instance->getDecoder(data_type, std::move(sharedIn), config);
        }

        static std::unique_ptr<Decoder> getDecoder(const std::string& data_type, const std::string& algorithm_name, std::shared_ptr<utils::BlockStreamReader> sharedIn) {
            auto instance = getAlgorithm(algorithm_name);
            return instance->getDecoder(data_type, std::move(sharedIn));
        }
    };
    
    // Static member initialization
    std::map<std::string, std::function<std::unique_ptr<Algorithm>()>> AlgorithmsManager::algorithmFactoryMap;

}
