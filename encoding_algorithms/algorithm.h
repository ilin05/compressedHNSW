#pragma once

#include "encoder.h"
#include "decoder.h"
#include <string>
#include <map>
#include <set>
#include <memory>
#include <functional>
#include <stdexcept>

namespace encoding_algorithm {

    class Algorithm {
    protected:
        using EncoderFactory = std::function<std::unique_ptr<Encoder>(const std::string&)>;
        using DecoderFactory = std::function<std::unique_ptr<Decoder>(const std::string&)>;
    using EncoderFactoryWithConfig = std::function<std::unique_ptr<Encoder>(const std::string&, const std::string&)>;
    using DecoderFactoryWithConfig = std::function<std::unique_ptr<Decoder>(const std::string&, const std::string&)>;

        std::map<std::string, EncoderFactory> encoderFactoryMap;
        std::map<std::string, DecoderFactory> decoderFactoryMap;
        std::map<std::string, EncoderFactoryWithConfig> encoderFactoryWithConfigMap;
        std::map<std::string, DecoderFactoryWithConfig> decoderFactoryWithConfigMap;

    public:
        virtual ~Algorithm() = default;

        std::set<std::string> getSupportedDataTypes() {
            std::set<std::string> keys;
            for (const auto& pair : encoderFactoryMap) {
                keys.insert(pair.first);
            }
            return keys;
        }

        std::unique_ptr<Encoder> getEncoder(const std::string& data_type, const std::string& output_path) {
            auto it = encoderFactoryMap.find(data_type);
            if (it != encoderFactoryMap.end()) {
                return it->second(output_path);
            }
            throw std::runtime_error("No Such Encoder for data type: " + data_type);
        }

        std::unique_ptr<Decoder> getDecoder(const std::string& data_type, const std::string& input_path) {
            auto it = decoderFactoryMap.find(data_type);
            if (it != decoderFactoryMap.end()) {
                return it->second(input_path);
            }
            throw std::runtime_error("No Such Decoder for data type: " + data_type);
        }

        std::unique_ptr<Encoder> getEncoder(const std::string& data_type, const std::string& output_path, const std::string& config) {
            auto it = encoderFactoryWithConfigMap.find(data_type);
            if (it != encoderFactoryWithConfigMap.end()) {
                return it->second(output_path, config);
            }
            throw std::runtime_error("No Such Encoder with config for data type: " + data_type);
        }

        std::unique_ptr<Decoder> getDecoder(const std::string& data_type, const std::string& input_path, const std::string& config) {
            auto it = decoderFactoryWithConfigMap.find(data_type);
            if (it != decoderFactoryWithConfigMap.end()) {
                return it->second(input_path, config);
            }
            throw std::runtime_error("No Such Decoder with config for data type: " + data_type);
        }
    };

}
