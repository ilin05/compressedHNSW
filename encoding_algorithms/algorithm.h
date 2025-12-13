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
        using EncoderFactoryWithSharedOut = std::function<std::unique_ptr<Encoder>(std::shared_ptr<utils::BaseStreamWriter>)>;
        using DecoderFactoryWithSharedIn = std::function<std::unique_ptr<Decoder>(std::shared_ptr<utils::BaseBlockStreamReader>)>;
        using EncoderFactoryWithSharedOutAndConfig = std::function<std::unique_ptr<Encoder>(std::shared_ptr<utils::BaseStreamWriter>, const std::string&)>;
        using DecoderFactoryWithSharedInAndConfig = std::function<std::unique_ptr<Decoder>(std::shared_ptr<utils::BaseBlockStreamReader>, const std::string&)>;

        std::map<std::string, EncoderFactory> encoderFactoryMap;
        std::map<std::string, DecoderFactory> decoderFactoryMap;
        std::map<std::string, EncoderFactoryWithConfig> encoderFactoryWithConfigMap;
        std::map<std::string, DecoderFactoryWithConfig> decoderFactoryWithConfigMap;
        std::map<std::string, EncoderFactoryWithSharedOut> encoderFactoryWithSharedOutMap;
        std::map<std::string, DecoderFactoryWithSharedIn> decoderFactoryWithSharedInMap;
        std::map<std::string, EncoderFactoryWithSharedOutAndConfig> encoderFactoryWithSharedOutAndConfigMap;
        std::map<std::string, DecoderFactoryWithSharedInAndConfig> decoderFactoryWithSharedInAndConfigMap;

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

        // 用StreamWriter构造Encoder
        std::unique_ptr<Encoder> getEncoder(const std::string& data_type, std::shared_ptr<utils::BaseStreamWriter> sharedOut, const std::string& config) {
            auto it = encoderFactoryWithSharedOutAndConfigMap.find(data_type);
            if (it != encoderFactoryWithSharedOutAndConfigMap.end()) {
                return it->second(std::move(sharedOut), config);
            }
            throw std::runtime_error("No Such Encoder with config for data type: " + data_type);
        }

        std::unique_ptr<Encoder> getEncoder(const std::string& data_type, std::shared_ptr<utils::BaseStreamWriter> sharedOut) {
            auto it = encoderFactoryWithSharedOutMap.find(data_type);
            if (it != encoderFactoryWithSharedOutMap.end()) {
                return it->second(std::move(sharedOut));
            }
            throw std::runtime_error("No Such Encoder with config for data type: " + data_type);
        }

        // 用BlockStreamReader构造Decoder
        std::unique_ptr<Decoder> getDecoder(const std::string& data_type, std::shared_ptr<utils::BaseBlockStreamReader> sharedIn, const std::string& config) {
            auto it = decoderFactoryWithSharedInAndConfigMap.find(data_type);
            if (it != decoderFactoryWithSharedInAndConfigMap.end()) {
                return it->second(std::move(sharedIn), config);
            }
            throw std::runtime_error("No Such Decoder with config for data type: " + data_type);
        }

        std::unique_ptr<Decoder> getDecoder(const std::string& data_type, std::shared_ptr<utils::BaseBlockStreamReader> sharedIn) {
            auto it = decoderFactoryWithSharedInMap.find(data_type);
            if (it != decoderFactoryWithSharedInMap.end()) {
                return it->second(std::move(sharedIn));
            }
            throw std::runtime_error("No Such Decoder with config for data type: " + data_type);
        }


    };

}
