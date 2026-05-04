#pragma once

#include "../algorithm.h"
#include "encoder/double_gorilla_encoder.h"
#include "decoder/double_gorilla_decoder.h"
#include <memory>
#include <string>

namespace encoding_algorithm {
namespace gorilla {

    // Assuming DataTypeEnums.DOUBLE.getType() returns "Double"
    const std::string DATA_TYPE_DOUBLE = "Double";

    class Gorilla : public Algorithm {
    public:
        Gorilla() {
            // Encoder
            encoderFactoryMap[DATA_TYPE_DOUBLE] = [](const std::string& path) {
                return std::make_unique<DoubleGorillaEncoder>(path);
            };
            encoderFactoryWithConfigMap[DATA_TYPE_DOUBLE] = [](const std::string& path, const std::string& config) {
                return std::make_unique<DoubleGorillaEncoder>(path, config);
            };
            encoderFactoryWithSharedOutMap[DATA_TYPE_DOUBLE] = [](std::shared_ptr<utils::BaseStreamWriter> sharedOut) {
                return std::make_unique<DoubleGorillaEncoder>(sharedOut);
            };
            encoderFactoryWithSharedOutAndConfigMap[DATA_TYPE_DOUBLE] = [](std::shared_ptr<utils::BaseStreamWriter> sharedOut, const std::string& config) {
                return std::make_unique<DoubleGorillaEncoder>(sharedOut, config);
            };

            // Decoder
            decoderFactoryMap[DATA_TYPE_DOUBLE] = [](const std::string& path) {
                return std::make_unique<DoubleGorillaDecoder>(path);
            };
            decoderFactoryWithConfigMap[DATA_TYPE_DOUBLE] = [](const std::string& path, const std::string& config) {
                return std::make_unique<DoubleGorillaDecoder>(path, config);
            };
            decoderFactoryWithSharedInMap[DATA_TYPE_DOUBLE] = [](std::shared_ptr<utils::BaseBlockStreamReader> sharedIn) {
                return std::make_unique<DoubleGorillaDecoder>(std::move(sharedIn));
            };
            decoderFactoryWithSharedInAndConfigMap[DATA_TYPE_DOUBLE] = [](std::shared_ptr<utils::BaseBlockStreamReader> sharedIn, const std::string& config) {
                return std::make_unique<DoubleGorillaDecoder>(std::move(sharedIn), config);
            };
        }
    };

}
}
