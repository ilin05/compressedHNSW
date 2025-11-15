#pragma once

#include "../algorithm.h"
#include "encoder/double_gorilla_encoder.h"
#include "decoder/double_gorilla_decoder.h"
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

            // Decoder
            decoderFactoryMap[DATA_TYPE_DOUBLE] = [](const std::string& path) {
                return std::make_unique<DoubleGorillaDecoder>(path);
            };
            decoderFactoryWithConfigMap[DATA_TYPE_DOUBLE] = [](const std::string& path, const std::string& config) {
                return std::make_unique<DoubleGorillaDecoder>(path, config);
            };
        }
    };

}
}
