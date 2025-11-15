#pragma once

#include "../algorithm.h"
#include "encoder/double_camel_encoder.h"
#include "decoder/double_camel_decoder.h"

#include <memory>
#include <string>

namespace encoding_algorithm {
namespace camel {

    inline const std::string DATA_TYPE_DOUBLE = "Double";

    class Camel : public Algorithm {
    public:
        Camel() {
            encoderFactoryMap[DATA_TYPE_DOUBLE] = [](const std::string& path) {
                return std::make_unique<DoubleCamelEncoder>(path);
            };
            encoderFactoryWithConfigMap[DATA_TYPE_DOUBLE] = [](const std::string& path, const std::string& config) {
                return std::make_unique<DoubleCamelEncoder>(path, config);
            };

            decoderFactoryMap[DATA_TYPE_DOUBLE] = [](const std::string& path) {
                return std::make_unique<DoubleCamelDecoder>(path);
            };
            decoderFactoryWithConfigMap[DATA_TYPE_DOUBLE] = [](const std::string& path, const std::string& config) {
                return std::make_unique<DoubleCamelDecoder>(path, config);
            };
        }
    };

}
}
