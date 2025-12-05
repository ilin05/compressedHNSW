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
            encoderFactoryWithSharedOutMap[DATA_TYPE_DOUBLE] = [](std::shared_ptr<utils::StreamWriter> sharedOut) {
                return std::make_unique<DoubleCamelEncoder>(sharedOut);
            };
            encoderFactoryWithSharedOutAndConfigMap[DATA_TYPE_DOUBLE] = [](std::shared_ptr<utils::StreamWriter> sharedOut, const std::string& config) {
                return std::make_unique<DoubleCamelEncoder>(sharedOut, config);
            };

            decoderFactoryMap[DATA_TYPE_DOUBLE] = [](const std::string& path) {
                return std::make_unique<DoubleCamelDecoder>(path);
            };
            decoderFactoryWithConfigMap[DATA_TYPE_DOUBLE] = [](const std::string& path, const std::string& config) {
                return std::make_unique<DoubleCamelDecoder>(path, config);
            };
            decoderFactoryWithSharedInMap[DATA_TYPE_DOUBLE] = [](std::shared_ptr<utils::BlockStreamReader> sharedIn) {
                return std::make_unique<DoubleCamelDecoder>(std::move(sharedIn));
            };
            decoderFactoryWithSharedInAndConfigMap[DATA_TYPE_DOUBLE] = [](std::shared_ptr<utils::BlockStreamReader> sharedIn, const std::string& config) {
                return std::make_unique<DoubleCamelDecoder>(std::move(sharedIn), config);
            };
        }
    };

}
}
