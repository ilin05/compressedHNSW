#pragma once

#include "../algorithm.h"
#include "encoder/double_elf_encoder.h"
#include "decoder/double_elf_decoder.h"
#include <memory>
#include <string>

namespace encoding_algorithm {
namespace elf {

    const std::string DATA_TYPE_DOUBLE = "Double";

    class Elf : public Algorithm {
    public:
        Elf() {
            encoderFactoryMap[DATA_TYPE_DOUBLE] = [](const std::string& path) {
                return std::make_unique<DoubleElfEncoder>(path);
            };
            encoderFactoryWithConfigMap[DATA_TYPE_DOUBLE] = [](const std::string& path, const std::string& config) {
                return std::make_unique<DoubleElfEncoder>(path, config);
            };
            encoderFactoryWithSharedOutMap[DATA_TYPE_DOUBLE] = [](std::shared_ptr<utils::BaseStreamWriter> sharedOut) {
                return std::make_unique<DoubleElfEncoder>(sharedOut);
            };
            encoderFactoryWithSharedOutAndConfigMap[DATA_TYPE_DOUBLE] = [](std::shared_ptr<utils::BaseStreamWriter> sharedOut, const std::string& config) {
                return std::make_unique<DoubleElfEncoder>(sharedOut, config);
            };

            decoderFactoryMap[DATA_TYPE_DOUBLE] = [](const std::string& path) {
                return std::make_unique<DoubleElfDecoder>(path);
            };
            decoderFactoryWithConfigMap[DATA_TYPE_DOUBLE] = [](const std::string& path, const std::string& config) {
                return std::make_unique<DoubleElfDecoder>(path, config);
            };
            decoderFactoryWithSharedInMap[DATA_TYPE_DOUBLE] = [](std::shared_ptr<utils::BaseBlockStreamReader> sharedIn) {
                return std::make_unique<DoubleElfDecoder>(std::move(sharedIn));
            };
            decoderFactoryWithSharedInAndConfigMap[DATA_TYPE_DOUBLE] = [](std::shared_ptr<utils::BaseBlockStreamReader> sharedIn, const std::string& config) {
                return std::make_unique<DoubleElfDecoder>(std::move(sharedIn), config);
            };
        }
    };

}
}
