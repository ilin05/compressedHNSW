#pragma once

#include "../algorithm.h"
#include "encoder/double_elf_encoder.h"
#include "decoder/double_elf_decoder.h"
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

            decoderFactoryMap[DATA_TYPE_DOUBLE] = [](const std::string& path) {
                return std::make_unique<DoubleElfDecoder>(path);
            };
            decoderFactoryWithConfigMap[DATA_TYPE_DOUBLE] = [](const std::string& path, const std::string& config) {
                return std::make_unique<DoubleElfDecoder>(path, config);
            };
        }
    };

}
}
