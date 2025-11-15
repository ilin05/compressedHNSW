#pragma once

#include "../algorithm.h"
#include "encoder/double_dexor_encoder.h"
#include "decoder/double_dexor_decoder.h"
#include <string>

namespace encoding_algorithm {
namespace dexor {

    // Assuming DataTypeEnums.DOUBLE.getType() returns "Double"
    const std::string DATA_TYPE_DOUBLE = "Double";

    class DeXOR : public Algorithm {
    public:
        DeXOR() {
            // Encoder
            encoderFactoryMap[DATA_TYPE_DOUBLE] = [](const std::string& path) {
                return std::make_unique<DoubleDeXOREncoder>(path);
            };
            encoderFactoryWithConfigMap[DATA_TYPE_DOUBLE] = [](const std::string& path, const std::string& config) {
                return std::make_unique<DoubleDeXOREncoder>(path, config);
            };

            // Decoder
            decoderFactoryMap[DATA_TYPE_DOUBLE] = [](const std::string& path) {
                return std::make_unique<DoubleDeXORDecoder>(path);
            };
            decoderFactoryWithConfigMap[DATA_TYPE_DOUBLE] = [](const std::string& path, const std::string& config) {
                return std::make_unique<DoubleDeXORDecoder>(path, config);
            };
        }
    };

}
}
