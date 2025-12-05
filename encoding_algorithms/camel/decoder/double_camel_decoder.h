#pragma once

#include "../../decoder.h"
#include "../camel_tools.h"
#include "../../../examples/utils/binary_tools.h"

#include <string>
#include <memory>

namespace encoding_algorithm {
namespace camel {

    class DoubleCamelDecoder : public Decoder {
    private:
        static const int size = 64;
        long long previous_integer = 0;
        bool first = true;

        long long integer_decode();
        double decimal_decode();

    public:
        explicit DoubleCamelDecoder(const std::string& inputPath);
        DoubleCamelDecoder(const std::string& inputPath, const std::string& config);
        DoubleCamelDecoder(std::shared_ptr<utils::BlockStreamReader> sharedIn);
        DoubleCamelDecoder(std::shared_ptr<utils::BlockStreamReader> sharedIn, const std::string& config);

        double decodeDouble() override;
    };

}
}
