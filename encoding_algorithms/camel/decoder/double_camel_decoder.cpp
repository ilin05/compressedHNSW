#include "double_camel_decoder.h"

#include <cmath>

namespace encoding_algorithm {
namespace camel {

    DoubleCamelDecoder::DoubleCamelDecoder(const std::string& inputPath)
        : Decoder(inputPath) {}

    DoubleCamelDecoder::DoubleCamelDecoder(const std::string& inputPath, const std::string& config)
        : Decoder(inputPath, config) {}

    // 用BlockStreamReader构造Decoder
    DoubleCamelDecoder::DoubleCamelDecoder(std::shared_ptr<utils::BaseBlockStreamReader> sharedIn)
        : Decoder(std::move(sharedIn)) {}

    DoubleCamelDecoder::DoubleCamelDecoder(std::shared_ptr<utils::BaseBlockStreamReader> sharedIn, const std::string& config)
        : Decoder(std::move(sharedIn), config) {}

    long long DoubleCamelDecoder::integer_decode() {
        long long diff = blockIn->readLong(2);
        if (diff <= 2) {
            return previous_integer + diff - 1;
        }
        long long sign = blockIn->readBoolean() ? 1 : -1;
        bool gt8 = blockIn->readBoolean();
        long long magnitude = blockIn->readLong(gt8 ? 16 : 3);
        return previous_integer + sign * magnitude;
    }

    double DoubleCamelDecoder::decimal_decode() {
        int l = blockIn->readInt(2) + 1;
        long long ldxor = 0;
        bool c1 = blockIn->readBoolean();
        unsigned long long vd = 0;
        if (c1 && l > 0) {
            unsigned long long center = static_cast<unsigned long long>(blockIn->readLong(l));
            vd = center << (52 - l);
        }

        if (l == 1) {
            ldxor = blockIn->readLong(3);
        } else if (l == 2) {
            bool gt8 = blockIn->readBoolean();
            ldxor = blockIn->readLong(gt8 ? 5 : 3);
        } else if (l == 3) {
            const int cost_bits[4] = {1, 3, 5, -l + CamelTools::calculate_max(l)};
            int code = blockIn->readInt(2);
            ldxor = blockIn->readLong(cost_bits[code]);
        } else if (l == 4) {
            const int cost_bits[4] = {4, 6, 8, -l + CamelTools::calculate_max(l)};
            int code = blockIn->readInt(2);
            ldxor = blockIn->readLong(cost_bits[code]);
        }

        double dxor = static_cast<double>(ldxor) / CamelTools::quick_pow10(l);

        if (c1) {
            dxor = utils::binary_tools::xor_double(static_cast<long long>(vd), 1.0 + dxor) - 1.0;
        }
        return dxor;
    }

    double DoubleCamelDecoder::decodeDouble() {
        double value = 0.0;
        if (first) {
            value = blockIn->readDouble(size);
            previous_integer = CamelTools::truncate(value);
            first = false;
        } else {
            previous_integer = integer_decode();
            value = static_cast<double>(previous_integer) + decimal_decode();
        }
        return value;
    }

}
}
