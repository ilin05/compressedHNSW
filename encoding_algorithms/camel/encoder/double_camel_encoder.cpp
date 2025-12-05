#include "double_camel_encoder.h"

#include <cmath>

namespace encoding_algorithm {
namespace camel {

    DoubleCamelEncoder::DoubleCamelEncoder(const std::string& outputPath)
        : Encoder(outputPath) {}

    DoubleCamelEncoder::DoubleCamelEncoder(const std::string& outputPath, const std::string& config)
        : Encoder(outputPath, config) {}
    
    DoubleCamelEncoder::DoubleCamelEncoder(std::shared_ptr<utils::StreamWriter> sharedOut)
        : Encoder(std::move(sharedOut)) {}

    DoubleCamelEncoder::DoubleCamelEncoder(std::shared_ptr<utils::StreamWriter> sharedOut, const std::string& config)
        : Encoder(std::move(sharedOut), config) {}

    std::unique_ptr<Encoder> DoubleCamelEncoder::deepCopy() {
        auto copy = std::make_unique<DoubleCamelEncoder>(this->outputPath);
        this->copyBaseTo(copy.get());
        copy->previous_integer = this->previous_integer;
        copy->first = this->first;
        return copy;
    }

    void DoubleCamelEncoder::integer_encode(long long integer) {
        long long diff = integer - previous_integer;
        if (diff >= -1 && diff <= 1) {
            out->write(static_cast<long long>(diff + 1), 2);
        } else {
            out->write(3, 2);
            bool non_negative = diff >= 0;
            out->write(non_negative);
            long long abs_diff = diff >= 0 ? diff : -diff;
            bool large = abs_diff >= 8;
            out->write(large);
            out->write(abs_diff, large ? 16 : 3);
        }
        previous_integer = integer;
    }

    void DoubleCamelEncoder::decimal_compression(double value, double dec) {
    int l = CamelTools::decimal_count(value);
    out->write(l - 1, 2);

        double dxor = dec;
        if (dec >= CamelTools::quick_pow2(-l)) {
            out->write(true);
            dxor = CamelTools::calculate_dxor(dec, l);
            long long vd_bits = utils::binary_tools::xor_double(1.0 + dec, 1.0 + dxor);
            unsigned long long shifted = static_cast<unsigned long long>(vd_bits) >> (52 - l);
            out->write(static_cast<long long>(shifted), l);
        } else {
            out->write(false);
        }

        long long ldxor = static_cast<long long>(std::llround(dxor * CamelTools::quick_pow10(l)));

        if (l == 1) {
            out->write(ldxor, 3);
        } else if (l == 2) {
            bool gt8 = ldxor >= 8;
            out->write(gt8);
            out->write(ldxor, gt8 ? 5 : 3);
        } else if (l == 3) {
            const int thresholds[3] = {2, 8, 32};
            const int cost_bits[4] = {1, 3, 5, -l + CamelTools::calculate_max(l)};

            int code = 0;
            for (int i = 0; i <= 3; ++i) {
                code = i;
                if (i < 3 && ldxor < thresholds[i]) {
                    break;
                }
            }

            out->write(code, 2);
            out->write(ldxor, cost_bits[code]);
        } else if (l == 4) {
            const int thresholds[3] = {16, 64, 256};
            const int cost_bits[4] = {4, 6, 8, -l + CamelTools::calculate_max(l)};

            int code = 0;
            for (int i = 0; i <= 3; ++i) {
                code = i;
                if (i < 3 && ldxor < thresholds[i]) {
                    break;
                }
            }

            out->write(code, 2);
            out->write(ldxor, cost_bits[code]);
        }
    }

    void DoubleCamelEncoder::Camel(double value, long long integer) {
        double dec = value - static_cast<double>(integer);
        integer_encode(integer);
        decimal_compression(value, dec);
    }

    int DoubleCamelEncoder::encode(double value) {
        long long integer = static_cast<long long>(std::floor(value));
        if (first) {
            out->write(value, size);
            first = false;
        } else {
            Camel(value, integer);
        }
        previous_integer = integer;
        return out->track_bits();
    }

}
}
