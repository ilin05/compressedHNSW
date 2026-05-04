#include "double_dexor_decoder.h"

#include <cstddef>

namespace encoding_algorithm {
namespace dexor {

    DoubleDeXORDecoder::DoubleDeXORDecoder(const std::string& inputPath)
        : Decoder(inputPath) {
        initializeMethod();
    }

    DoubleDeXORDecoder::DoubleDeXORDecoder(const std::string& inputPath, const std::string& configStr)
        : Decoder(inputPath, configStr) {
        initializeMethod();
    }

    DoubleDeXORDecoder::DoubleDeXORDecoder(std::shared_ptr<utils::BaseBlockStreamReader> sharedIn)
        : Decoder(std::move(sharedIn)) {
        initializeMethod();
    }

    DoubleDeXORDecoder::DoubleDeXORDecoder(std::shared_ptr<utils::BaseBlockStreamReader> sharedIn, const std::string& configStr)
        : Decoder(std::move(sharedIn), configStr) {
        initializeMethod();
    }

    void DoubleDeXORDecoder::initializeMethod() {
        auto parseInt = [&](const char* key, int current) -> int {
            auto it = config.find(key);
            if (it != config.end()) {
                try {
                    return std::stoi(it->second);
                } catch (...) {
                    return current;
                }
            }
            return current;
        };

        buffer_bits = parseInt("buffer_bits", buffer_bits);
        rho = parseInt("rho", rho);
        skip_available = parseInt("skip_available", skip_available);

        if (buffer_bits > 0) {
            buffer.assign(static_cast<size_t>(1) << buffer_bits, 0.0);
            method = std::make_unique<Buffered>(this);
        } else if (skip_available >= 0) {
            method = std::make_unique<Skippable>(this);
        } else {
            method = std::make_unique<Native>(this);
        }
    }

    double DoubleDeXORDecoder::decodeDouble() {
        return method ? method->decodeDouble() : 0.0;
    }

    double DoubleDeXORDecoder::ExceptionDecode() {
        const int bias = DeXORTools::getP2(EL - 1) - 1;
        const long long delta = blockIn->readLong(EL) - bias;
        uint64_t lv = 0;

        if (delta >= -bias && delta <= bias) {
            previous_exp += delta;
            const uint64_t sign = static_cast<uint64_t>(blockIn->readLong(1));
            const uint64_t mantissa = static_cast<uint64_t>(blockIn->readLong(52));
            lv = (sign << 63) | (static_cast<uint64_t>(previous_exp) << 52) | mantissa;

            if (EL > 1) {
                const int su_bias = DeXORTools::getP2(EL - 2) - 1;
                if (delta >= -su_bias && delta <= su_bias) {
                    contract_step++;
                } else {
                    contract_step = 0;
                }
                if (contract_step == rho) {
                    EL--;
                    contract_step = 0;
                }
            }
        } else {
            lv = static_cast<uint64_t>(blockIn->readLong(64));
            previous_exp = DeXORTools::segment(static_cast<long long>(lv), 2, 12);

            if (EL < 10) {
                EL++;
                contract_step = 0;
            }
        }

        union {
            uint64_t bits;
            double value;
        } converter{};
        converter.bits = lv;
        return converter.value;
    }

    double DoubleDeXORDecoder::Native::decodeDouble() {
        auto* dec = decoder;
        int con = dec->blockIn->readInt(2);
        if (con == 3) {
            return dec->ExceptionDecode();
        }

        if (con == 0 || con == 1) {
            if (con == 0) {
                dec->previous_q = dec->blockIn->readInt(5) - 20;
            }
            dec->previous_delta = dec->blockIn->readInt(4);
            const double pow = DeXORTools::getP10(dec->previous_q + dec->previous_delta);
            const double truncated = static_cast<double>(DeXORTools::truncate(dec->previous_value / pow));
            dec->previous_alpha = truncated * pow;
        }

        long long sign = dec->previous_alpha > 0 ? 1LL : -1LL;
        if (DeXORTools::comp(dec->previous_alpha, 0) == 0) {
            sign = dec->blockIn->readBoolean() ? 1LL : -1LL;
        }

        const int bits = DeXORTools::decimalBits(dec->previous_delta);
        long long magnitude = bits > 0 ? dec->blockIn->readLong(bits) : 0;
        const double beta = static_cast<double>(sign * magnitude) * DeXORTools::getP10(dec->previous_q);

        dec->previous_value = dec->previous_alpha + beta;
        return dec->previous_value;
    }

    double DoubleDeXORDecoder::Buffered::decodeDouble() {
        auto* dec = decoder;
        int con = dec->blockIn->readInt(2);
        if (con == 3) {
            return dec->ExceptionDecode();
        }

        const size_t capacity = dec->buffer.size();
        if (dec->buffer_bits > 0 && capacity > 0) {
            const size_t idx = static_cast<size_t>(dec->blockIn->readInt(dec->buffer_bits)) % capacity;
            dec->previous_value = dec->buffer[idx];
        } else {
            dec->previous_value = 0.0;
        }

        if (con == 0 || con == 1) {
            if (con == 0) {
                dec->previous_q = dec->blockIn->readInt(5) - 20;
            }
            dec->previous_delta = dec->blockIn->readInt(4);
        }

        const double pow = DeXORTools::getP10(dec->previous_q + dec->previous_delta);
        const double truncated = static_cast<double>(DeXORTools::truncate(dec->previous_value / pow));
        dec->previous_alpha = truncated * pow;

        long long sign = dec->previous_alpha > 0 ? 1LL : -1LL;
        if (DeXORTools::comp(dec->previous_alpha, 0) == 0) {
            sign = dec->blockIn->readBoolean() ? 1LL : -1LL;
        }

        const int bits = DeXORTools::decimalBits(dec->previous_delta);
        long long magnitude = bits > 0 ? dec->blockIn->readLong(bits) : 0;
        const double residual = static_cast<double>(sign * magnitude) * DeXORTools::getP10(dec->previous_q);

        dec->previous_value = dec->previous_alpha + residual;

        if (capacity > 0) {
            dec->buffer[total] = dec->previous_value;
            total = (total + 1) % capacity;
        }

        return dec->previous_value;
    }

    double DoubleDeXORDecoder::Skippable::decodeDouble() {
        auto* dec = decoder;
        if (dec->skip) {
            return dec->ExceptionDecode();
        }

        int con = dec->blockIn->readInt(2);
        if (con == 3) {
            exception_times++;
            if (dec->skip_available >= 0 && exception_times >= dec->skip_available) {
                dec->skip = true;
            }
            return dec->ExceptionDecode();
        }
        exception_times = 0;

        if (con == 0 || con == 1) {
            if (con == 0) {
                dec->previous_q = dec->blockIn->readInt(5) - 20;
            }
            dec->previous_delta = dec->blockIn->readInt(4);
            const double pow = DeXORTools::getP10(dec->previous_q + dec->previous_delta);
            const double truncated = static_cast<double>(DeXORTools::truncate(dec->previous_value / pow));
            dec->previous_alpha = truncated * pow;
        }

        long long sign = dec->previous_alpha > 0 ? 1LL : -1LL;
        if (DeXORTools::comp(dec->previous_alpha, 0) == 0) {
            sign = dec->blockIn->readBoolean() ? 1LL : -1LL;
        }

        const int bits = DeXORTools::decimalBits(dec->previous_delta);
        long long magnitude = bits > 0 ? dec->blockIn->readLong(bits) : 0;
        const double beta = static_cast<double>(sign * magnitude) * DeXORTools::getP10(dec->previous_q);

        dec->previous_value = dec->previous_alpha + beta;
        return dec->previous_value;
    }

}
}
