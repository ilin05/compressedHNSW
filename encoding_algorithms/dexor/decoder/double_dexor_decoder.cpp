#include "double_dexor_decoder.h"

namespace encoding_algorithm {
namespace dexor {

    // Constructor implementations
    DoubleDeXORDecoder::DoubleDeXORDecoder(const std::string& inputPath) : Decoder(inputPath) {
        method = std::make_unique<Native>(this);
    }

    DoubleDeXORDecoder::DoubleDeXORDecoder(const std::string& inputPath, const std::string& configStr) : Decoder(inputPath, configStr) {
        auto it_buffer = config.find("buffer_bits");
        if (it_buffer != config.end()) {
            buffer_bits = std::stoi(it_buffer->second);
        }
        auto it_rho = config.find("rho");
        if (it_rho != config.end()) {
            rho = std::stoi(it_rho->second);
        }
        auto it_skip = config.find("skip_available");
        if (it_skip != config.end()) {
            skip_available = std::stoi(it_skip->second);
        }

        if (buffer_bits > 0) {
            buffer.resize(1 << buffer_bits);
            method = std::make_unique<Buffered>(this);
        } else if (skip_available >= 0) {
            method = std::make_unique<Skippable>(this);
        } else {
            method = std::make_unique<Native>(this);
        }
    }

    // Main decode method
    double DoubleDeXORDecoder::decodeDouble() {
        return method->decodeDouble();
    }

    // ExceptionDecode
    double DoubleDeXORDecoder::ExceptionDecode() {
        int bias = DeXORTools::getP2(EL - 1) - 1;
        long long delta = in->readLong(EL) - bias;
        uint64_t lv;

        if (delta >= -bias && delta <= bias) {
            previous_exp += delta;
            long long sign = in->readLong(1);
            lv = (sign << 63) | (previous_exp << 52) | in->readLong(52);

            if (EL > 1) {
                contract_step++;
                if (contract_step == rho) {
                    EL--;
                    contract_step = 0;
                }
            }
        } else {
            lv = in->readLong(64);
            previous_exp = DeXORTools::segment(lv, 2, 12);

            if (EL < 10) {
                EL++;
                contract_step = 0;
            }
        }
        union { uint64_t l; double d; } u;
        u.l = lv;
        return u.d;
    }

    // Native::decodeDouble
    double DoubleDeXORDecoder::Native::decodeDouble() {
        if (decoder->skip) {
            // Placeholder for Decimal_XOR decoding logic
            decoder->skip = false;
            decoder->previous_value = 0.0; // This needs actual implementation
        } else {
            decoder->previous_value = decoder->ExceptionDecode();
            decoder->skip = true;
        }
        return decoder->previous_value;
    }

    // Buffered::decodeDouble
    double DoubleDeXORDecoder::Buffered::decodeDouble() {
        // Placeholder for Buffered decoding logic
        return 0.0;
    }

    // Skippable::decodeDouble
    double DoubleDeXORDecoder::Skippable::decodeDouble() {
        // Placeholder for Skippable decoding logic
        return 0.0;
    }

}
}
