#include "double_elf_encoder.h"
#include <algorithm>
#include <limits>

namespace encoding_algorithm {
namespace elf {

    DoubleElfEncoder::DoubleElfEncoder(const std::string& outputPath) : Encoder(outputPath) {}

    DoubleElfEncoder::DoubleElfEncoder(const std::string& outputPath, const std::string& config)
        : Encoder(outputPath, config) {}

    std::unique_ptr<Encoder> DoubleElfEncoder::deepCopy() {
        auto copy = std::make_unique<DoubleElfEncoder>(this->outputPath);
        this->copyBaseTo(copy.get());
        copy->previous_long_value = this->previous_long_value;
        copy->previous_lead = this->previous_lead;
        copy->previous_tail = this->previous_tail;
        copy->previous_betaStar = this->previous_betaStar;
        copy->first = this->first;
        return copy;
    }

    long long DoubleElfEncoder::eraser(double v) {
        union {
            double d;
            unsigned long long ull;
        } converter{};
        converter.d = v;

        if (v == 0.0 || std::isinf(v)) {
            out->write(false);
            return static_cast<long long>(converter.ull);
        }

        auto alphaBeta = Elf64Utils::getAlphaAndBetaStar(v, previous_betaStar);
        int exponent = static_cast<int>((converter.ull >> 52) & 0x7FFULL);
        int gAlpha = Elf64Utils::getFAlpha(alphaBeta.first) + exponent - 1023;
        int eraseBits = 52 - gAlpha;
        int betaStar = alphaBeta.second;

        unsigned long long mask;
        if (eraseBits <= 0) {
            mask = std::numeric_limits<unsigned long long>::max();
        } else {
            unsigned int shift = static_cast<unsigned int>(eraseBits) & 0x3F;
            mask = std::numeric_limits<unsigned long long>::max() << shift;
        }
        unsigned long long delta = (~mask) & converter.ull;

        if (betaStar < 16 && delta != 0 && eraseBits > 4) {
            out->write((betaStar | 0x10), 5);
            previous_betaStar = betaStar;
            converter.ull = mask & converter.ull;
        } else {
            out->write(false);
        }
        return static_cast<long long>(converter.ull);
    }

    void DoubleElfEncoder::ElfXor(long long vLong) {
        unsigned long long current = static_cast<unsigned long long>(vLong);
        unsigned long long previous = static_cast<unsigned long long>(previous_long_value);
        unsigned long long xor_val = current ^ previous;

        int lead = utils::binary_tools::leadZeros(static_cast<long long>(xor_val), size);
        int tail = utils::binary_tools::tailZeros(static_cast<long long>(xor_val), size);
        lead = std::min(lead, 7);

        bool reuse = (xor_val != 0) && (lead == previous_lead) && (tail >= previous_tail);

        if (xor_val != 0 && reuse) {
            out->write(0, 2);
            int len = size - previous_lead - previous_tail;
            unsigned long long payload = xor_val >> previous_tail;
            out->write(static_cast<long long>(payload), len);
        } else if (xor_val == 0) {
            out->write(1, 2);
        } else {
            out->write(true);
            int len = size - lead - tail;
            unsigned long long payload = xor_val >> tail;
            if (len <= 16) {
                out->write(false);
                out->write(lead, 3);
                out->write(len - 1, 4);
            } else {
                out->write(true);
                out->write(lead, 3);
                out->write(len - 1, 6);
            }
            out->write(static_cast<long long>(payload), len);
        }

        previous_lead = lead;
        previous_tail = tail;
    }

    int DoubleElfEncoder::encode(double value) {
        if (first) {
            out->write(value, size);
            union {
                double d;
                long long ll;
            } converter{};
            converter.d = value;
            previous_long_value = converter.ll;
            first = false;
        } else {
            long long vLong = eraser(value);
            ElfXor(vLong);
            previous_long_value = vLong;
        }
        return out->track_bits();
    }

}
}
