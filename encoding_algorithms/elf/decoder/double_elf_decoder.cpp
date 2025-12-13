#include "double_elf_decoder.h"
#include <algorithm>
#include <cmath>

namespace encoding_algorithm {
namespace elf {

    DoubleElfDecoder::DoubleElfDecoder(const std::string& inputPath) : Decoder(inputPath) {}

    DoubleElfDecoder::DoubleElfDecoder(const std::string& inputPath, const std::string& config)
        : Decoder(inputPath, config) {}

    DoubleElfDecoder::DoubleElfDecoder(std::shared_ptr<utils::BaseBlockStreamReader> sharedIn)
        : Decoder(std::move(sharedIn)) {}

    DoubleElfDecoder::DoubleElfDecoder(std::shared_ptr<utils::BaseBlockStreamReader> sharedIn, const std::string& config)
        : Decoder(std::move(sharedIn), config) {}

    double DoubleElfDecoder::recover(double vPrime, int betaStar) {
        double res;
        int sp = Elf64Utils::getSP(std::fabs(vPrime));
        if (betaStar == 0) {
            res = Elf64Utils::get10iN(-sp - 1);
            if (vPrime < 0) {
                res = -res;
            }
        } else {
            int alpha = betaStar - sp - 1;
            res = Elf64Utils::roundUp(vPrime, alpha);
        }
        return res;
    }

    void DoubleElfDecoder::ElfXorDecoder() {
        int c2 = blockIn->readInt(2);
        long long xor_val = 0;
        if (c2 == 0) {
            int len = size - previous_lead - previous_tail;
            xor_val = blockIn->readLong(len) << previous_tail;
            previous_lead = utils::binary_tools::leadZeros(xor_val, size);
            previous_lead = std::min(previous_lead, 7);
            previous_tail = utils::binary_tools::tailZeros(xor_val, size);
        } else if (c2 == 1) {
            xor_val = 0;
            previous_lead = 7;
            previous_tail = size;
        } else {
            previous_lead = blockIn->readInt(3);
            int len = (c2 == 2) ? (blockIn->readInt(4) + 1) : (blockIn->readInt(6) + 1);
            previous_tail = size - len - previous_lead;
            xor_val = blockIn->readLong(len) << previous_tail;
        }
        previous_value = utils::binary_tools::xor_double(xor_val, previous_value);
    }

    double DoubleElfDecoder::decodeDouble() {
        if (first) {
            previous_value = blockIn->readDouble(size);
            first = false;
        } else {
            bool c1 = blockIn->readBoolean();
            int betaStar = 0;
            if (c1) {
                betaStar = blockIn->readInt(4);
            }
            ElfXorDecoder();
            if (c1) {
                return recover(previous_value, betaStar);
            }
        }
        return previous_value;
    }

}
}
