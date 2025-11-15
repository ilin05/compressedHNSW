#include "double_gorilla_decoder.h"

namespace encoding_algorithm {
namespace gorilla {

    DoubleGorillaDecoder::DoubleGorillaDecoder(const std::string& inputPath) : Decoder(inputPath) {}
    DoubleGorillaDecoder::DoubleGorillaDecoder(const std::string& inputPath, const std::string& config) : Decoder(inputPath, config) {}

    double DoubleGorillaDecoder::decodeDouble() {
        if (first) {
            previous_value = in->readDouble(size);
            first = false;
        } else {
            bool c1 = in->readBoolean();
            if (c1) {
                return previous_value;
            }

            bool c2 = in->readBoolean();
            long long xor_val = 0;
            if (c2) {
                int len = size - previous_lead - previous_tail;
                xor_val = in->readLong(len) << previous_tail;
            } else {
                int lim_lead = in->readInt(5);
                int len = in->readInt(6) + 1;
                int tail = size - len - lim_lead;
                xor_val = in->readLong(len) << tail;
            }
            previous_lead = utils::binary_tools::leadZeros(xor_val, size);
            previous_tail = utils::binary_tools::tailZeros(xor_val, size);
            previous_value = utils::binary_tools::xor_double(xor_val, previous_value);
        }
        return previous_value;
    }

}
}
