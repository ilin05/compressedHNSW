#include "double_gorilla_encoder.h"
#include <algorithm>

namespace encoding_algorithm {
namespace gorilla {

    DoubleGorillaEncoder::DoubleGorillaEncoder(const std::string& outputPath) : Encoder(outputPath) {}
    DoubleGorillaEncoder::DoubleGorillaEncoder(const std::string& outputPath, const std::string& config) : Encoder(outputPath, config) {}

    std::unique_ptr<Encoder> DoubleGorillaEncoder::deepCopy() {
        auto copy = std::make_unique<DoubleGorillaEncoder>(this->outputPath);
        this->copyBaseTo(copy.get());
        copy->previous_value = this->previous_value;
        copy->previous_lead = this->previous_lead;
        copy->previous_tail = this->previous_tail;
        copy->first = this->first;
        return copy;
    }

    int DoubleGorillaEncoder::encode(double value) {
        total++;
        if (first) {
            out->write(value, size);
            first = false;
        } else {
            if (value == previous_value) {
                out->write(true);
                return out->track_bits();
            }
            out->write(false);

            long long xor_val = utils::binary_tools::xor_double(value, previous_value);
            int lead = utils::binary_tools::leadZeros(xor_val, size);
            int tail = utils::binary_tools::tailZeros(xor_val, size);

            if (lead >= previous_lead && tail >= previous_tail) {
                out->write(true);
                int len = size - previous_lead - previous_tail;
                out->write(static_cast<long long>(static_cast<unsigned long long>(xor_val) >> previous_tail), len);
            } else {
                out->write(false);
                int lim_lead = std::min(lead, 31);
                out->write(lim_lead, 5);
                int len = size - lim_lead - tail;
                out->write(len - 1, 6);
                out->write(static_cast<long long>(static_cast<unsigned long long>(xor_val) >> tail), len);
            }
            previous_lead = lead;
            previous_tail = tail;
        }
        this->previous_value = value;
        return out->track_bits();
    }

}
}
