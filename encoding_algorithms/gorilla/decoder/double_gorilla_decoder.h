#pragma once

#include "../../decoder.h"
#include "../../../examples/utils/binary_tools.h"
#include <string>

namespace encoding_algorithm {
namespace gorilla {

    class DoubleGorillaDecoder : public Decoder {
    private:
        static const int size = 64;
        double previous_value = 0;
        int previous_lead = 0;
        int previous_tail = 0;
        bool first = true;

    public:
        DoubleGorillaDecoder(const std::string& inputPath);
        DoubleGorillaDecoder(const std::string& inputPath, const std::string& config);

        double decodeDouble() override;
    };

}
}
