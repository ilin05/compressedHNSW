#pragma once

#include "../../encoder.h"
#include "../../../examples/utils/binary_tools.h"
#include <string>
#include <memory>

namespace encoding_algorithm {
namespace gorilla {

    class DoubleGorillaEncoder : public Encoder {
    private:
        static const int size = 64;
        double previous_value = 0;
        int previous_lead = 0;
        int previous_tail = 0;
        bool first = true;
        long long total = 0;

    public:
        DoubleGorillaEncoder(const std::string& outputPath);
        DoubleGorillaEncoder(const std::string& outputPath, const std::string& config);

        std::unique_ptr<Encoder> deepCopy() override;
        int encode(double value) override;
    };

}
}
