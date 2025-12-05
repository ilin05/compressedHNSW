#pragma once

#include "../../encoder.h"
#include "../elf64_utils.h"
#include "../../../examples/utils/binary_tools.h"
#include <string>
#include <memory>
#include <cmath>

namespace encoding_algorithm {
namespace elf {

    class DoubleElfEncoder : public Encoder {
    private:
        static const int size = 64;
        long long previous_long_value = 0;
        int previous_lead = 0;
        int previous_tail = 0;
        int previous_betaStar = 0;
        bool first = true;

        long long eraser(double v);
        void ElfXor(long long vLong);

    public:
        DoubleElfEncoder(const std::string& outputPath);
        DoubleElfEncoder(const std::string& outputPath, const std::string& config);
        DoubleElfEncoder(std::shared_ptr<utils::StreamWriter> sharedOut);
        DoubleElfEncoder(std::shared_ptr<utils::StreamWriter> sharedOut, const std::string& config);

        std::unique_ptr<Encoder> deepCopy() override;
        int encode(double value) override;
    };

}
}
