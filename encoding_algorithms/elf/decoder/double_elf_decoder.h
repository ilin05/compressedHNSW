#pragma once

#include "../../decoder.h"
#include "../elf64_utils.h"
#include "../../../examples/utils/binary_tools.h"
#include <string>
#include <memory>

namespace encoding_algorithm {
namespace elf {

    class DoubleElfDecoder : public Decoder {
    private:
        static const int size = 64;
        double previous_value = 0.0;
        int previous_lead = 0;
        int previous_tail = 0;
        bool first = true;

        double recover(double vPrime, int betaStar);
        void ElfXorDecoder();

    public:
        DoubleElfDecoder(const std::string& inputPath);
        DoubleElfDecoder(const std::string& inputPath, const std::string& config);
        DoubleElfDecoder(std::shared_ptr<utils::BlockStreamReader> sharedIn);
        DoubleElfDecoder(std::shared_ptr<utils::BlockStreamReader> sharedIn, const std::string& config);

        double decodeDouble() override;
    };

}
}
