#pragma once

#include "../../encoder.h"
#include "../camel_tools.h"
#include "../../../examples/utils/binary_tools.h"

#include <string>
#include <memory>

namespace encoding_algorithm {
namespace camel {

    class DoubleCamelEncoder : public Encoder {
    private:
        static const int size = 64;
        long long previous_integer = 0;
        bool first = true;

        void integer_encode(long long integer);
        void decimal_compression(double value, double dec);
        void Camel(double value, long long integer);

    public:
        explicit DoubleCamelEncoder(const std::string& outputPath);
        DoubleCamelEncoder(const std::string& outputPath, const std::string& config);

        std::unique_ptr<Encoder> deepCopy() override;
        int encode(double value) override;
    };

}
}
