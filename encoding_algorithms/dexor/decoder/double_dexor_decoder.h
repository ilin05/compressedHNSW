#pragma once

#include "../../decoder.h"
#include "../dexor_tools.h"
#include <string>
#include <vector>
#include <cmath>
#include <memory>
#include <cstddef>

namespace encoding_algorithm {
namespace dexor {

    class DoubleDeXORDecoder : public Decoder {
    protected:
        static const int size = 64;
        double previous_value = 0;
        int previous_q = 0;
        int previous_delta = 0;
        long long previous_exp = 1023;
        int EL = 1;
        int contract_step = 0;
        double previous_alpha = 0;
        bool skip = false;

        // --- Nested Method classes ---
        class Method {
        public:
            DoubleDeXORDecoder* decoder;
            Method(DoubleDeXORDecoder* dec) : decoder(dec) {}
            virtual ~Method() = default;
            virtual double decodeDouble() = 0;
        };

        class Native : public Method {
        public:
            Native(DoubleDeXORDecoder* dec) : Method(dec) {}
            double decodeDouble() override;
        };

        class Buffered : public Method {
        public:
            size_t total = 0;
            Buffered(DoubleDeXORDecoder* dec) : Method(dec) {}
            double decodeDouble() override;
        };

        class Skippable : public Method {
        public:
            int exception_times = 0;
            Skippable(DoubleDeXORDecoder* dec) : Method(dec) {}
            double decodeDouble() override;
        };
        // --- End of Nested Method classes ---

        std::unique_ptr<Method> method;

        // from config
        int buffer_bits = 0;
        std::vector<double> buffer;
        int rho = 8;
        int skip_available = -1;

        double ExceptionDecode();
        void initializeMethod();

    public:
        DoubleDeXORDecoder(const std::string& inputPath);
        DoubleDeXORDecoder(const std::string& inputPath, const std::string& config);
        DoubleDeXORDecoder(std::shared_ptr<utils::BlockStreamReader> sharedIn);
        DoubleDeXORDecoder(std::shared_ptr<utils::BlockStreamReader> sharedIn, const std::string& config);

        double decodeDouble() override;
    };

}
}
