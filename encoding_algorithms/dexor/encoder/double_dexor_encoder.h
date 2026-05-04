#pragma once

#include "../../encoder.h"
#include "../dexor_tools.h"
#include <string>
#include <vector>
#include <cmath>

namespace encoding_algorithm {
namespace dexor {

    class DoubleDeXOREncoder : public Encoder {
    protected:
        static const int size = 64; // bits for double
        double previous_value = 0;
        int previous_q = 0;
        int previous_delta = 0;
        long long previous_exp = 1023;
        int EL = 1;
        int contract_step = 0;
        bool skip = false;

        // --- Nested Method classes ---
        class Method {
        public:
            DoubleDeXOREncoder* encoder;
            Method(DoubleDeXOREncoder* enc) : encoder(enc) {}
            virtual ~Method() = default;
            virtual int encode(double value) = 0;
        protected:
            virtual void Decimal_XOR(double value);
        };

        class Native : public Method {
        public:
            Native(DoubleDeXOREncoder* enc) : Method(enc) {}
            int encode(double value) override;
        };

        class Buffered : public Method {
        public:
            int total = 0;
            Buffered(DoubleDeXOREncoder* enc) : Method(enc) {}
            int encode(double value) override;
        protected:
            void Decimal_XOR(double value) override;
        };

        class Skippable : public Method {
        public:
            int exception_times = 0;
            Skippable(DoubleDeXOREncoder* enc) : Method(enc) {}
            int encode(double value) override;
        protected:
            void Decimal_XOR(double value);
        };
        // --- End of Nested Method classes ---

        std::unique_ptr<Method> method;

        // from config
        int buffer_bits = 0;
        std::vector<double> buffer;
        int rho = 8;
        int skip_available = -1;

        void ExceptionHandle(double value);

    public:
        DoubleDeXOREncoder(const std::string& outputPath);
        DoubleDeXOREncoder(const std::string& outputPath, const std::string& config);
        DoubleDeXOREncoder(std::shared_ptr<utils::BaseStreamWriter> sharedOut);
        DoubleDeXOREncoder(std::shared_ptr<utils::BaseStreamWriter> sharedOut, const std::string& config);

        std::unique_ptr<Encoder> deepCopy() override;

        int encode(double value) override;
    };

}
}
