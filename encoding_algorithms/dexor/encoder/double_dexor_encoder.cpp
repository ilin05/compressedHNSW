#include "double_dexor_encoder.h"
#include <cstring>

namespace encoding_algorithm {
namespace dexor {

    // Constructor implementations
    DoubleDeXOREncoder::DoubleDeXOREncoder(const std::string& outputPath) : Encoder(outputPath) {
        method = std::make_unique<Native>(this);
    }

    DoubleDeXOREncoder::DoubleDeXOREncoder(const std::string& outputPath, const std::string& configStr) : Encoder(outputPath, configStr) {
        auto it_buffer = config.find("buffer_bits");
        if (it_buffer != config.end()) {
            buffer_bits = std::stoi(it_buffer->second);
        }
        auto it_rho = config.find("rho");
        if (it_rho != config.end()) {
            rho = std::stoi(it_rho->second);
        }
        auto it_skip = config.find("skip_available");
        if (it_skip != config.end()) {
            skip_available = std::stoi(it_skip->second);
        }

        if (buffer_bits > 0) {
            buffer.resize(1 << buffer_bits);
            method = std::make_unique<Buffered>(this);
        } else if (skip_available >= 0) {
            method = std::make_unique<Skippable>(this);
        } else {
            method = std::make_unique<Native>(this);
        }
    }

    // deepCopy implementation
    std::unique_ptr<Encoder> DoubleDeXOREncoder::deepCopy() {
        auto copy = std::make_unique<DoubleDeXOREncoder>(this->outputPath);
        this->copyBaseTo(copy.get());
        copy->previous_value = this->previous_value;
        copy->previous_q = this->previous_q;
        copy->previous_delta = this->previous_delta;
        copy->previous_exp = this->previous_exp;
        copy->EL = this->EL;
        copy->contract_step = this->contract_step;
        copy->skip = this->skip;
        copy->buffer_bits = this->buffer_bits;
        copy->buffer = this->buffer;
        copy->rho = this->rho;
        copy->skip_available = this->skip_available;

        if (dynamic_cast<Native*>(this->method.get())) {
            copy->method = std::make_unique<Native>(copy.get());
        } else if (dynamic_cast<Buffered*>(this->method.get())) {
            copy->method = std::make_unique<Buffered>(copy.get());
        } else if (dynamic_cast<Skippable*>(this->method.get())) {
            copy->method = std::make_unique<Skippable>(copy.get());
        }
        return copy;
    }

    // Main encode method
    int DoubleDeXOREncoder::encode(double value) {
        return this->method->encode(value);
    }

    // ExceptionHandle
    void DoubleDeXOREncoder::ExceptionHandle(double value) {
        union { double d; long long l; } u;
        u.d = value;
        long long exp = DeXORTools::segment(u.l, 2, 12);
        long long delta = exp - previous_exp;
        int bias = DeXORTools::getP2(EL - 1) - 1;

        if (delta >= -bias && delta <= bias) {
            out->write(delta + bias, EL);
            out->write(u.l < 0); // sign bit
            out->write(u.l, 52); // mantissa

            if (EL > 1) {
                int su_bias = DeXORTools::getP2(EL - 2) - 1;
                if(delta >= -su_bias && delta <= su_bias) {
                    contract_step++;
                }else{
                    contract_step = 0;
                }
                if (contract_step == rho) {
                    EL--;
                    contract_step = 0;
                }
            }
        } else {
            out->write(DeXORTools::getP2(EL) - 1, EL);
            out->write(u.l, 64);
            contract_step = 0;

            if (EL < 10) {
                EL++;
            }
        }
        previous_exp = exp;
    }

    // Method::Decimal_XOR
    void DoubleDeXOREncoder::Method::Decimal_XOR(double value) {
        // This is a complex method. A direct port requires careful handling of floating point logic.
        // The logic from the Java file needs to be translated here.
        // This is a placeholder for the complex logic.
        int q = DeXORTools::getEnd(value, encoder->previous_q);

        int delta = 0;
        double alpha = 0;
        while(delta < 16){
            double pow = DeXORTools::getP10(q + delta);
            long long a = DeXORTools::truncate(value / pow);
            long b = DeXORTools::truncate(encoder->previous_value / pow);
            if(a == b){
                alpha = a * pow;
                break;
            }
            delta++;
        }

        double pow = DeXORTools::getP10(q);
        double residual = value - alpha;
        long long beta = std::round(residual / pow);

        if(delta >= 16 || DeXORTools::comp(alpha + beta * pow, value, pow) != 0){
            encoder->out->write(true);
            encoder->out->write(false);
            encoder->ExceptionHandle(value);
            return;
        }

        beta = abs(beta);
        bool flag = q == encoder->previous_q;
        if(flag && delta == encoder->previous_delta){
            encoder->out->write(true);
            encoder->out->write(false);
        }else{
            encoder->out->write(false);
            encoder->out->write(flag);
            if(!flag){
                encoder->out->write(q + 20, 5);
                encoder->previous_q = q;
            }
            encoder->out->write(delta, 4);
            encoder->previous_delta = delta;
        }

        if(DeXORTools::comp(alpha, 0) == 0){
            encoder->out->write(value > 0); // sign bit
        }

        encoder->out->write(beta, DeXORTools::decimalBits(delta));
        encoder->previous_value = value;
    }

    // Native::encode
    int DoubleDeXOREncoder::Native::encode(double value) {
        Decimal_XOR(value);
        return encoder->out->track_bits();
    }

    // Buffered::Decimal_XOR
    void DoubleDeXOREncoder::Buffered::Decimal_XOR(double value) {
        int q = DeXORTools::getEnd(value, encoder->previous_q);

        int delta = 0;
        double alpha = 0;
        int id = 0;

        while (delta < 16) {
            double pow = DeXORTools::getP10(q + delta);
            long long a = DeXORTools::truncate(value / pow);
            long long b = DeXORTools::truncate(encoder->buffer[0] / pow);
            if (a == b) {
                alpha = a * pow;
                break;
            }
            delta++;
        }

        double pow = DeXORTools::getP10(q + delta - 1);
        long long a = DeXORTools::truncate(value / pow);

        for (size_t i = 1; delta > 0 && i < encoder->buffer.size(); i++) {
            long long b = DeXORTools::truncate(encoder->buffer[i] / pow);
            while (delta > 0 && a == b) {
                alpha = a * pow;
                id = i;
                delta--;
                pow = DeXORTools::getP10(q + delta - 1);
                a = DeXORTools::truncate(value / pow);
                b = DeXORTools::truncate(encoder->buffer[i] / pow);
            }
        }

        double q_pow = DeXORTools::getP10(q);
        double residual = value - alpha;
        long long beta = std::round(residual / q_pow);

        if (delta >= 16 || DeXORTools::comp(alpha + beta * q_pow, value, q_pow) != 0) {
            encoder->out->write(true);
            encoder->out->write(true);
            encoder->ExceptionHandle(value);
            return;
        }

        beta = std::abs(beta);
        bool flag = q == encoder->previous_q;
        if (flag && delta == encoder->previous_delta) {
            encoder->out->write(true);
            encoder->out->write(false);
            encoder->out->write(id, encoder->buffer_bits);
        } else {
            encoder->out->write(false);
            encoder->out->write(flag);
            encoder->out->write(id, encoder->buffer_bits);
            if (!flag) {
                encoder->out->write(q + 20, 5);
                encoder->previous_q = q;
            }
            encoder->out->write(delta, 4);
            encoder->previous_delta = delta;
        }

        if (DeXORTools::comp(alpha, 0) == 0) {
            encoder->out->write(value > 0);
        }

        encoder->out->write(beta, DeXORTools::decimalBits(delta));
        encoder->buffer[total++] = value;
        total %= encoder->buffer.size();
    }

    // Buffered::encode
    int DoubleDeXOREncoder::Buffered::encode(double value) {
        Decimal_XOR(value);
        return encoder->out->track_bits();
    }

    // Skippable::Decimal_XOR
    void DoubleDeXOREncoder::Skippable::Decimal_XOR(double value) {
        int q = DeXORTools::getEnd(value, encoder->previous_q);

        int delta = 0;
        double alpha = 0;
        while (delta < 16) {
            double pow = DeXORTools::getP10(q + delta);
            long long a = DeXORTools::truncate(value / pow);
            long long b = DeXORTools::truncate(encoder->previous_value / pow);
            if (a == b) {
                alpha = a * pow;
                break;
            }
            delta++;
        }
        double pow = DeXORTools::getP10(q);
        double residual = value - alpha;
        long long beta = std::round(residual / pow);

        if (delta >= 16 || DeXORTools::comp(alpha + beta * pow, value, pow) != 0) {
            encoder->out->write(true);
            encoder->out->write(true);
            exception_times++;
            if (exception_times >= encoder->skip_available) encoder->skip = true;
            encoder->ExceptionHandle(value);
            return;
        }

        exception_times = 0;
        beta = std::abs(beta);
        bool flag = q == encoder->previous_q;
        if (flag && delta == encoder->previous_delta) {
            encoder->out->write(true);
            encoder->out->write(false);
        } else {
            encoder->out->write(false);
            encoder->out->write(flag);
            if (!flag) {
                encoder->out->write(q + 20, 5);
                encoder->previous_q = q;
            }
            encoder->out->write(delta, 4);
            encoder->previous_delta = delta;
        }

        if (DeXORTools::comp(alpha, 0) == 0) {
            encoder->out->write(value > 0);
        }

        encoder->out->write(beta, DeXORTools::decimalBits(delta));
        encoder->previous_value = value;
    }

    // Skippable::encode
    int DoubleDeXOREncoder::Skippable::encode(double value) {
        if (encoder->skip) {
            encoder->ExceptionHandle(value);
        } else {
            Decimal_XOR(value);
        }
        return encoder->out->track_bits();
    }

}
}
