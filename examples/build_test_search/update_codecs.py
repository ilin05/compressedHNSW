import sys

new_h = '''#pragma once

#include <string>
#include <algorithm>
#include <cmath>
#include <limits>
#include <vector>
#include "../examples/utils/memory_stream_writer.h"
#include "../examples/utils/memory_block_stream_reader.h"
#include "../examples/utils/binary_tools.h"
#include "../encoding_algorithms/dexor/dexor_tools.h"
#include "../encoding_algorithms/camel/camel_tools.h"
#include "../encoding_algorithms/elf/elf64_utils.h"

namespace hnswlib {
namespace codecs {

enum class AlgorithmType { DeXOR, Gorilla, Elf, Camel };

inline AlgorithmType getAlgorithmType(const std::string& name) {
    if (name == "Gorilla") return AlgorithmType::Gorilla;
    if (name == "Elf") return AlgorithmType::Elf;
    if (name == "Camel") return AlgorithmType::Camel;
    return AlgorithmType::DeXOR; // default
}

// ---------------- DeXOR Policy ----------------
struct DeXORState {
    double current_value = 0.0;
    double previous_value = 0.0;
    int previous_q = 0;
    int previous_delta = 0;
    long long previous_exp = 1023;
    int EL = 1;
    int contract_step = 0;
    int rho = 1;

    void reset() {
        current_value = 0.0;
        previous_value = 0.0;
        previous_q = 0;
        previous_delta = 0;
        previous_exp = 1023;
        EL = 1;
        contract_step = 0;
        rho = 1;
    }
};

struct DeXORCodec {
    using StateType = DeXORState;
    static std::string name() { return "DeXOR"; }

    static inline void dexor_exception_handle(double value, StateType& state, utils::MemoryStreamWriter& writer) {
        using namespace encoding_algorithm::dexor;
        union { double d; long long l; } u;
        u.d = value;
        long long exp = DeXORTools::segment(u.l, 2, 12);
        long long delta = exp - state.previous_exp;
        int bias = DeXORTools::getP2(state.EL - 1) - 1;

        if (delta >= -bias && delta <= bias) {
            writer.write(delta + bias, state.EL);
            writer.write(u.l < 0);
            writer.write(u.l, 52);

            if (state.EL > 1) {
                int su_bias = DeXORTools::getP2(state.EL - 2) - 1;
                if(delta >= -su_bias && delta <= su_bias) {
                    state.contract_step++;
                } else {
                    state.contract_step = 0;
                }
                if (state.contract_step == state.rho) {
                    state.EL--;
                    state.contract_step = 0;
                }
            }
        } else {
            writer.write(DeXORTools::getP2(state.EL) - 1, state.EL);
            writer.write(u.l, 64);
            state.contract_step = 0;

            if (state.EL < 10) {
                state.EL++;
            }
        }
        state.previous_exp = exp;
    }

    static inline void encode(double value, StateType& state, utils::MemoryStreamWriter& writer) {
        state.current_value = value;
        using namespace encoding_algorithm::dexor;
        int q = DeXORTools::getEnd(value, state.previous_q);
        int delta = 0;
        double alpha = 0;
        while (delta < 16) {
            double pow = DeXORTools::getP10(q + delta);
            long long a = DeXORTools::truncate(value / pow);
            long long b = DeXORTools::truncate(state.previous_value / pow);
            if (a == b) {
                alpha = a * pow;
                break;
            }
            delta++;
        }

        double pow = DeXORTools::getP10(q);
        double residual = value - alpha;
        long long beta = std::llround(residual / pow);

        if (delta >= 16 || DeXORTools::comp(alpha + beta * pow, value, pow) != 0) {
            writer.write(true);
            writer.write(true);
            dexor_exception_handle(value, state, writer);
            return;
        }

        beta = std::llabs(beta);
        bool flag = q == state.previous_q;
        if (flag && delta == state.previous_delta) {
            writer.write(true);
            writer.write(false);
        } else {
            writer.write(false);
            writer.write(flag);
            if (!flag) {
                writer.write(q + 20, 5);
                state.previous_q = q;
            }
            writer.write(delta, 4);
            state.previous_delta = delta;
        }

        if (DeXORTools::comp(alpha, 0) == 0) {
            writer.write(value > 0);
        }

        writer.write(beta, DeXORTools::decimalBits(delta));
        state.previous_value = value;
    }

    static inline double dexor_exception_decode(StateType& state, utils::MemoryBlockStreamReader& reader) {
        using namespace encoding_algorithm::dexor;
        int bias = DeXORTools::getP2(state.EL - 1) - 1;
        long long delta = reader.readLong(state.EL) - bias;
        uint64_t lv = 0;

        if (delta >= -bias && delta <= bias) {
            state.previous_exp += delta;
            uint64_t sign = reader.readLong(1);
            uint64_t mantissa = reader.readLong(52);
            lv = (sign << 63) | ((uint64_t)state.previous_exp << 52) | mantissa;

            if (state.EL > 1) {
                int su_bias = DeXORTools::getP2(state.EL - 2) - 1;
                if (delta >= -su_bias && delta <= su_bias) {
                    state.contract_step++;
                } else {
                    state.contract_step = 0;
                }
                if (state.contract_step == state.rho) {
                    state.EL--;
                    state.contract_step = 0;
                }
            }
        } else {
            lv = reader.readLong(64);
            state.previous_exp = DeXORTools::segment((long long)lv, 2, 12);
            if (state.EL < 10) {
                state.EL++;
                state.contract_step = 0;
            }
        }
        union { uint64_t bits; double value; } converter{};
        converter.bits = lv;
        return converter.value;
    }

    static inline double decode(StateType& state, utils::MemoryBlockStreamReader& reader) {
        using namespace encoding_algorithm::dexor;
        int con = reader.readInt(2);
        if (con == 3) {
            double v = dexor_exception_decode(state, reader);
            state.previous_value = v;
            state.current_value = v;
            return v;
        }

        if (con == 0 || con == 1) {
            if (con == 0) {
                state.previous_q = reader.readInt(5) - 20;
            }
            state.previous_delta = reader.readInt(4);
        }

        double pow = DeXORTools::getP10(state.previous_q + state.previous_delta);
        double truncated = DeXORTools::truncate(state.previous_value / pow);
        double previous_alpha = truncated * pow;

        long long sign = previous_alpha > 0 ? 1LL : -1LL;
        if (DeXORTools::comp(previous_alpha, 0) == 0) {
            sign = reader.readBoolean() ? 1LL : -1LL;
        }

        int bits = DeXORTools::decimalBits(state.previous_delta);
        long long magnitude = bits > 0 ? reader.readLong(bits) : 0;
        double beta = (sign * magnitude) * DeXORTools::getP10(state.previous_q);

        state.previous_value = previous_alpha + beta;
        state.current_value = state.previous_value;
        return state.previous_value;
    }
};

// ---------------- Gorilla Policy ----------------
struct GorillaState {
    double current_value = 0.0;
    double previous_value = 0.0;
    int previous_lead = 0;
    int previous_tail = 0;
    bool first = true;

    void reset() {
        current_value = 0.0;
        previous_value = 0.0;
        previous_lead = 0;
        previous_tail = 0;
        first = true;
    }
};

struct GorillaCodec {
    using StateType = GorillaState;
    static std::string name() { return "Gorilla"; }

    static inline void encode(double value, StateType& state, utils::MemoryStreamWriter& writer) {
        state.current_value = value;
        union { double d; long long l; } u_val, u_prev;
        u_val.d = value;
        u_prev.d = state.previous_value;

        if (state.first) {
            writer.write(u_val.l, 64);
            state.first = false;
            state.previous_value = value;
            return;
        }

        long long xor_val = u_val.l ^ u_prev.l;
        if (xor_val == 0) {
            writer.write(false);
            return;
        }
        writer.write(true);

        int lead = __builtin_clzll(xor_val);
        int tail = __builtin_ctzll(xor_val);

        if (lead >= state.previous_lead && tail >= state.previous_tail) {
            writer.write(false);
            writer.write(xor_val >> state.previous_tail, 64 - state.previous_lead - state.previous_tail);
        } else {
            writer.write(true);
            writer.write(lead, 6);
            writer.write(64 - lead - tail, 6);
            writer.write(xor_val >> tail, 64 - lead - tail);
            state.previous_lead = lead;
            state.previous_tail = tail;
        }
        state.previous_value = value;
    }

    static inline double decode(StateType& state, utils::MemoryBlockStreamReader& reader) {
        if (state.first) {
            union { long long l; double d; } u;
            u.l = reader.readLong(64);
            state.previous_value = u.d;
            state.first = false;
            state.current_value = u.d;
            return u.d;
        }

        if (!reader.readBoolean()) {
            state.current_value = state.previous_value;
            return state.previous_value;
        }

        if (reader.readBoolean()) {
            state.previous_lead = reader.readInt(6);
            int len = reader.readInt(6);
            if (len == 0) len = 64;
            state.previous_tail = 64 - state.previous_lead - len;
        }

        int length = 64 - state.previous_lead - state.previous_tail;
        long long xor_val = reader.readLong(length);
        xor_val <<= state.previous_tail;

        union { double d; long long l; } u;
        u.d = state.previous_value;
        u.l ^= xor_val;
        state.previous_value = u.d;
        state.current_value = u.d;
        return u.d;
    }
};

// ---------------- Elf Policy ----------------
struct ElfState {
    double current_value = 0.0;
    long long previous_long_value = 0;
    int previous_lead = 0;
    int previous_tail = 0;
    int previous_betaStar = 0;
    bool first = true;

    void reset() {
        current_value = 0.0;
        previous_long_value = 0;
        previous_lead = 0;
        previous_tail = 0;
        previous_betaStar = 0;
        first = true;
    }
};

struct ElfCodec {
    using StateType = ElfState;
    static std::string name() { return "Elf"; }

    static inline void encode(double value, StateType& state, utils::MemoryStreamWriter& writer) {
        using namespace encoding_algorithm::elf;
        state.current_value = value;
        if (state.first) {
            union { double d; long long l; } u; u.d = value;
            writer.write(u.l, 64);
            state.previous_long_value = u.l;
            state.first = false;
            return;
        }

        Elf64_Tools::Elf64_Data data = Elf64_Tools::getElf64Data(value);
        int alphaStar = data.alphaStar;
        int betaStar = data.betaStar;
        if (betaStar == state.previous_betaStar) {
            writer.write(false);
            long long xor_val = alphaStar ^ state.previous_long_value;
            if (xor_val == 0) {
                writer.write(false);
            } else {
                writer.write(true);
                int lead = __builtin_clzll(xor_val);
                int tail = __builtin_ctzll(xor_val);
                if (lead >= state.previous_lead && tail >= state.previous_tail) {
                    writer.write(false);
                    writer.write(xor_val >> state.previous_tail, 64 - state.previous_lead - state.previous_tail);
                } else {
                    writer.write(true);
                    writer.write(lead, 6);
                    writer.write(64 - lead - tail, 6);
                    writer.write(xor_val >> tail, 64 - lead - tail);
                    state.previous_lead = lead;
                    state.previous_tail = tail;
                }
            }
            state.previous_long_value = alphaStar;
        } else {
            writer.write(true);
            writer.write(betaStar, 4);
            long long xor_val = alphaStar ^ state.previous_long_value;
            if (xor_val == 0) {
                writer.write(false);
            } else {
                writer.write(true);
                int lead = __builtin_clzll(xor_val);
                int tail = __builtin_ctzll(xor_val);
                if (lead >= state.previous_lead && tail >= state.previous_tail) {
                    writer.write(false);
                    writer.write(xor_val >> state.previous_tail, 64 - state.previous_lead - state.previous_tail);
                } else {
                    writer.write(true);
                    writer.write(lead, 6);
                    writer.write(64 - lead - tail, 6);
                    writer.write(xor_val >> tail, 64 - lead - tail);
                    state.previous_lead = lead;
                    state.previous_tail = tail;
                }
            }
            state.previous_betaStar = betaStar;
            state.previous_long_value = alphaStar;
        }
    }

    static inline double decode(StateType& state, utils::MemoryBlockStreamReader& reader) {
        using namespace encoding_algorithm::elf;
        if (state.first) {
            state.previous_long_value = reader.readLong(64);
            state.first = false;
            union { long long l; double d; } u; u.l = state.previous_long_value;
            state.current_value = u.d;
            return u.d;
        }

        if (reader.readBoolean()) {
            state.previous_betaStar = reader.readInt(4);
        }

        if (reader.readBoolean()) {
            if (reader.readBoolean()) {
                state.previous_lead = reader.readInt(6);
                int len = reader.readInt(6);
                if (len == 0) len = 64;
                state.previous_tail = 64 - state.previous_lead - len;
            }
            int length = 64 - state.previous_lead - state.previous_tail;
            long long xor_val = reader.readLong(length);
            xor_val <<= state.previous_tail;
            state.previous_long_value ^= xor_val;
        }
        
        double v = Elf64_Tools::getValues(state.previous_long_value, state.previous_betaStar);
        state.current_value = v;
        return v;
    }
};

// ---------------- Camel Policy ----------------
struct CamelState {
    double current_value = 0.0;
    long long previous_integer = 0;
    bool first = true;

    void reset() {
        current_value = 0.0;
        previous_integer = 0;
        first = true;
    }
};

struct CamelCodec {
    using StateType = CamelState;
    static std::string name() { return "Camel"; }

    static inline void encode(double value, StateType& state, utils::MemoryStreamWriter& writer) {
        using namespace encoding_algorithm::camel;
        state.current_value = value;
        union { double d; long long l; } cur; cur.d = value;
        if (state.first) {
            writer.write(cur.l, 64);
            state.previous_integer = cur.l;
            state.first = false;
            return;
        }

        long long current_integer = cur.l;
        long long delta = current_integer - state.previous_integer;
        
        if (delta == 0) {
            writer.write(0, 2);
        } else if (delta >= -128 && delta <= 127) {
            writer.write(1, 2);
            writer.write(delta + 128, 8);
        } else if (delta >= -32768 && delta <= 32767) {
            writer.write(2, 2);
            writer.write(delta + 32768, 16);
        } else {
            writer.write(3, 2);
            writer.write(current_integer, 64);
        }
        state.previous_integer = current_integer;
    }

    static inline double decode(StateType& state, utils::MemoryBlockStreamReader& reader) {
        using namespace encoding_algorithm::camel;
        if (state.first) {
            state.previous_integer = reader.readLong(64);
            state.first = false;
            union { long long l; double d; } u; u.l = state.previous_integer;
            state.current_value = u.d;
            return u.d;
        }

        int flag = reader.readInt(2);
        if (flag == 0) {
            // unchanged
        } else if (flag == 1) {
            long long delta = reader.readLong(8) - 128;
            state.previous_integer += delta;
        } else if (flag == 2) {
            long long delta = reader.readLong(16) - 32768;
            state.previous_integer += delta;
        } else {
            state.previous_integer = reader.readLong(64);
        }

        union { long long l; double d; } u; u.l = state.previous_integer;
        state.current_value = u.d;
        return u.d;
    }
};

} // namespace codecs
} // namespace hnswlib
'''

with open(r'd:\ZJU\SuDIS\hnswlib_cpp_py\hnswlib\compressed_codecs.h', 'w', encoding='utf-8') as f:
    f.write(new_h)
