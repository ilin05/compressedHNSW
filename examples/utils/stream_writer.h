#pragma once

#include <string>
#include <vector>
#include <fstream>
#include <cstdint> // For int32_t and uint64_t
#include <algorithm> // For std::min
#include <stdexcept> // For std::runtime_error

namespace utils {
    class StreamWriter {
    private:
        static const int bufferSize = 1024;
        char buffer[bufferSize];
        int pointer = 0;
        int currentByte = 0;
        int leftBits = 8;
        long long delta_bits = 0;
        std::string path;

        void writeToDisk(const char* data, int size) {
            std::ofstream outputStream(path, std::ios::binary | std::ios::app);
            if (outputStream.is_open()) {
                outputStream.write(data, size);
            }
        }

        void saveByte() {
            if (leftBits == 8) return;
            currentByte <<= leftBits; // padding with 0
            buffer[pointer++] = static_cast<char>(currentByte & 0xFF);
            leftBits = 8;
            currentByte = 0;
            if (pointer >= bufferSize) {
                writeToDisk(buffer, bufferSize);
                pointer = 0;
            }
        }

        void init() {
            leftBits = 8;
            pointer = 0;
            currentByte = 0;
            delta_bits = 0;
        }

    public:
        explicit StreamWriter(const std::string& path) {
            this->path = path;
            init();
        }

        ~StreamWriter() {
            clear();
        }

        long long track_bits() {
            long long b = delta_bits;
            this->delta_bits = 0;
            return b;
        }

        void clear() {
            if (pointer == 0 && currentByte == 0 && leftBits == 8) return;
            saveByte();
            if (pointer > 0) {
                writeToDisk(buffer, pointer);
            }
            init();
        }

        void write(bool b) {
            delta_bits += 1;
            leftBits--;
            currentByte = (currentByte << 1) | (b ? 1 : 0);
            if (leftBits == 0) {
                saveByte();
            }
        }

        void write(long long value, int size) {
            delta_bits += std::max(0, size);
            while (size > 0) {
                int len = std::min(size, leftBits);
                int mask = (1 << len) - 1;
                currentByte <<= len;
                currentByte |= (int)((value >> (size - len)) & mask);
                leftBits -= len;
                if (leftBits == 0) {
                    saveByte();
                }
                size -= len;
            }
        }

        void write(int value, int size) {
            write(static_cast<long long>(value), size);
        }

        void write(float value, int size) {
            union {
                float f;
                int32_t i;
            } u;
            u.f = value;
            write(static_cast<long long>(u.i), size);
        }

        void write(double value, int size) {
            union {
                double d;
                uint64_t l;
            } u;
            u.d = value;
            write(static_cast<long long>(u.l), size);
        }
    };
}
