#pragma once

#include <cstdint>
#include <algorithm>

namespace utils {
    class MemoryStreamReader {
    private:
        const unsigned char* data_source;
        size_t pointer = 0;
        int leftBits = 8;

    public:
        MemoryStreamReader(const unsigned char* data) : data_source(data) {
            init();
        }

        void init() {
            pointer = 0;
            leftBits = 8;
        }

        long long readLong(int size) {
            long long result = 0;
            while (size > 0) {
                int bitsToRead = std::min(leftBits, size);
                leftBits -= bitsToRead;
                unsigned int mask = (1u << bitsToRead) - 1u;
                unsigned int chunk = (data_source[pointer] >> leftBits) & mask;
                result = (result << bitsToRead) | static_cast<long long>(chunk);
                size -= bitsToRead;
                
                if (leftBits == 0) {
                    pointer++;
                    leftBits = 8;
                }
            }
            return result;
        }

        int readInt(int size) {
            return static_cast<int>(readLong(size));
        }
    };
}
