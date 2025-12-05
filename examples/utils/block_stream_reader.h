#pragma once

#include <algorithm>
#include <cstdint>
#include <cstring>
#include <fstream>
#include <string>
#include <vector>

namespace utils {

    class BlockStreamReader {
    private:
        std::string fileName;
        std::vector<unsigned char> buffer;
        size_t cachedBytes = 0;
        size_t pointer = 0;
        int leftBits = 8;

    public:
        BlockStreamReader(const std::string& path, size_t dataSize)
            : fileName(path), buffer(std::max<size_t>(dataSize, 1), 0) {}

        void init() {
            pointer = 0;
            leftBits = 8;
        }

        void cacheData(int beginBit, int endBit) {
            if (endBit <= beginBit) {
                cachedBytes = 0;
                init();
                return;
            }

            std::ifstream input(fileName, std::ios::binary);
            if (!input) {
                cachedBytes = 0;
                init();
                return;
            }

            size_t byteBegin = static_cast<size_t>(beginBit) / 8;
            size_t byteEnd = (static_cast<size_t>(endBit) + 7) / 8; // endBit exclusive
            size_t bytesToRead = (byteEnd > byteBegin) ? (byteEnd - byteBegin) : 0;

            if (bytesToRead == 0) {
                cachedBytes = 0;
                init();
                return;
            }

            bytesToRead = std::min(bytesToRead, buffer.size());

            input.seekg(static_cast<std::streamoff>(byteBegin), std::ios::beg);
            if (!input) {
                cachedBytes = 0;
                init();
                return;
            }

            input.read(reinterpret_cast<char*>(buffer.data()), static_cast<std::streamsize>(bytesToRead));
            cachedBytes = static_cast<size_t>(input.gcount());

            if (cachedBytes < bytesToRead) {
                std::fill(buffer.begin() + cachedBytes, buffer.begin() + bytesToRead, 0);
            }
            cachedBytes = bytesToRead;

            int offset = beginBit % 8;
            if (offset != 0 && cachedBytes > 0) {
                for (size_t i = 0; i < cachedBytes; ++i) {
                    unsigned char curr = buffer[i];
                    unsigned char next = (i + 1 < cachedBytes) ? buffer[i + 1] : 0;
                    buffer[i] = static_cast<unsigned char>(((curr << offset) & 0xFFu) | (next >> (8 - offset)));
                }
            }

            init();
        }

        long readLong(int size) {
            long result = 0;
            while (size > 0 && pointer < cachedBytes) {
                int bitsToRead = std::min(leftBits, size);
                leftBits -= bitsToRead;
                unsigned int mask = (1u << bitsToRead) - 1u;
                unsigned int chunk = (buffer[pointer] >> leftBits) & mask;
                result = (result << bitsToRead) | static_cast<long>(chunk);
                size -= bitsToRead;
                if (leftBits == 0) {
                    pointer++;
                    leftBits = 8;
                }
            }

            if (size > 0) {
                result <<= size; // pad remaining bits with zeros when data exhausted
            }

            return result;
        }

        int readInt(int size) {
            return static_cast<int>(readLong(size));
        }

        float readFloat(int size) {
            std::uint32_t bits = static_cast<std::uint32_t>(readLong(size));
            float value;
            std::memcpy(&value, &bits, sizeof(float));
            return value;
        }

        double readDouble(int size) {
            std::uint64_t bits = static_cast<std::uint64_t>(readLong(size));
            double value;
            std::memcpy(&value, &bits, sizeof(double));
            return value;
        }

        bool readBoolean() {
            return readLong(1) > 0;
        }
    };

}  // namespace utils
