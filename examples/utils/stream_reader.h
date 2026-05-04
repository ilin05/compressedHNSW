#pragma once

#include <string>
#include <vector>
#include <fstream>
#include <cstring>

namespace utils {
    class StreamReader {
    private:
        int bufferSize = 1024;
        char buffer[1024];
        int cacheBytes = 0;
        int pointer = 0;
        int leftBits = 8;
        long breakpoint = 0;
        std::string fileName;

    public:
        explicit StreamReader(const std::string& path){
            fileName = path;
        }

        void cacheData(){
            std::ifstream inputStream;
            inputStream.open(fileName, std::ios::binary);
            inputStream.clear(); // clear any eof/fail bits before seeking
            inputStream.seekg(breakpoint, std::ios::beg);
            cacheBytes = inputStream.readsome(buffer, bufferSize);
            init();
        }

        void init(){
            leftBits = 8;
            pointer = 0;
        }

        long readLong(int size){
            long res = 0;
            while(size >= leftBits){
                int mask = (1 << leftBits) - 1;
                res = (res << leftBits) | ((buffer[pointer] & mask));
                size -= leftBits;
                pointer++;
                if(pointer >= cacheBytes){
                    cacheData();
                }
                leftBits = 8;
            }
            if(size > 0){
                leftBits -= size;
                int mask = (1 << size) - 1;
                res = (res << size) | ((buffer[pointer] >> leftBits) & mask);
            }
            return res;
        }

        int readInt(int size){
            return (int)readLong(size);
        }

        float readFloat(int size){
            long long_bits = readLong(size);
            float f;
            std::memcpy(&f, &long_bits, sizeof(float));
            return f;
        }

        bool readBoolean(){
            return readLong(1) > 0;
        }

        double readDouble(int size){
            long long_bits = readLong(size);
            double d;
            std::memcpy(&d, &long_bits, sizeof(double));
            return d;
        }
    };
}
