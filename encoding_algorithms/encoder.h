#pragma once

#include "../examples/utils/stream_writer.h"
#include "../examples/utils/memory_stream_writer.h"
#include "../examples/utils/base_stream_writer.h"
#include <string>
#include <vector>
#include <map>
#include <memory>
#include <sstream>
#include <utility>

#ifndef ENCODING_ALGORITHMS_MAKE_UNIQUE_COMPAT
#define ENCODING_ALGORITHMS_MAKE_UNIQUE_COMPAT
#if __cplusplus < 201402L
namespace std {
    template <class T, class... Args>
    std::unique_ptr<T> make_unique(Args&&... args) {
        return std::unique_ptr<T>(new T(std::forward<Args>(args)...));
    }
}
#endif
#endif

namespace encoding_algorithm {

    class Encoder {
    protected:
        std::shared_ptr<utils::BaseStreamWriter> out;
        std::map<std::string, double> meta;
        std::map<std::string, std::string> config;
        std::string outputPath;

        void copyBaseTo(Encoder* target) {
            target->meta = this->meta;
            target->config = this->config;
        }

        std::map<std::string, std::string> parseStringToMap(const std::string& input) {
            std::map<std::string, std::string> map;
            std::string processed_input = input;
            if (!processed_input.empty() && processed_input.front() == '{' && processed_input.back() == '}') {
                processed_input = processed_input.substr(1, processed_input.length() - 2);
            }

            std::string current_pair;
            std::stringstream ss(processed_input);

            while (std::getline(ss, current_pair, ',')) {
                std::stringstream pair_ss(current_pair);
                std::string key, value;
                if (std::getline(pair_ss, key, ':') && std::getline(pair_ss, value)) {
                    // Trim whitespace
                    key.erase(0, key.find_first_not_of(" \t\n\r"));
                    key.erase(key.find_last_not_of(" \t\n\r") + 1);
                    value.erase(0, value.find_first_not_of(" \t\n\r"));
                    value.erase(value.find_last_not_of(" \t\n\r") + 1);
                    map[key] = value;
                }
            }
            return map;
        }


    public:
        // 默认构造函数
        Encoder() = default;

        Encoder(const std::string& outputPath) : outputPath(outputPath) {
            this->out = std::make_shared<utils::StreamWriter>(outputPath);
        }

        Encoder(const std::string& outputPath, const std::string& configStr) : outputPath(outputPath) {
            this->out = std::make_shared<utils::StreamWriter>(outputPath);
            this->config = parseStringToMap(configStr);
        }

        // 多个Encoder共用一个StreamWriter时使用
        Encoder(std::shared_ptr<utils::BaseStreamWriter> sharedOut, const std::string& configStr)
            : out(std::move(sharedOut)) {
            this->config = parseStringToMap(configStr);
        }

        Encoder(std::shared_ptr<utils::BaseStreamWriter> sharedOut)
            : out(std::move(sharedOut)) {}

        virtual ~Encoder() = default;

        virtual std::unique_ptr<Encoder> deepCopy() = 0;

        void flush() {
            this->out->clear();
        }

        virtual int encode(int value) { return 0; }
        virtual int encode(long value) { return 0; }
        virtual int encode(float value) { return 0; }
        virtual int encode(double value) { return 0; }

        virtual int close() {
            return 0;
        }

        std::map<std::string, double> getMeta() {
            return meta;
        }
    };

}
