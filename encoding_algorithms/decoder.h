#pragma once

#include "../examples/utils/stream_reader.h"
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

    class Decoder {
    protected:
        std::unique_ptr<utils::StreamReader> in;
        std::map<std::string, std::string> config;

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
        Decoder(const std::string& inputPath) {
            this->in = std::make_unique<utils::StreamReader>(inputPath);
        }

        Decoder(const std::string& inputPath, const std::string& configStr) {
            this->in = std::make_unique<utils::StreamReader>(inputPath);
            this->config = parseStringToMap(configStr);
        }

        virtual ~Decoder() = default;

        virtual int decodeInt() { return 0; }
        virtual long decodeLong() { return 0; }
        virtual float decodeFloat() { return 0; }
        virtual double decodeDouble() { return 0; }
    };

}
