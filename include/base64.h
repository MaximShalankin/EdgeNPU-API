// Simple Base64 decoder - header only
// Public domain implementation

#ifndef BASE64_H
#define BASE64_H

#include <string>
#include <vector>
#include <stdexcept>

namespace base64 {

static const std::string base64_chars =
    "ABCDEFGHIJKLMNOPQRSTUVWXYZabcdefghijklmnopqrstuvwxyz0123456789+/";

static inline bool is_base64(unsigned char c) {
    return (isalnum(c) || (c == '+') || (c == '/'));
}

static inline std::string from_base64(const std::string& encoded_string) {
    size_t in_len = encoded_string.size();
    size_t i = 0;
    size_t j = 0;
    int in_ = 0;
    unsigned char char_array_4[4], char_array_3[3];
    std::string ret;

    // Skip data URL prefix if present
    size_t start = 0;
    size_t comma_pos = encoded_string.find(",");
    if (comma_pos != std::string::npos) {
        start = comma_pos + 1;
        in_len -= start;
    }

    const char* encoded_data = encoded_string.c_str() + start;

    while (in_len-- && (encoded_data[in_] != '=') && is_base64(encoded_data[in_])) {
        char_array_4[i++] = encoded_data[in_]; in_++;
        if (i == 4) {
            for (i = 0; i < 4; i++)
                char_array_4[i] = base64_chars.find(char_array_4[i]);

            char_array_3[0] = (char_array_4[0] << 2) + ((char_array_4[1] & 0x30) >> 4);
            char_array_3[1] = ((char_array_4[1] & 0xf) << 4) + ((char_array_4[2] & 0x3c) >> 2);
            char_array_3[2] = ((char_array_4[2] & 0x3) << 6) + char_array_4[3];

            for (i = 0; i < 3; i++)
                ret += char_array_3[i];
            i = 0;
        }
    }

    if (i) {
        for (j = i; j < 4; j++)
            char_array_4[j] = 0;

        for (j = 0; j < 4; j++)
            char_array_4[j] = base64_chars.find(char_array_4[j]);

        char_array_3[0] = (char_array_4[0] << 2) + ((char_array_4[1] & 0x30) >> 4);
        char_array_3[1] = ((char_array_4[1] & 0xf) << 4) + ((char_array_4[2] & 0x3c) >> 2);

        for (j = 0; j < i - 1; j++)
            ret += char_array_3[j];
    }

    return ret;
}

} // namespace base64

#endif // BASE64_H
