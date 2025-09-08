#include "preprocess.h"
#include <algorithm>

void bgr_to_rgb(const uint8_t* src, int width, int height, uint8_t* dst) {
    const int total = width * height;
    for (int i = 0; i < total; ++i) {
        const int idx = i * 3;
        dst[idx + 0] = src[idx + 2];
        dst[idx + 1] = src[idx + 1];
        dst[idx + 2] = src[idx + 0];
    }
}

void normalize_rgb(const uint8_t* src, int width, int height, float* dst) {
    const int total = width * height * 3;
    for (int i = 0; i < total; ++i) {
        dst[i] = static_cast<float>(src[i]) / 255.0f;
    }
}
