#pragma once
#include <cstdint>

void bgr_to_rgb(const uint8_t* src, int width, int height, uint8_t* dst);
void normalize_rgb(const uint8_t* src, int width, int height, float* dst);
