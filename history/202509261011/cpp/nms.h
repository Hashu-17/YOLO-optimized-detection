#pragma once
#include <vector>

struct Box {
    float x1;
    float y1;
    float x2;
    float y2;
    float score;
};

std::vector<int> nms(const std::vector<Box>& boxes, float iou_threshold);
