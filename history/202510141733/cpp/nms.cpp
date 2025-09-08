// Simple NMS helper for experiments
#include "nms.h"
#include <algorithm>

static float iou(const Box& a, const Box& b) {
    const float xx1 = std::max(a.x1, b.x1);
    const float yy1 = std::max(a.y1, b.y1);
    const float xx2 = std::min(a.x2, b.x2);
    const float yy2 = std::min(a.y2, b.y2);
    const float w = std::max(0.0f, xx2 - xx1);
    const float h = std::max(0.0f, yy2 - yy1);
    const float inter = w * h;
    const float area_a = (a.x2 - a.x1) * (a.y2 - a.y1);
    const float area_b = (b.x2 - b.x1) * (b.y2 - b.y1);
    const float uni = area_a + area_b - inter;
    if (uni <= 0.0f) {
        return 0.0f;
    }
    return inter / uni;
}

std::vector<int> nms(const std::vector<Box>& boxes, float iou_threshold) {
    std::vector<int> order(boxes.size());
    for (size_t i = 0; i < boxes.size(); ++i) {
        order[i] = static_cast<int>(i);
    }
    std::sort(order.begin(), order.end(), [&](int a, int b) {
        return boxes[a].score > boxes[b].score;
    });
    std::vector<int> keep;
    std::vector<bool> suppressed(boxes.size(), false);
    for (size_t i = 0; i < order.size(); ++i) {
        int idx = order[i];
        if (suppressed[idx]) {
            continue;
        }
        keep.push_back(idx);
        for (size_t j = i + 1; j < order.size(); ++j) {
            int next = order[j];
            if (suppressed[next]) {
                continue;
            }
            if (iou(boxes[idx], boxes[next]) >= iou_threshold) {
                suppressed[next] = true;
            }
        }
    }
    return keep;
}
