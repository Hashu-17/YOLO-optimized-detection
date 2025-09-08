class LineCounter:
    def __init__(self, line_y, direction="down", history=30):
        self.line_y = line_y
        self.direction = direction
        self.history = history
        self.track_history = {}
        self.counted = set()

    def update(self, detections):
        for det in detections:
            track_id = det["track_id"]
            cx, cy = det["center"]
            track = self.track_history.setdefault(track_id, [])
            track.append((cx, cy))
            if len(track) > self.history:
                track.pop(0)
            if len(track) < 2:
                continue
            prev_y = track[-2][1]
            curr_y = track[-1][1]
            if self.direction == "down":
                crossed = prev_y < self.line_y <= curr_y
            else:
                crossed = prev_y > self.line_y >= curr_y
            if crossed:
                self.counted.add(track_id)
        return len(self.counted)
