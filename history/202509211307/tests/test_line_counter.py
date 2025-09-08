from src.counting.line_counter import LineCounter

def det(track_id, cx, cy):
    return {"track_id": track_id, "center": (cx, cy)}

def test_line_counter_counts_once():
    counter = LineCounter(line_y=100, direction="down")
    counter.update([det(1, 10, 90)])
    counter.update([det(1, 10, 110)])
    assert counter.update([]) == 1

def test_line_counter_up_direction():
    counter = LineCounter(line_y=100, direction="up")
    counter.update([det(2, 10, 110)])
    counter.update([det(2, 10, 90)])
    assert counter.update([]) == 1
