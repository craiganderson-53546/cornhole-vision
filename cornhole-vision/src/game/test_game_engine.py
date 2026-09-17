"""
Exercises GameEngine with synthetic BlobReading sequences -- no camera,
no detect_bags binary, just the pure game logic. Run directly:
    python3 test_game_engine.py
"""

from game_engine import GameEngine, BlobReading, parse_detect_bags_output

def show(label, engine):
    s = engine.snapshot()
    active = [b for b in s.bags if b.active]
    print(f"[{label}] bag_count={s.bag_count} total_thrown={s.total_thrown} "
          f"score A={s.score_a} B={s.score_b} round_ending={s.round_ending}")
    print(f"        status: {s.status_msg}")
    return s

def test_parse():
    out = "bag team=A in_hole=0 cx=612 cy=430 area=812\nbag team=B in_hole=1 cx=780 cy=415 area=790\n"
    readings = parse_detect_bags_output(out)
    assert len(readings) == 2
    assert readings[0].team == 'A' and readings[0].in_hole is False
    assert readings[1].team == 'B' and readings[1].in_hole is True
    print("test_parse: OK")

def test_add_requires_two_polls():
    e = GameEngine()
    reading = BlobReading(team='A', in_hole=False, cx=500, cy=300, area=800)
    e.process_poll([reading])
    s = show("after 1 poll (should NOT be confirmed yet)", e)
    assert s.total_thrown == 0, "bag confirmed too early"
    e.process_poll([reading])
    s = show("after 2 polls (should be confirmed)", e)
    assert s.total_thrown == 1 and s.bag_count == 1
    print("test_add_requires_two_polls: OK\n")

def test_single_noise_blob_ignored():
    e = GameEngine()
    e.process_poll([BlobReading(team='A', in_hole=False, cx=100, cy=100, area=800)])
    # different location next poll -> treated as a new, still-unconfirmed candidate
    e.process_poll([BlobReading(team='A', in_hole=False, cx=900, cy=600, area=800)])
    s = show("two one-off blobs in different spots", e)
    assert s.total_thrown == 0, "transient noise should not be confirmed as a bag"
    print("test_single_noise_blob_ignored: OK\n")

def test_removal_and_board_clear_scores_immediately():
    e = GameEngine()
    a1 = BlobReading(team='A', in_hole=True, cx=500, cy=300, area=800)   # hole = 3pts
    b1 = BlobReading(team='B', in_hole=False, cx=700, cy=300, area=800)  # board = 1pt
    for _ in range(ADD_POLLS := 2):
        e.process_poll([a1, b1])
    s = show("both bags confirmed", e)
    assert s.bag_count == 2 and s.total_thrown == 2

    # remove both bags (empty poll) -- should debounce over 2 polls, then
    # score immediately on board-clear (not wait for 8 bags / 3 min hold)
    e.process_poll([])
    show("1 empty poll (should still show 2 on board)", e)
    e.process_poll([])
    s = show("2 empty polls -> board clears -> should auto-score", e)
    assert s.bag_count == 0 and s.total_thrown == 0, "round should have reset"
    assert s.score_a == 2, f"expected RED net +2 (3 raw - 1 raw), got {s.score_a}"
    assert s.score_b == 0, f"expected BLUE net 0 (cancelled out), got {s.score_b}"
    print("test_removal_and_board_clear_scores_immediately: OK\n")

def test_full_round_all_eight_then_clear():
    e = GameEngine()
    # 4 red bags on board (1pt each = 4), 4 blue bags in hole (3pt each = 12)
    reds = [BlobReading(team='A', in_hole=False, cx=100+i*30, cy=100, area=800) for i in range(4)]
    blues = [BlobReading(team='B', in_hole=True, cx=800+i*30, cy=500, area=800) for i in range(4)]
    all_bags = reds + blues

    # confirm one at a time to also check MAX_BAGS bookkeeping
    seen = []
    for bag in all_bags:
        seen.append(bag)
        e.process_poll(list(seen))
        e.process_poll(list(seen))  # 2nd poll to cross ADD_CONFIRM_POLLS
    s = show("all 8 thrown", e)
    assert s.total_thrown == 8
    assert s.round_ending is True, "should be waiting for board clear after 8th bag"

    # clear the board -> should score immediately, not wait for the 3-min hold
    e.process_poll([])
    e.process_poll([])
    s = show("board cleared after all 8 thrown", e)
    assert s.bag_count == 0 and s.total_thrown == 0
    assert s.score_a == 0, f"expected RED net 0 (4 vs 12 -> blue wins), got {s.score_a}"
    assert s.score_b == 8, f"expected BLUE net +8 (12-4), got {s.score_b}"
    print("test_full_round_all_eight_then_clear: OK\n")

def test_new_game_resets_scores():
    e = GameEngine()
    e.state.score_a = 5
    e.state.score_b = 3
    e.new_game()
    s = show("after new_game()", e)
    assert s.score_a == 0 and s.score_b == 0 and s.total_thrown == 0
    print("test_new_game_resets_scores: OK\n")

if __name__ == "__main__":
    test_parse()
    test_add_requires_two_polls()
    test_single_noise_blob_ignored()
    test_removal_and_board_clear_scores_immediately()
    test_full_round_all_eight_then_clear()
    test_new_game_resets_scores()
    print("ALL TESTS PASSED")
