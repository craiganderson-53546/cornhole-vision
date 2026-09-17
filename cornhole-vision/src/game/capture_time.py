import time
from game_server import capture_frame, run_detect_bags

times = []
for i in range(10):
    t0 = time.time()
    capture_frame("/tmp/timing_frame.yuv420")
    t1 = time.time()
    run_detect_bags("./detect_bags", "/tmp/timing_frame.yuv420", "session.cal")
    t2 = time.time()
    times.append((t1 - t0, t2 - t1, t2 - t0))
    print(f"run {i}: capture={t1-t0:.3f}s detect={t2-t1:.3f}s total={t2-t0:.3f}s")

totals = [t[2] for t in times]
print(f"\nmin={min(totals):.3f}s  max={max(totals):.3f}s  avg={sum(totals)/len(totals):.3f}s")
