import json

with open("ai/checkpoints/rect-9-9_s2_p2_tsl1.2_s2p1k1.2k2/cnn_000135_traj.json") as f:
    j = json.load(f)
    print(j[0][1])