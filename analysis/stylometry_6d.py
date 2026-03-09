# Stylometric verification (6D): k-NN re-identification using mean + std humanness per phase (opening/mid/endgame).
# See stylometry_3d.py for the simpler 3D version using means only.

import os
import pickle
import numpy as np
from maia2 import model, inference

with open('data/final_data.pkl', 'rb') as file:
    data = pickle.load(file)

players = list(data.keys())

# Training vectors: [open_mean, mid_mean, end_mean, open_std, mid_std, end_std]
# Train splits already have move_probs from pipeline/inference.py
train_vectors = []
for player_data in data.values():
    o_probs = [m['move_probs'].get(m['move'], 0.0) for _, m in player_data[0].iterrows()]
    m_probs = [m['move_probs'].get(m['move'], 0.0) for _, m in player_data[2].iterrows()]
    e_probs = [m['move_probs'].get(m['move'], 0.0) for _, m in player_data[4].iterrows()]
    train_vectors.append([
        np.mean(o_probs), np.mean(m_probs), np.mean(e_probs),
        np.std(o_probs),  np.std(m_probs),  np.std(e_probs),
    ])
train_vectors = np.array(train_vectors)

CACHE = 'data/stylo_data_6d.pkl'

if os.path.exists(CACHE):
    print("Loading cached test vectors...")
    with open(CACHE, 'rb') as f:
        test_vectors = pickle.load(f)
else:
    # Run Maia-2 on the held-out test splits (indices 1, 3, 5, no move_probs yet)
    maia2_model = model.from_pretrained(type="rapid", device="cpu")

    test_vectors = []
    for player, player_data in data.items():
        opening_test, _ = inference.inference_batch(player_data[1], maia2_model, verbose=0, batch_size=128, num_workers=0)
        mid_test, _     = inference.inference_batch(player_data[3], maia2_model, verbose=0, batch_size=128, num_workers=0)
        end_test, _     = inference.inference_batch(player_data[5], maia2_model, verbose=0, batch_size=128, num_workers=0)

        o_probs = [m['move_probs'].get(m['move'], 0.0) for _, m in opening_test.iterrows()]
        m_probs = [m['move_probs'].get(m['move'], 0.0) for _, m in mid_test.iterrows()]
        e_probs = [m['move_probs'].get(m['move'], 0.0) for _, m in end_test.iterrows()]

        test_vectors.append([
            np.mean(o_probs), np.mean(m_probs), np.mean(e_probs),
            np.std(o_probs),  np.std(m_probs),  np.std(e_probs),
        ])
        print(f"{player} done")

    test_vectors = np.array(test_vectors)

    os.makedirs('data', exist_ok=True)
    with open(CACHE, 'wb') as f:
        pickle.dump(test_vectors, f)
    print(f"Saved test vectors to {CACHE}")

# Nearest-neighbour re-identification
top1, top3, top5 = 0, 0, 0

for i, player in enumerate(players):
    distances = np.linalg.norm(train_vectors - test_vectors[i], axis=1)
    ranked    = np.argsort(distances)

    for rank, idx in enumerate(ranked[:5]):
        print(f"  {rank + 1} → {player}: {players[idx]}  (dist={distances[idx]:.4f})")
        if players[idx] == player:
            if rank == 0: top1 += 1
            if rank <= 2: top3 += 1
            top5 += 1
            break
    print()

print(f"Top-1: {top1}/{len(players)}   Top-3: {top3}/{len(players)}   Top-5: {top5}/{len(players)}")
