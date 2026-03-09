# Stylometric verification (3D): k-NN re-identification using mean humanness per phase (opening/mid/endgame).
# See stylometry_6d.py for the 6D version that also includes std per phase.

import os
import pickle
import numpy as np
from maia2 import model, inference

with open('data/final_data.pkl', 'rb') as file:
    data = pickle.load(file)

players = list(data.keys())

# Training humanness vectors, train splits already have move_probs from pipeline/inference.py
train_vectors = []
for player_data in data.values():
    opening_acc = sum(m['move_probs'].get(m['move'], 0.0) for _, m in player_data[0].iterrows()) / len(player_data[0])
    mid_acc     = sum(m['move_probs'].get(m['move'], 0.0) for _, m in player_data[2].iterrows()) / len(player_data[2])
    end_acc     = sum(m['move_probs'].get(m['move'], 0.0) for _, m in player_data[4].iterrows()) / len(player_data[4])
    train_vectors.append([opening_acc, mid_acc, end_acc])
train_vectors = np.array(train_vectors)

CACHE = 'data/stylo_data_3d.pkl'

if os.path.exists(CACHE):
    print("Loading cached test vectors...")
    with open(CACHE, 'rb') as f:
        cached = pickle.load(f)
    test_acc = np.array([[cached['x_test_acc'][i], cached['y_test_acc'][i], cached['z_test_acc'][i]] for i in range(len(players))])
    test_std = np.array([[cached['x_test_std'][i], cached['y_test_std'][i], cached['z_test_std'][i]] for i in range(len(players))])
else:
    # Run Maia-2 on the held-out test splits (indices 1, 3, 5, no move_probs yet)
    maia2_model = model.from_pretrained(type="rapid", device="cpu")

    test_acc = []   # (opening_acc, mid_acc, end_acc) per player
    test_std = []   # (opening_std, mid_std, end_std) per player

    for player, player_data in data.items():
        opening_test, _ = inference.inference_batch(player_data[1], maia2_model, verbose=0, batch_size=128, num_workers=0)
        mid_test, _     = inference.inference_batch(player_data[3], maia2_model, verbose=0, batch_size=128, num_workers=0)
        end_test, _     = inference.inference_batch(player_data[5], maia2_model, verbose=0, batch_size=128, num_workers=0)

        o_probs = [m['move_probs'].get(m['move'], 0.0) for _, m in opening_test.iterrows()]
        m_probs = [m['move_probs'].get(m['move'], 0.0) for _, m in mid_test.iterrows()]
        e_probs = [m['move_probs'].get(m['move'], 0.0) for _, m in end_test.iterrows()]

        test_acc.append([np.mean(o_probs), np.mean(m_probs), np.mean(e_probs)])
        test_std.append([np.std(o_probs),  np.std(m_probs),  np.std(e_probs)])
        print(f"{player} done")

    test_acc = np.array(test_acc)
    test_std = np.array(test_std)

    os.makedirs('data', exist_ok=True)
    with open(CACHE, 'wb') as f:
        pickle.dump({
            'players':    players,
            'x_test_acc': test_acc[:, 0].tolist(),
            'y_test_acc': test_acc[:, 1].tolist(),
            'z_test_acc': test_acc[:, 2].tolist(),
            'x_test_std': test_std[:, 0].tolist(),
            'y_test_std': test_std[:, 1].tolist(),
            'z_test_std': test_std[:, 2].tolist(),
        }, f)
    print(f"Saved test vectors to {CACHE}")

# Nearest-neighbour re-identification
top1, top3, top5 = 0, 0, 0

for i, player in enumerate(players):
    distances = np.linalg.norm(train_vectors - test_acc[i], axis=1)
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
