import os
import pickle
import numpy as np
import matplotlib.pyplot as plt
from sklearn.decomposition import PCA
import textalloc as ta

with open('data/final_data.pkl', 'rb') as file:
    data = pickle.load(file)

x, y, z = [], [], []
for player, player_data in data.items():
    x.append(sum(m['move_probs'].get(m['move'], 0.0) for _, m in player_data[0].iterrows()) / len(player_data[0]))
    y.append(sum(m['move_probs'].get(m['move'], 0.0) for _, m in player_data[2].iterrows()) / len(player_data[2]))
    z.append(sum(m['move_probs'].get(m['move'], 0.0) for _, m in player_data[4].iterrows()) / len(player_data[4]))

X = np.array([x, y, z]).T
pca = PCA(n_components=3)
pc1_values = pca.fit_transform(X)[:, 0]

# Normalise display names
display_names = []
for player in data.keys():
    if player == 'VachierLagrave':    display_names.append('Lagrave')
    elif player == 'Nepomniachtchi':  display_names.append('Nepo')
    else:                             display_names.append(player)

sorted_pairs   = sorted(zip(pc1_values, display_names))
pc1_sorted     = [v for v, _ in sorted_pairs]
names_sorted   = [n for _, n in sorted_pairs]

zeros = np.zeros(len(pc1_sorted))

fig, ax = plt.subplots(figsize=(12, 4))
ax.scatter(pc1_sorted, zeros, s=10, alpha=0.9)

ta.allocate(ax, pc1_sorted, zeros, names_sorted,
            x_scatter=pc1_sorted, y_scatter=zeros,
            textsize=8, draw_lines=True, linecolor='black', linewidth=0.5,
            max_distance=0.2)

ax.set_yticks([])
ax.set_xlabel('PC1 Projection (Humanness)')
ax.set_title('Projection of Players onto Principal Component 1')
plt.tight_layout()

os.makedirs('outputs', exist_ok=True)
plt.savefig('outputs/pc1_projection.png', dpi=300, bbox_inches='tight', pad_inches=0.3)
