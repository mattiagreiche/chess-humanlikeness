import pickle
import matplotlib.pyplot as plt
import numpy as np
from sklearn.decomposition import PCA

with open('data/final_data.pkl', 'rb') as file:
    data = pickle.load(file)

plt.style.use('seaborn-v0_8-paper')

fig_3d = plt.figure(figsize=(10, 8))
ax = plt.axes(projection='3d')

x = []
y = []
z = []

for player, player_data in data.items():
    opening_acc_cumul = 0
    for index, move in player_data[0].iterrows():
        opening_acc_cumul += move['move_probs'].get(move['move'], 0.0)
    opening_acc = opening_acc_cumul/len(player_data[0])  # Average over all moves

    middle_acc_cumul = 0
    for index, move in player_data[2].iterrows():
        middle_acc_cumul += move['move_probs'].get(move['move'], 0.0)
    middle_acc = middle_acc_cumul/len(player_data[2])

    end_acc_cumul = 0
    for index, move in player_data[4].iterrows():
        end_acc_cumul += move['move_probs'].get(move['move'], 0.0)
    end_acc = end_acc_cumul/len(player_data[4])

    x.append(opening_acc) # Opening humanness
    y.append(middle_acc) # Middle game humanness
    z.append(end_acc) # End game humanness

# PCA
X = np.array([x, y, z]).T
pca = PCA(n_components=3)
X_pca = pca.fit_transform(X)
mean = np.mean(X, axis=0)
pc1 = pca.components_[0]

ax.scatter(x, y, z, alpha=0.9)

for i, player in enumerate(data.keys()):
    ax.text(x[i], y[i], z[i], player, size=6, zorder=1, color='k')

# PCA line
line_length = 0.05
start = mean - line_length * pc1
end = mean + line_length * pc1

ax.plot([start[0], end[0]],
        [start[1], end[1]],
        [start[2], end[2]],
        color='red', linewidth=2, label='PC1 (main trend)')

ax.set_xlabel('Opening Humanness')
ax.set_ylabel('Middle Game Humanness')
ax.set_zlabel('End Game Humanness')

ax.set_title('Human-Likeness of Chess Grandmasters by Game Phase')
plt.show()

X = np.column_stack((x, y, z))  # shape (n_samples, 3)

pca = PCA(n_components=3)
pca.fit(X)

print("Explained variance ratio:", pca.explained_variance_ratio_)
print("Variance explained by PC1:", pca.explained_variance_ratio_[0])
