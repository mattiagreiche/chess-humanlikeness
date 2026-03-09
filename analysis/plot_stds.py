import os
import pickle
import matplotlib.pyplot as plt
import numpy as np
import textalloc as ta
from matplotlib.lines import Line2D
from sklearn.decomposition import PCA

with open('data/final_data.pkl', 'rb') as file:
    data = pickle.load(file)

plt.style.use('seaborn-v0_8-paper')

fig_3d = plt.figure(figsize=(12, 12))
ax = plt.axes(projection='3d', proj_type='persp')

x = []
y = []
z = []

for player, player_data in data.items():
    opening_probs = []
    for index, move in player_data[0].iterrows():
        opening_probs.append(move['move_probs'].get(move['move'], 0.0))
    opening_std = np.std(opening_probs)  # Standard deviation over all moves

    middle_probs = []
    for index, move in player_data[2].iterrows():
        middle_probs.append(move['move_probs'].get(move['move'], 0.0))
    middle_std = np.std(middle_probs)

    end_probs = []
    for index, move in player_data[4].iterrows():
        end_probs.append(move['move_probs'].get(move['move'], 0.0))
    end_std = np.std(end_probs)

    x.append(opening_std) # Opening humanness
    y.append(middle_std) # Middle game humanness
    z.append(end_std) # End game humanness

# PCA
X = np.array([x, y, z]).T
pca = PCA(n_components=3)
pca.fit(X)
mean = np.mean(X, axis=0)
pc1 = pca.components_[0]

ax.scatter(x, y, z, alpha=0.9)

ta.allocate(ax, x, y, list(data.keys()), z=z,
            x_scatter=x, y_scatter=y, z_scatter=z,
            textsize=6, draw_lines=True, linecolor='black', linewidth=0.5)

# PCA line
line_length = 0.01
start = mean - line_length * pc1
end = mean + line_length * pc1

ax.quiver(start[0], start[1], start[2],
          end[0] - start[0], end[1] - start[1], end[2] - start[2],
          color='red', linewidth=2, arrow_length_ratio=0.03, alpha=0.5)

pc1_scores = X @ pc1
r2_opening = np.corrcoef(x, pc1_scores)[0, 1] ** 2
r2_middle  = np.corrcoef(y, pc1_scores)[0, 1] ** 2
r2_end     = np.corrcoef(z, pc1_scores)[0, 1] ** 2
pc1_pct = pca.explained_variance_ratio_[0]

blank = Line2D([0], [0], alpha=0)
ax.legend(
    [Line2D([0], [0], color='red', linewidth=2), blank, blank, blank],
    [r'$R^2$ with PC1:',
     f'  \u2022 Opening = {r2_opening:.2f}',
     f'  \u2022 Middle = {r2_middle:.2f}',
     f'  \u2022 Endgame = {r2_end:.2f}'],
    title=f'PC1 ({pc1_pct:.1%})',
    loc='lower right', fontsize=8, title_fontsize=9, framealpha=0.9,
)

ax.set_xlabel('Opening Humanness STDEV', labelpad=8)
ax.set_ylabel('Middle Game Humanness STDEV', labelpad=8)
ax.set_zlabel('End Game Humanness STDEV', labelpad=8)
ax.set_title('STDEV of Human-Likeness of Chess Grandmasters by Game Phase', fontsize=16)

print("Explained variance ratio:", pca.explained_variance_ratio_)
print("Variance explained by PC1:", pca.explained_variance_ratio_[0])

ax.view_init(elev=30, azim=-45)
os.makedirs('outputs', exist_ok=True)
plt.savefig('outputs/regression_stds_plot.png', dpi=300, bbox_inches='tight', pad_inches=0.3)
