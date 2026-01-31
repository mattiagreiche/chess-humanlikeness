import pickle
import numpy as np
import plotly.graph_objects as go
from sklearn.decomposition import PCA

# --- Data Loading and Processing (Kept Identical) ---
with open('final_data.pkl', 'rb') as file:
    data = pickle.load(file)

x = []
y = []
z = []
names = [] # List to store player names for Plotly labels

for player, player_data in data.items():
    names.append(player)
    
    # Opening
    opening_acc_cumul = 0
    for index, move in player_data[0].iterrows():
        for predicted_move, probability in move['move_probs'].items():
            if predicted_move == move['move']: 
                opening_acc_cumul += probability
    opening_acc = opening_acc_cumul/len(player_data[0])
    
    # Middle Game
    middle_acc_cumul = 0
    for index, move in player_data[2].iterrows():
        for predicted_move, probability in move['move_probs'].items():
            if predicted_move == move['move']: 
                middle_acc_cumul += probability
    middle_acc = middle_acc_cumul/len(player_data[2])  
    
    # End Game
    end_acc_cumul = 0
    for index, move in player_data[4].iterrows():
        for predicted_move, probability in move['move_probs'].items():
            if predicted_move == move['move']:
                end_acc_cumul += probability
    end_acc = end_acc_cumul/len(player_data[4])  
    
    x.append(opening_acc)
    y.append(middle_acc)
    z.append(end_acc)

# --- PCA Logic (Kept Identical) ---
X = np.array([x, y, z]).T
pca = PCA(n_components=3)
X_pca = pca.fit_transform(X)
mean = np.mean(X, axis=0)
pc1 = pca.components_[0]

# PCA line calculation
line_length = 0.05
start = mean - line_length * pc1
end = mean + line_length * pc1

# --- Plotly Visualization ---
fig = go.Figure()

# 1. Add Scatter Points with Text Labels
fig.add_trace(go.Scatter3d(
    x=x, y=y, z=z,
    mode='markers+text',
    text=names,
    textposition="top center", # Adjusts text slightly above markers to prevent overlap
    textfont=dict(size=10, color='black'),
    marker=dict(
        size=5,
        color='blue', # Standard color, can be changed
        opacity=0.9
    ),
    name='Players'
))

# 2. Add PCA Line (Red)
fig.add_trace(go.Scatter3d(
    x=[start[0], end[0]],
    y=[start[1], end[1]],
    z=[start[2], end[2]],
    mode='lines',
    line=dict(color='red', width=5),
    name='PC1 (main trend)'
))

# 3. Layout Configuration
fig.update_layout(
    title='Human-Likeness of Chess Grandmasters by Game Phase',
    width=1000,
    height=800,
    scene=dict(
        xaxis_title='Opening Human-likeness',
        yaxis_title='Middle Game Human-likeness',
        zaxis_title='End Game Human-likeness'
    ),
    margin=dict(l=0, r=0, b=0, t=50) # Tighter margins
)

fig.update_layout(
    uniformtext_minsize=5,   # Minimum font size
    uniformtext_mode='hide'  # Hide if it doesn't fit
)

fig.write_html("plotly_3d.html")

fig.show()

# --- PCA Stats Output ---
print("Explained variance ratio:", pca.explained_variance_ratio_)
print("Variance explained by PC1:", pca.explained_variance_ratio_[0])