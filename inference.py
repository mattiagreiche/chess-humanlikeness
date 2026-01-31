from parse_raw_data import load_data
from maia2 import model, inference
import pandas as pd
import pickle

data = load_data()

maia2_model = model.from_pretrained(type="rapid", device="cpu")

num_train_moves = 6_000 # Number of moves to consider for 'training' (identity vector) for each phase
# Note that 'training' is not actually training a neural net model, but running inference on Maia-2 to get player humanness values
num_test_moves = 1_500 # Number of moves to consider for stylometric testing for each phase

num_total_moves = num_train_moves + num_test_moves

final_data = {}

for player, data in data.items():
    if len(data[0]) < num_total_moves or len(data[1]) < num_total_moves or len(data[2]) < num_total_moves:
        print(f"Skipping {player} due to insufficient data.")
        continue
    opening_data = pd.DataFrame(data[0], columns=['board', 'move', 'active_elo', 'inactive_elo']).sample(frac=1, random_state=0).reset_index(drop=True)
    mid_data = pd.DataFrame(data[1], columns=['board', 'move', 'active_elo', 'inactive_elo']).sample(frac=1, random_state=0).reset_index(drop=True)
    end_data = pd.DataFrame(data[2], columns=['board', 'move', 'active_elo', 'inactive_elo']).sample(frac=1, random_state=0).reset_index(drop=True)
    
    opening_train_data = opening_data.iloc[:num_train_moves]
    mid_train_data = mid_data.iloc[:num_train_moves]
    end_train_data = end_data.iloc[:num_train_moves]
    
    opening_test_data = opening_data.iloc[num_train_moves:num_total_moves]
    mid_test_data = mid_data.iloc[num_train_moves:num_total_moves]
    end_test_data = end_data.iloc[num_train_moves:num_total_moves]
    
    opening_train_data, opening_acc = inference.inference_batch(opening_train_data, maia2_model, verbose=0, batch_size=128, num_workers=0)
    print(f'{player} - Opening Accuracy: {opening_acc:.3f}')
    middle_train_data, mid_acc = inference.inference_batch(mid_train_data, maia2_model, verbose=0, batch_size=128, num_workers=0)
    print(f'{player} - Mid Accuracy: {mid_acc:.3f}')
    end_train_data, end_acc = inference.inference_batch(end_train_data, maia2_model, verbose=0, batch_size=128, num_workers=0)
    print(f'{player} - End Accuracy: {end_acc:.3f}')
    
    # We don't need to store these accuracy values, since they're top-1 accuracy, and we will be using the probabilities of the actual moves to calculate accuracy

    final_data[player] = [opening_train_data, opening_test_data, mid_train_data, mid_test_data, end_train_data, end_test_data]
    
with open('final_data.pkl', 'wb') as f:
    pickle.dump(final_data, f)
    print("Final data saved to final_data.pkl")
