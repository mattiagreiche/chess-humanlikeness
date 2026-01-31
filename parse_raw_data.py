from pathlib import Path
import chess.pgn
import pandas as pd
import pickle
from phase_divider import is_opening, is_midgame, is_endgame

def load_data(force_reload=False):
    cache_file = "cached_data.pkl"
    
    # Load from cache if available (this data load function is quite costly)
    if not force_reload and Path(cache_file).exists():
        with open(cache_file, 'rb') as f:
            print("Loading cached data...")
            return pickle.load(f)
    
    pgn_folder = Path('pgn_files')

    data = {}

    for pgn_file in pgn_folder.glob('*.pgn'):
        print(f"Processing {pgn_file.name}...")
        early_game = []
        mid_game = []
        late_game = []
        with open(pgn_file, 'r', encoding='latin1') as file:
            while True:
                game = chess.pgn.read_game(file)
                if game is None:
                    break
                
                board = game.board()
                player_name = pgn_file.name.replace('.pgn', '')
                headers = game.headers
                
                if player_name in headers.get('White'):
                    player_color = 'white'
                elif player_name in headers.get('Black'):
                    player_color = 'black'
                if player_color is None:
                    raise ValueError(f"Player {player_name} not found in game headers.")
                
                try:
                    white_elo = int(headers.get('WhiteElo', 2500))
                except ValueError:
                    white_elo = 2500
                try:
                    black_elo = int(headers.get('BlackElo', 2500))
                except ValueError:
                    black_elo = 2500
                    
                active_elo = white_elo if player_color == 'white' else black_elo
                inactive_elo = black_elo if player_color == 'white' else white_elo
                
                is_players_turn = (player_color == 'white')
                
                for move in game.mainline_moves():
                    if is_players_turn:
                        turn_data = [
                            board.fen(),
                            move.uci(),
                            active_elo,
                            inactive_elo
                        ]
                        if is_opening:
                            early_game.append(turn_data)
                        if is_midgame(board):
                            mid_game.append(turn_data)
                        if is_endgame(board):
                            late_game.append(turn_data)
                    board.push(move)
                    is_players_turn = not is_players_turn

        data[player_name] = (early_game, mid_game, late_game)
     
    # Save result
    with open(cache_file, 'wb') as f:
        pickle.dump(data, f)
        print("Cached parsed data to disk.")   
        
    return data
