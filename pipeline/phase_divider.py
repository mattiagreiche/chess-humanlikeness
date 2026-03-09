import chess

# Definitions of opening, middle, and end game used by Lichess

def majors_and_minors(board) -> int:
    major_and_minor_pieces = [chess.ROOK, chess.QUEEN, chess.BISHOP, chess.KNIGHT]
    return sum(
        len(board.pieces(piece_type, chess.WHITE)) + len(board.pieces(piece_type, chess.BLACK)) for piece_type in major_and_minor_pieces
    )
    
def backrank_sparse(board) -> bool:
    white_back = sum(1 for square in chess.SquareSet(chess.BB_RANK_1) if board.piece_type_at(square))
    black_back = sum(1 for square in chess.SquareSet(chess.BB_RANK_8) if board.piece_type_at(square))
    return white_back < 4 or black_back < 4

def region_score(y, white, black) -> int:
    # Translated from Lichess' Scala source code
    match (white, black):
        case (0, 0): return 0
        case (1, 0): return 1 + (8 - y)
        case (2, 0): return 2 + max(y - 2, 0) if y > 2 else 0
        case (3, 0): return 3 + max(y - 1, 0) if y > 1 else 0
        case (4, 0): return 3 + max(y - 1, 0) if y > 1 else 0
        case (0, 1): return 1 + y
        case (1, 1): return 5 + abs(4 - y)
        case (2, 1): return 4 + (y - 1)
        case (3, 1): return 5 + (y - 1)
        case (0, 2): return 2 + (6 - y) if y < 6 else 0
        case (1, 2): return 4 + (7 - y)
        case (2, 2): return 7
        case (0, 3): return 3 + (7 - y) if y < 7 else 0
        case (1, 3): return 5 + (7 - y)
        case (0, 4): return 3 + (7 - y) if y < 7 else 0
        case _: return 0
        
def mixedness(board) -> int:
    # Also translated from Lichess' source code
    score = 0
    region_mask = 0x0303
    for y in range(7):
        for x in range(7):
            shift = x + 8 * y
            region = region_mask << shift
            white = bin(board.occupied_co[chess.WHITE] & region).count("1")
            black = bin(board.occupied_co[chess.BLACK] & region).count("1")
            score += region_score(y + 1, white, black)
    return score

def is_endgame(board) -> bool:
    return majors_and_minors(board) <= 6

def is_midgame(board) -> bool:
    return (majors_and_minors(board) <= 10 or
            backrank_sparse(board) or
            mixedness(board) > 150) and not is_endgame(board)

def is_opening(board) -> bool:
    return not is_midgame(board) and not is_endgame(board)