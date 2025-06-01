import random

def evaluate(game, player):
    return len(game.available_moves())

def minimax(game, depth, maximizing, player):
    winner = game.winner()
    if winner == player:
        return 1000
    elif winner is not None:
        return -1000
    elif depth == 0 or game.game_over():
        return evaluate(game, player)

    if maximizing:
        max_eval = float('-inf')
        for move in get_all_moves(game):
            new_game = game.copy()
            new_game.make_move(move)
            eval = minimax(new_game, depth - 1, False, player)
            max_eval = max(max_eval, eval)
        return max_eval
    else:
        min_eval = float('inf')
        for move in get_all_moves(game):
            new_game = game.copy()
            new_game.make_move(move)
            eval = minimax(new_game, depth - 1, True, player)
            min_eval = min(min_eval, eval)
        return min_eval

def get_all_moves(game):
    moves = []
    for pos in game.available_moves():
        row, col = pos
        for idx, piece in enumerate(game.all_pieces):
            if piece in game.available_pieces and piece != game.selected_piece:
                moves.append((row, col, idx))
    return moves

def best_move_quarto(game, depth=2):
    player = game.current
    best_score = float('-inf')
    best_move = None
    moves = get_all_moves(game)
    random.shuffle(moves)  # add aleatoriedade

    for move in moves:
        new_game = game.copy()
        new_game.make_move(move)
        score = minimax(new_game, depth - 1, False, player)
        if score > best_score:
            best_score = score
            best_move = move
    return best_move
