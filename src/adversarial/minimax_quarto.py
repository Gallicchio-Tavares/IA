from quarto_game import QuartoGame
import random

def heuristic(game, player):
    score = 0
    lines = []
    
    # Gera todas as linhas possíveis (horizontais, verticais, diagonais)
    for i in range(game.size):
        lines.append([game.board[i][j] for j in range(game.size)])  # horizontal
        lines.append([game.board[j][i] for j in range(game.size)])  # vertical
    
    lines.append([game.board[i][i] for i in range(game.size)])  # diagonal principal
    lines.append([game.board[i][game.size-1-i] for i in range(game.size)])  # diagonal secundária
    
    for line in lines:
        filled = [p for p in line if p is not None]  # Remove posições vazias
        if len(filled) < 2:  # Ignora linhas com menos de 2 peças
            continue
            
        # Extrai apenas as peças (ignora o jogador)
        pieces = [p[0] for p in filled]
        
        for attr in range(4):  # Verifica cada atributo (0-3)
            values = [p[attr] for p in pieces]
            if all(v == values[0] for v in values):
                # Pontua mais se for quase uma linha completa
                multiplier = len(filled) ** 2  # Quadrado do número de peças
                score += multiplier if player == game.current_player else -multiplier
    return score

def minimax(game, depth, maximizing_player, player):
    if game.game_over() or depth == 0:
        win = game.winner()
        if win == player:
            return 1000
        elif win is not None:  # Oponente venceu
            return -1000
        return heuristic(game, player)
    
    if maximizing_player:
        max_eval = float('-inf')
        for position in game.available_positions():
            for piece in game.available_pieces():
                new_game = game.copy()
                try:
                    new_game.make_move(position, piece)
                    eval = minimax(new_game, depth - 1, False, player)
                    max_eval = max(max_eval, eval)
                except:
                    continue  # Ignora movimentos inválidos
        return max_eval
    else:
        min_eval = float('inf')
        for position in game.available_positions():
            for piece in game.available_pieces():
                new_game = game.copy()
                try:
                    new_game.make_move(position, piece)
                    eval = minimax(new_game, depth - 1, True, player)
                    min_eval = min(min_eval, eval)
                except:
                    continue
        return min_eval

def minimax_agent(game, depth=2):
    best_score = float('-inf')
    best_move = None
    
    for position in game.available_positions():
        for piece in game.available_pieces():
            new_game = game.copy()
            try:
                new_game.make_move(position, piece)
                score = minimax(new_game, depth - 1, False, game.current_player)
                if score > best_score or best_move is None:
                    best_score = score
                    best_move = (position, piece)
            except:
                continue
    
    if best_move is None:  # Fallback caso não encontre movimento válido
        available_pos = game.available_positions()
        available_pieces = game.available_pieces()
        if available_pos and available_pieces:
            return (random.choice(available_pos), random.choice(available_pieces))
    
    return best_move