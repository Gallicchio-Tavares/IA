from quarto_game import QuartoGame
from minimax_quarto import minimax_agent
from mcts_quarto import mcts_agent
import random

def human_turn(game, print_board=True):
    if print_board:
        game.print_board()
    print(f"\nSua vez. Peça pra colocar: {game.given_piece}")
    r = int(input("Linha (0-3): "))
    c = int(input("Coluna (0-3): "))
    print("Peças que você pode escolher:")
    print(game.available_pieces())
    piece_input = input("Digite uma peça (ex: 0101): ")
    piece = tuple(int(x) for x in piece_input)
    return (r, c), piece

def play_game(agent_type='mcts'):
    game = QuartoGame()
    first_piece = random.choice(list(game.pieces))
    game.given_piece = first_piece

    while not game.game_over():
        if game.current_player == 'X':
            pos, piece = human_turn(game)  # Já imprime o tabuleiro dentro de human_turn()
        else:
            print("\nRobô pensando...")
            if agent_type == 'mcts':
                pos, piece = mcts_agent(game, iterations=500)
            else:
                pos, piece = minimax_agent(game, depth=2)
            print(f"O robô joga em {pos} e dá {piece}")
            # Mostra o tabuleiro APENAS após a jogada do robô
            game.print_board()

        game.make_move(pos, piece)

    # Mostra o estado final do tabuleiro
    game.print_board()
    winner = game.winner()
    if winner:
        print(f"\nJogador {winner} venceu!")
    else:
        print("\nEmpate!")

if __name__ == "__main__":
    mode = input("Jogue contra (mcts/minimax): ").strip().lower()
    play_game(agent_type=mode)
