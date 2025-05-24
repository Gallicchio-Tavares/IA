from quarto import Quarto
from minimax_quarto import best_move_quarto
from mcts_quarto import quarto_mcts     
from colorama import Fore, init
import random
import time

init(autoreset=True)

###########################
# Jogador Humano vs Minimax
###########################
def play_human_vs_minimax():
    game = Quarto()
    start_piece = random.choice(game.available_pieces)
    game.selected_piece = start_piece
    game.available_pieces.remove(start_piece)

    while not game.game_over():
        game.print_board()
        if game.current == 0:
            row = int(input("Linha: "))
            col = int(input("Coluna: "))
            game.print_available_pieces()
            next_piece = int(input("Índice da peça para o oponente: "))
            move = (row, col, next_piece)
        else:
            print(Fore.BLUE + "Minimax pensando...")
            move = best_move_quarto(game, depth=2)
            print(Fore.BLUE + f"Minimax jogou {move}")

        game.make_move(move)

    game.print_board()
    winner = game.winner()
    print(Fore.GREEN + (f"Jogador {winner} venceu!" if winner is not None else "Empate!"))


###########################
# Jogador Humano vs MCTS
###########################
def play_human_vs_mcts():
    game = Quarto()
    start_piece = random.choice(game.available_pieces)
    game.selected_piece = start_piece
    game.available_pieces.remove(start_piece)

    while not game.game_over():
        game.print_board()
        if game.current == 0:
            row = int(input("Linha: "))
            col = int(input("Coluna: "))
            game.print_available_pieces()
            next_piece = int(input("Índice da peça para o oponente: "))
            move = (row, col, next_piece)
        else:
            print(Fore.BLUE + "MCTS pensando...")
            move = quarto_mcts(game, iterations=1000, time_limit=2)
            print(Fore.BLUE + f"MCTS jogou {move}")

        game.make_move(move)

    game.print_board()
    winner = game.winner()
    print(Fore.GREEN + (f"Jogador {winner} venceu!" if winner is not None else "Empate!"))


###########################
# MCTS vs Minimax
###########################
def play_mcts_vs_minimax():
    game = Quarto()
    start_piece = random.choice(game.available_pieces)
    game.selected_piece = start_piece
    game.available_pieces.remove(start_piece)

    while not game.game_over():
        game.print_board()
        if game.current == 0:
            print(Fore.BLUE + "MCTS pensando...")
            move = quarto_mcts(game, iterations=1000, time_limit=1)
            print(Fore.BLUE + f"MCTS jogou {move}")
        else:
            print(Fore.RED + "Minimax pensando...")
            move = best_move_quarto(game, depth=2)
            print(Fore.RED + f"Minimax jogou {move}")

        game.make_move(move)
        time.sleep(0.5)

    game.print_board()
    winner = game.winner()
    print(Fore.GREEN + (f"Jogador {winner} venceu!" if winner is not None else "Empate!"))


###########################
# MCTS vs MCTS
###########################
def play_mcts_vs_mcts():
    game = Quarto()
    start_piece = random.choice(game.available_pieces)
    game.selected_piece = start_piece
    game.available_pieces.remove(start_piece)

    while not game.game_over():
        game.print_board()
        print(Fore.BLUE + f"MCTS Jogador {game.current} pensando...")
        move = quarto_mcts(game, iterations=1000, time_limit=1)
        print(Fore.BLUE + f"MCTS Jogador {game.current} jogou {move}")
        game.make_move(move)
        time.sleep(0.5)

    game.print_board()
    winner = game.winner()
    print(Fore.GREEN + (f"Jogador {winner} venceu!" if winner is not None else "Empate!"))


###########################
# Minimax vs Minimax
###########################
def play_minimax_vs_minimax():
    game = Quarto()
    start_piece = random.choice(game.available_pieces)
    game.selected_piece = start_piece
    game.available_pieces.remove(start_piece)

    while not game.game_over():
        game.print_board()
        print(Fore.RED + f"Minimax Jogador {game.current} pensando...")
        move = best_move_quarto(game, depth=2)
        print(Fore.RED + f"Minimax Jogador {game.current} jogou {move}")
        game.make_move(move)
        time.sleep(0.5)

    game.print_board()
    winner = game.winner()
    print(Fore.GREEN + (f"Jogador {winner} venceu!" if winner is not None else "Empate!"))


###########################
# MENU
###########################
def main_menu():
    while True:
        print(Fore.CYAN + "\n==== MENU QUARTO ====")
        print("1. Humano vs Minimax")
        print("2. Humano vs MCTS")
        print("3. MCTS vs Minimax")
        print("4. MCTS vs MCTS")
        print("5. Minimax vs Minimax")
        print("6. Sair")
        op = input("Escolha: ")
        if op == "1":
            play_human_vs_minimax()
        elif op == "2":
            play_human_vs_mcts()
        elif op == "3":
            play_mcts_vs_minimax()
        elif op == "4":
            play_mcts_vs_mcts()
        elif op == "5":
            play_minimax_vs_minimax()
        elif op == "6":
            print("Saindo...")
            break
        else:
            print(Fore.RED + "Opção inválida.")

if __name__ == "__main__":
    main_menu()