from quarto import Quarto
from minimax_quarto import best_move_quarto
from mcts_quarto import quarto_mcts
from colorama import Fore, init
import random

init(autoreset=True)


def end_game_message(game):
    game.print_board()
    winner = game.winner()
    if winner is not None:
        print(Fore.GREEN + f"\nJogador {winner} venceu!")
    else:
        print(Fore.GREEN + "\nEmpate!")


def human_move(game):
    while True:
        try:
            row = int(input("Linha (0-3): "))
            col = int(input("Coluna (0-3): "))

            if not (0 <= row <= 3) or not (0 <= col <= 3):
                print(Fore.RED + "Posição fora do tabuleiro. Tente novamente.")
                continue

            if (row, col) not in game.available_moves():
                print(Fore.RED + "Posição já ocupada. Tente novamente.")
                continue

            game.print_available_pieces()
            next_piece = int(input("Índice da peça para o oponente: "))

            if not (0 <= next_piece < len(game.all_pieces)):
                print(Fore.RED + "Índice de peça inválido.")
                continue

            if game.all_pieces[next_piece] not in game.available_pieces:
                print(Fore.RED + "Essa peça já foi usada. Escolha outra.")
                continue

            return (row, col, next_piece)

        except ValueError:
            print(Fore.RED + "Entrada inválida. Digite números inteiros válidos.")


def play_human_vs_ai(ai_function, ai_name="IA"):
    game = Quarto()
    start_piece = random.choice(game.available_pieces)
    game.selected_piece = start_piece
    game.available_pieces.remove(start_piece)

    while not game.game_over():
        game.print_board()

        if game.current == 0:
            move = human_move(game)
        else:
            print(Fore.BLUE + f"{ai_name} pensando...")
            move = ai_function(game)
            print(Fore.BLUE + f"{ai_name} jogou {move}")

        try:
            game.make_move(move)
        except ValueError as e:
            print(Fore.RED + str(e))

    end_game_message(game)


def main_menu():
    while True:
        print(Fore.CYAN + "\n==== MENU QUARTO ====")
        print("1. Humano vs Minimax")
        print("2. Humano vs MCTS")
        print("3. Sair")
        op = input("Escolha: ")
        if op == "1":
            play_human_vs_ai(lambda g: best_move_quarto(g, depth=2), ai_name="Minimax")
        elif op == "2":
            play_human_vs_ai(lambda g: quarto_mcts(g, iterations=1000, time_limit=2), ai_name="MCTS")
        elif op == "3":
            print("Saindo...")
            break
        else:
            print(Fore.RED + "Opção inválida.")


if __name__ == "__main__":
    main_menu()
