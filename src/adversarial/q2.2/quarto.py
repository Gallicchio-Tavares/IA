from board_game import BoardGame
from colorama import Fore, Style, init

init(autoreset=True)

class Quarto(BoardGame):
    def __init__(self):
        super().__init__(4, 4)
        self.all_pieces = [(a, b, c, d) for a in (0, 1) for b in (0, 1) for c in (0, 1) for d in (0, 1)]
        self.available_pieces = self.all_pieces.copy()
        self.selected_piece = None
        self.pieces_on_board = []
        self.current = 0  # 0 = humano, 1 = IA

    def piece_to_str(self, piece):
        if piece == ' ' or piece is None:
            return "    "
        shape = "●" if piece[0] else "■"
        color = Fore.RED if piece[1] else Fore.BLUE
        height = "▲" if piece[2] else "▼"
        hole = "○" if piece[3] else "•"
        return f"{color}{shape}{height}{hole}{Style.RESET_ALL}"


    def available_moves(self):
        return [(r, c) for r in range(4) for c in range(4) if self.board[r][c] == ' ']

    def make_move(self, move):
        row, col, next_piece_idx = move

        if not (0 <= row < 4) or not (0 <= col < 4):
            raise ValueError("Posição fora do tabuleiro.")

        if (row, col) not in self.available_moves():
            raise ValueError("Posição já ocupada.")

        if next_piece_idx is not None:
            if not (0 <= next_piece_idx < len(self.all_pieces)):
                raise ValueError("Índice da peça inválido.")
            if self.all_pieces[next_piece_idx] not in self.available_pieces:
                raise ValueError("Peça já foi utilizada.")

        self.board[row][col] = self.selected_piece
        self.pieces_on_board.append(self.selected_piece)
        if self.selected_piece in self.available_pieces:
            self.available_pieces.remove(self.selected_piece)

        if next_piece_idx is not None:
            self.selected_piece = self.all_pieces[next_piece_idx]
            if self.selected_piece in self.available_pieces:
                self.available_pieces.remove(self.selected_piece)
        else:
            self.selected_piece = None

        self.current = 1 - self.current
        return True

    def winner(self):
        lines = []

        for i in range(4):
            lines.append([self.board[i][j] for j in range(4)])
            lines.append([self.board[j][i] for j in range(4)])

        lines.append([self.board[i][i] for i in range(4)])
        lines.append([self.board[i][3 - i] for i in range(4)])

        for line in lines:
            if any(cell == ' ' or cell is None for cell in line):
                continue

            pieces = [cell for cell in line if cell != ' ' and cell is not None]

            for attr in range(4):
                if all(p[attr] == pieces[0][attr] for p in pieces):
                    return 1 if self.current == 0 else 0

        return None

    def game_over(self):
        return self.winner() is not None or (not self.available_moves())

    def print_board(self):
        print(Fore.CYAN + "\n    0       1       2       3")
        for r in range(4):
            row = []
            for c in range(4):
                if self.board[r][c] == ' ':
                    row.append("[    ]")
                else:
                    row.append(f"[{self.piece_to_str(self.board[r][c])}]")
            print(Fore.CYAN + f"{r}  " + ' '.join(row))
        print(Fore.YELLOW + f"\nPeça atual: {self.piece_to_str(self.selected_piece) if self.selected_piece else 'Nenhuma'}")
        print(Fore.MAGENTA + f"Jogador atual: {'Humano (0)' if self.current == 0 else 'IA (1)'}")

    def print_available_pieces(self):
        print(Fore.CYAN + "\nPeças disponíveis:")
        pieces_with_idx = [(i, p) for i, p in enumerate(self.all_pieces) if p in self.available_pieces]

        for idx, (i, p) in enumerate(pieces_with_idx):
            shape = "●" if p[0] else "■"
            color = Fore.RED if p[1] else Fore.BLUE
            height = "▲" if p[2] else "▼"
            hole = "○" if p[3] else "•"
            piece_str = f"{color}{shape}{height}{hole}{Style.RESET_ALL}"
            print(f"[{i:2}] {piece_str}", end="   ")

            if (idx + 1) % 4 == 0:
                print()
        print()
