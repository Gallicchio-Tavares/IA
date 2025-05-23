import itertools
import random

# quarto_game.py
class QuartoGame:
    # Atributos das peças (em ordem fixa):
    # [0] Forma: 0=quadrado/paralelepípedo, 1=redondo/cilindro
    # [1] Cor: 0=preta, 1=branca
    # [2] Altura: 0=baixa, 1=alta
    # [3] Furo: 0=sem buraco/maciça, 1=com buraco/oca
    
    def __init__(self):
        self.size = 4
        self.board = [[None for _ in range(self.size)] for _ in range(self.size)]
        # Gera todas as combinações possíveis (16 peças únicas)
        self.pieces = set(itertools.product((0, 1), (0, 1), (0, 1), (0, 1)))
        self.current_player = 'X'
        self.given_piece = None

    def piece_description(self, piece):
        """Retorna descrição textual da peça"""
        if piece is None:
            return "Nenhuma"
        shape = "Redondo" if piece[0] else "Quadrado"
        color = "Branco" if piece[1] else "Preto"
        height = "Alto" if piece[2] else "Baixo"
        hole = "Com furo" if piece[3] else "Sem furo"
        return f"{shape} {color} {height} {hole}"

    def print_board(self):
        def piece_repr(p):
            if p is None:
                return "[____]"
            piece, player = p
            # Símbolos visuais para cada atributo
            shape = '●' if piece[0] else '■'
            color = 'B' if piece[1] else 'P'
            height = 'A' if piece[2] else 'a'
            hole = '○' if piece[3] else '•'
            return f"[{shape}{color}{height}{hole}]"

        print("\n   " + "    ".join(str(c) for c in range(self.size)))
        for r in range(self.size):
            row = [piece_repr(self.board[r][c]) for c in range(self.size)]
            print(f"{r}  {' '.join(row)}")
        
        if self.given_piece:
            print(f"\nPeça a ser colocada: {self.piece_description(self.given_piece)}")
        print(f"Vez do: {'Jogador (X)' if self.current_player == 'X' else 'Robô (O)'}")
        print(f"Peças restantes: {len(self.pieces)}")
        
    def copy(self):
        """Cria uma cópia profunda do estado atual do jogo"""
        new_game = QuartoGame()
        new_game.board = [row.copy() for row in self.board]
        new_game.pieces = self.pieces.copy()
        new_game.current_player = self.current_player
        new_game.given_piece = self.given_piece
        return new_game

    def available_positions(self):
        """Retorna lista de posições vazias no tabuleiro"""
        return [(r, c) for r in range(self.size) for c in range(self.size) if self.board[r][c] is None]

    def available_pieces(self):
        """Retorna peças disponíveis para escolha (excluindo a peça atual)"""
        return [p for p in self.pieces if p != self.given_piece]

    def make_move(self, position, piece_to_give):
        """Executa um movimento no jogo"""
        r, c = position
        if self.board[r][c] is not None:
            raise ValueError("Posição já ocupada")
        
        # Armazena a peça junto com o jogador que a colocou
        self.board[r][c] = (self.given_piece, self.current_player)
        
        if self.given_piece in self.pieces:
            self.pieces.remove(self.given_piece)
        
        self.given_piece = piece_to_give
        self.current_player = 'O' if self.current_player == 'X' else 'X'

    def game_over(self):
        """Verifica se o jogo terminou (vitória ou empate)"""
        return self.winner() is not None or len(self.pieces) == 0

    def winner(self):
        lines = []
        # Adiciona todas linhas possíveis (horizontais, verticais, diagonais)
        for i in range(4):
            lines.append([self.board[i][j] for j in range(4)])  # Horizontal
            lines.append([self.board[j][i] for j in range(4)])  # Vertical
        lines.append([self.board[i][i] for i in range(4)])      # Diagonal \
        lines.append([self.board[i][3-i] for i in range(4)])    # Diagonal /

        for line in lines:
            if None in line:  # Ignora linhas incompletas
                continue
                
            pieces = [p[0] for p in line]  # Extrai apenas as peças (ignora o jogador)
            
            # Verifica se TODAS as peças compartilham algum atributo
            for attr in range(4):  # 0=forma, 1=cor, 2=altura, 3=furo
                if all(p[attr] == pieces[0][attr] for p in pieces):
                    # Retorna o jogador que colocou a ÚLTIMA peça da linha
                    return line[-1][1]  # [1] é o jogador (X ou O)
        
        return None