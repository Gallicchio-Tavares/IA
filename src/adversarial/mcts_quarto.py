import math
import random
from quarto_game import QuartoGame

class MCTSNode:
    def __init__(self, game, parent=None, move=None):
        self.game = game
        self.parent = parent
        self.move = move
        self.children = []
        self.visits = 0
        self.wins = 0
        self.untried_moves = [
            (pos, piece) for pos in game.available_positions() for piece in game.available_pieces()
        ]

    def ucb1(self, c=math.sqrt(2)):
        if self.visits == 0:
            return float('inf')
        return self.wins / self.visits + c * math.sqrt(math.log(self.parent.visits) / self.visits)

    def select_child(self):
        return max(self.children, key=lambda child: child.ucb1())

    def expand(self):
        move = self.untried_moves.pop()
        new_game = self.game.copy()
        new_game.make_move(*move)
        child = MCTSNode(new_game, parent=self, move=move)
        self.children.append(child)
        return child

    def update(self, result):
        self.visits += 1
        self.wins += result

# mcts_quarto.py
def mcts_agent(game, iterations=500):
    root = MCTSNode(game)

    for _ in range(iterations):
        node = root
        game_sim = game.copy()

        # Selection
        while node.untried_moves == [] and node.children:
            node = node.select_child()
            game_sim.make_move(*node.move)

        # Expansion
        if node.untried_moves:
            move = random.choice(node.untried_moves)
            game_sim.make_move(*move)
            node = node.expand()

        # Simulation
        while not game_sim.game_over():
            available_pos = game_sim.available_positions()
            available_pieces = game_sim.available_pieces()
            if not available_pos or not available_pieces:
                break
                
            pos = random.choice(available_pos)
            piece = random.choice(available_pieces)
            game_sim.make_move(pos, piece)

        # Backpropagation
        winner = game_sim.winner()
        result = 1 if winner == 'O' else 0  # Assumindo que o MCTS joga como 'O'

        while node is not None:
            node.update(result)
            node = node.parent

    if not root.children:
        # Caso de emergência - escolhe qualquer movimento válido
        available_pos = game.available_positions()
        available_pieces = game.available_pieces()
        if available_pos and available_pieces:
            return (random.choice(available_pos), random.choice(available_pieces))
        else:
            raise ValueError("Nenhum movimento válido disponível")

    best_child = max(root.children, key=lambda c: c.visits)
    return best_child.move
