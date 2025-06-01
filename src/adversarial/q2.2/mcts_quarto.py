import math, random, time

class Node:
    def __init__(self, game, parent=None, move=None):
        self.game = game
        self.parent = parent
        self.move = move
        self.children = []
        self.visits = 0
        self.wins = 0
        self.untried_moves = [(r, c, i) for (r, c) in game.available_moves()
                              for i, p in enumerate(game.all_pieces)
                              if p in game.available_pieces and p != game.selected_piece]

    def ucb1(self, c=1.41):
        if self.visits == 0:
            return float('inf')
        return self.wins / self.visits + c * math.sqrt(math.log(self.parent.visits) / self.visits)

    def select(self):
        return max(self.children, key=lambda child: child.ucb1())

    def expand(self):
        move = self.untried_moves.pop()
        new_game = self.game.copy()
        new_game.make_move(move)
        child = Node(new_game, parent=self, move=move)
        self.children.append(child)
        return child

    def update(self, result):
        self.visits += 1
        self.wins += result

def quarto_mcts(game, iterations=500, time_limit=None):
    root = Node(game)
    start_time = time.time()

    for _ in range(iterations):
        if time_limit and time.time() - start_time > time_limit:
            break

        node = root
        sim_game = game.copy()

        while node.untried_moves == [] and node.children:
            node = node.select()
            sim_game.make_move(node.move)

        if node.untried_moves:
            node = node.expand()
            sim_game = node.game.copy()

        while not sim_game.game_over():
            moves = [(r, c, i) for (r, c) in sim_game.available_moves()
                     for i, p in enumerate(sim_game.all_pieces)
                     if p in sim_game.available_pieces and p != sim_game.selected_piece]
            if not moves:
                break
            move = random.choice(moves)
            sim_game.make_move(move)

        winner = sim_game.winner()
        result = 1 if winner == game.current else 0

        while node is not None:
            node.update(result)
            node = node.parent

    return max(root.children, key=lambda c: c.visits).move if root.children else None
