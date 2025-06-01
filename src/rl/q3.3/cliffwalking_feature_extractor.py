import numpy as np

from feature_extractor import FeatureExtractor

important_locations_dict = {
    'goal_location': [3, 11],  # Posição da meta no ambiente (linha 3, coluna 11).
    'cliff': [3, [1, 2, 3, 4, 5, 6, 7, 8, 9, 10]],  # Posições do "penhasco" (linha 3 e colunas de 1 a 10).
    'bottom_row': 3,  # A última linha do grid.
    'right_column': 11,  # A última coluna do grid.
    'column_start': 0,  # A primeira coluna do grid.
    'column_end': 11,  # A última coluna do grid.
    'row_start': 0,  # A primeira linha do grid.
    'row_end': 3  # A última linha do grid.
}

class Actions:
    UP = 0
    RIGHT = 1
    DOWN = 2
    LEFT = 3

class CliffWalkingFeatureExtractor(FeatureExtractor):

    __actions_one_hot_encoding = {
        Actions.UP: np.array([1, 0, 0, 0]),
        Actions.RIGHT: np.array([0, 1, 0, 0]),
        Actions.DOWN: np.array([0, 0, 1, 0]),
        Actions.LEFT: np.array([0, 0, 0, 1]),
    }

    def __init__(self, env):
        self.env = env
        self.features_list = []
        self.features_list.append(self.f0)
        self.features_list.append(self.f1)
        self.features_list.append(self.f2)
        self.features_list.append(self.f3)
        self.features_list.append(self.f4)
        self.features_list.append(self.f5)
        self.features_list.append(self.f6)
        self.features_list.append(self.f7)
        self.features_list.append(self.f8)

    def get_num_features(self):
        return len(self.features_list) + self.get_num_actions()

    def get_num_actions(self):
        return len(self.get_actions())

    def get_action_one_hot_encoded(self, action):
        return self.__actions_one_hot_encoding[action]

    def is_terminal_state(self, state):
        agent_column = state % 12
        agent_row = state // 12
        isGoal = [agent_row, agent_column] == important_locations_dict['goal_location']
        isCliff = agent_column in important_locations_dict['cliff'][1] and agent_row == important_locations_dict['cliff'][0]
        return isGoal or isCliff

    def get_actions(self):
        return [Actions.DOWN, Actions.UP, Actions.RIGHT, Actions.LEFT]

    def get_features(self, state, action):
        feature_vector = np.zeros(len(self.features_list))
        for index, feature in enumerate(self.features_list):
            feature_vector[index] = feature(state, action)
        action_vector = self.get_action_one_hot_encoded(action)
        feature_vector = np.concatenate([feature_vector, action_vector])
        return feature_vector

    def _map_state_to_position(self, state):
        '''
        Mapeia um estado dado para uma posição (linha e coluna) no grid.
        O ambiente é representado por um grid 2D de 12 colunas, e a posição é derivada do estado.
        '''
        grid_size = 12  # Número de colunas no grid.
        return [state // grid_size, state % grid_size]  # Calcula a linha e a coluna a partir do estado.

    def f0(self, state, action):
        # Característica de viés (bias) para sempre retornar 1.
        return 1.0

    def f1(self, state, action):
        """
        Característica que calcula a distância entre a posição do agente e a coluna do objetivo.
        Quanto mais próxima a coluna do agente estiver da coluna da meta, maior será o valor.
        """
        agent_location = self._map_state_to_position(state)  # Obtém a posição (linha e coluna) do agente.
        agent_row = agent_location[0]  # Linha do agente.
        # Se o agente estiver na última linha, retorna 0.0 porque é o "penhasco".
        if agent_row == important_locations_dict['bottom_row']:
            return 0.0
        goal_location = important_locations_dict['goal_location']  # Localização do objetivo.
        agent_column = agent_location[1]  # Coluna do agente.
        goal_column = goal_location[1]  # Coluna do objetivo.
        column_distance = abs(goal_column - agent_column)  # Calcula a distância entre a coluna do agente e o objetivo.
        return 1 / (column_distance + 1)  # Retorna o inverso da distância para normalização.

    def f2(self, state, action):
        """
        Característica que calcula a distância entre a posição do agente e a linha do objetivo
        quando o agente está na coluna final.
        """
        agent_location = self._map_state_to_position(state)  # Obtém a posição (linha e coluna) do agente.
        agent_column = agent_location[1]  # Coluna do agente.
        # Se o agente não estiver na última coluna, retorna 0.0.
        if agent_column != important_locations_dict['right_column']:
            return 0.0
        goal_row = important_locations_dict['goal_location'][0]  # Linha do objetivo.
        agent_row = agent_location[0]  # Linha do agente.
        row_distance = abs(goal_row - agent_row)  # Calcula a distância entre a linha do agente e o objetivo.
        return 1 / (row_distance + 1)  # Retorna o inverso da distância para normalização.

    def f3(self, state, action):
        """
        Característica que calcula a distância do agente para o "penhasco".
        Se o agente estiver fora das colunas do "penhasco", retorna 0.0.
        """
        agent_location = self._map_state_to_position(state)  # Obtém a posição (linha e coluna) do agente.
        agent_column = agent_location[1]  # Coluna do agente.
        # Se o agente não estiver nas colunas do "penhasco", retorna 0.0.
        if agent_column not in important_locations_dict['cliff'][1]:
            return 0.0
        cliff_row = important_locations_dict['cliff'][0]  # Linha do "penhasco".
        agent_row = agent_location[0]  # Linha do agente.
        row_distance = abs(cliff_row - agent_row)  # Calcula a distância entre a linha do agente e o "penhasco".
        return row_distance  # Retorna a distância como a característica.

    def f4(self, state, action):
        """
        Obtém a posição do agente (linha e coluna) a partir do estado fornecido.
        """
        agent_location = self._map_state_to_position(state)
        # Verifica se o agente está na última linha do grid.
        if agent_location[0] != important_locations_dict['bottom_row']:
            return 0.0  # Se o agente não estiver na última linha, retorna 0.0.
        return 1.0  # Se o agente estiver na última linha, retorna 1.0.

    def f5(self, state, action):
        """
        Obtém a posição do agente (linha e coluna) a partir do estado fornecido.
        """
        agent_location = self._map_state_to_position(state)  # Obtém a posição (linha e coluna) do agente.
        column = agent_location[1]  # Obtém a coluna do agente.
        row = agent_location[0]  # Obtém a linha do agente.
        # Verifica se o agente está tentando se mover para fora dos limites do grid.
        border_bump = (
            column == important_locations_dict['column_start'] and action == Actions.LEFT or
            column == important_locations_dict['column_end'] and action == Actions.RIGHT or
            row == important_locations_dict['row_start'] and action == Actions.UP or
            row == important_locations_dict['row_end'] and action == Actions.DOWN
        )
        # Retorna 1 se o agente está tentando sair do grid, 0 caso contrário.
        return int(border_bump)

    def f6(self, state, action):
        """
        Calcula a distância Euclidiana entre a posição do agente e a posição do objetivo
        """
        agent_location = self._map_state_to_position(state)
        goal_location = important_locations_dict['goal_location']
        # Calcula a distância Euclidiana entre o agente e o objetivo.
        euclidean_distance = np.sqrt((goal_location[0] - agent_location[0]) ** 2 + (goal_location[1] - agent_location[1]) ** 2)
        return 1 / (euclidean_distance + 1)  # Normaliza retornando o inverso da distância + 1.

    def f7(self, state, action):
        agent_location = self._map_state_to_position(state)
        # Verifica se o agente está na linha logo acima do penhasco e se a ação é para baixo
        if agent_location[0] == important_locations_dict['cliff'][0] - 1 and action == Actions.DOWN:
            return 1.0
        return 0.0

    def f8(self, state, action):
        agent_location = self._map_state_to_position(state)
        start_location = [important_locations_dict['bottom_row'], 0]
        # Calcula a distância Manhattan da posição inicial
        distance_from_start = abs(agent_location[0] - start_location[0]) + abs(agent_location[1] - start_location[1])
        return 1 / (distance_from_start + 1)  # Normaliza retornando o inverso da distância + 1.
