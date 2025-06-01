import numpy as np

from feature_extractor import FeatureExtractor

important_locations_dict = {
    'goal_loc': [3, 11],
    'cliff': [3, [1, 2, 3, 4, 5, 6, 7, 8, 9, 10]],
    'bottom_row': 3,
    'right_column': 11,
    'column_start': 0,
    'column_end': 11,
    'row_start': 0,
    'row_end': 3
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
        isGoal = [agent_row, agent_column] == important_locations_dict['goal_loc']
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
        grid_size = 12 
        return [state // grid_size, state % grid_size]

    def f0(self, state, action):
        return 1.0

    def f1(self, state, action):
        """
        Característica que calcula a distância entre a posição do agente e a coluna do objetivo.
        Quanto mais próxima a coluna do agente estiver da coluna da meta, maior será o valor.
        """
        agent_loc = self._map_state_to_position(state)
        agent_row = agent_loc[0]
        
        if agent_row == important_locations_dict['bottom_row']:
            return 0.0
        goal_loc = important_locations_dict['goal_loc']
        agent_column = agent_loc[1]
        goal_column = goal_loc[1]
        column_distance = abs(goal_column - agent_column)
        return 1 / (column_distance + 1)

    def f2(self, state, action):
        """
        Característica que calcula a distância entre a posição do agente e a linha do objetivo
        quando o agente está na coluna final.
        """
        agent_loc = self._map_state_to_position(state)
        agent_column = agent_loc[1]
        
        if agent_column != important_locations_dict['right_column']:
            return 0.0
        goal_row = important_locations_dict['goal_loc'][0]
        agent_row = agent_loc[0]
        row_distance = abs(goal_row - agent_row)
        return 1 / (row_distance + 1)

    def f3(self, state, action):
        """
        Característica que calcula a distância do agente para o "penhasco".
        Se o agente estiver fora das colunas do "penhasco", retorna 0.0.
        """
        agent_loc = self._map_state_to_position(state)
        agent_column = agent_loc[1]
        
        if agent_column not in important_locations_dict['cliff'][1]:
            return 0.0
        cliff_row = important_locations_dict['cliff'][0]
        agent_row = agent_loc[0]
        row_distance = abs(cliff_row - agent_row)
        return row_distance

    def f4(self, state, action):
        """
        Obtém a posição do agente (linha e coluna) a partir do estado fornecido.
        """
        agent_loc = self._map_state_to_position(state)
        
        if agent_loc[0] != important_locations_dict['bottom_row']:
            return 0.0
        return 1.0

    def f5(self, state, action):
        """
        Obtém a posição do agente (linha e coluna) a partir do estado fornecido.
        """
        agent_loc = self._map_state_to_position(state)
        column = agent_loc[1]
        row = agent_loc[0]
        
        border_bump = (
            column == important_locations_dict['column_start'] and action == Actions.LEFT or
            column == important_locations_dict['column_end'] and action == Actions.RIGHT or
            row == important_locations_dict['row_start'] and action == Actions.UP or
            row == important_locations_dict['row_end'] and action == Actions.DOWN
        )
       
        return int(border_bump)

    def f6(self, state, action):
        """
        calcula a dist Euclidiana entre a posição do agente e a posição do objetivo
        """
        agent_loc = self._map_state_to_position(state)
        goal_loc = important_locations_dict['goal_loc']
       
        dist_eucl = np.sqrt((goal_loc[0] - agent_loc[0]) ** 2 + (goal_loc[1] - agent_loc[1]) ** 2)
        return 1 / (dist_eucl + 1)

    def f7(self, state, action):
        agent_loc = self._map_state_to_position(state)
        
        if agent_loc[0] == important_locations_dict['cliff'][0] - 1 and action == Actions.DOWN:
            return 1.0
        return 0.0

    def f8(self, state, action):
        agent_loc = self._map_state_to_position(state)
        start_location = [important_locations_dict['bottom_row'], 0]
        
        distance_from_start = abs(agent_loc[0] - start_location[0]) + abs(agent_loc[1] - start_location[1])
        return 1 / (distance_from_start + 1)
