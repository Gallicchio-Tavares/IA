import numpy as np

from feature_extractor import FeatureExtractor

important_locations_dict = {
    'goal_loc': [3, 3],
    'holes': [(3, 0), (1, 1), (1, 3), (2, 3)],
    'column_start': 0,
    'column_end': 3,
    'row_start': 0,
    'row_end': 3
}

class Actions:
    UP = 0
    RIGHT = 1
    DOWN = 2
    LEFT = 3

class FrozenLakeFeatureExtractor(FeatureExtractor):

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
        self.features_list.append(self.f9)

    def get_num_features(self):
        return len(self.features_list) + self.get_num_actions()

    def get_num_actions(self):
        return len(self.get_actions())

    def get_action_one_hot_encoded(self, action):
        return self.__actions_one_hot_encoding[action]

    def is_terminal_state(self, state):
        agent_loc = self._map_state_to_position(state)
        holes = important_locations_dict['holes']
        is_holes = agent_loc in holes
        is_goal = agent_loc == important_locations_dict['goal_loc']
        return is_holes or is_goal

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
        grid_size = 4
        return [state // grid_size, state % grid_size]

    def f0(self, state, action):
        return 1.0

    def f1(self, state, action):
        agent_loc = self._map_state_to_position(state)
        goal_manhattan_distance = self._manhattanDistance(
            agent_loc, important_locations_dict['goal_loc']
        )
        return 1 / (goal_manhattan_distance + 1)

    def f2(self, state, action):
        agent_loc = self._map_state_to_position(state)
        holes_manhattan_distance = [
            self._manhattanDistance(agent_loc, hole) for hole in important_locations_dict['holes']
        ]
        return 1 / (min(holes_manhattan_distance) + 1)

    def f3(self, state, action):
        agent_loc = self._map_state_to_position(state)
        min_distance_to_wall = min(
            agent_loc[0], 3 - agent_loc[0], agent_loc[1], 3 - agent_loc[1]
        )
        return 1 / (min_distance_to_wall + 1)

    def _manhattanDistance(self, loc1, loc2):
        return abs(loc1[0] - loc2[0]) + abs(loc1[1] - loc2[1])

    def f4(self, state, action):
        agent_loc = self._map_state_to_position(state)
        action_dir = self.get_actions()[action]
        new_agent_location = agent_loc.copy()

        if action_dir == Actions.DOWN:
            new_agent_location[0] += 1
        elif action_dir == Actions.UP:
            new_agent_location[0] -= 1
        elif action_dir == Actions.RIGHT:
            new_agent_location[1] += 1
        elif action_dir == Actions.LEFT:
            new_agent_location[1] -= 1

        if new_agent_location[0] < 0 or new_agent_location[0] > 3:
            return 0
        if new_agent_location[1] < 0 or new_agent_location[1] > 3:
            return 0

        holes_manhattan_distance = [
            self._manhattanDistance(new_agent_location, hole) for hole in important_locations_dict['holes']
        ]
        if min(holes_manhattan_distance) == 0:
            return 1
        return 0

    def f5(self, state, action):
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
        agent_loc = self._map_state_to_position(state)
        goal_loc = important_locations_dict['goal_loc']
        euclidean_distance = np.sqrt((agent_loc[0] - goal_loc[0])**2 + (agent_loc[1] - goal_loc[1])**2)
        return 1 / (euclidean_distance + 1)

    def f7(self, state, action):
        agent_loc = self._map_state_to_position(state)
        goal_loc = important_locations_dict['goal_loc']

        steps_to_goal = abs(agent_loc[0] - goal_loc[0]) + abs(agent_loc[1] - goal_loc[1])
        return 1 / (steps_to_goal + 1)

    def f8(self, state, action):
        agent_loc = self._map_state_to_position(state)
        new_loc = agent_loc.copy()

        if action == Actions.UP:
            new_loc[0] -= 1
        elif action == Actions.DOWN:
            new_loc[0] += 1
        elif action == Actions.LEFT:
            new_loc[1] -= 1
        elif action == Actions.RIGHT:
            new_loc[1] += 1

        if new_loc == agent_loc or not (0 <= new_loc[0] <= 3 and 0 <= new_loc[1] <= 3):
            return 1
        return 0

    def f9(self, state, action):
        agent_loc = self._map_state_to_position(state)
        goal_loc = important_locations_dict['goal_loc']
        holes = important_locations_dict['holes']

        path_clear = True

        if agent_loc[0] == goal_loc[0]:
            min_col, max_col = sorted([agent_loc[1], goal_loc[1]])
            for col in range(min_col + 1, max_col):
                if (agent_loc[0], col) in holes:
                    path_clear = False
                    break
        elif agent_loc[1] == goal_loc[1]:
            min_row, max_row = sorted([agent_loc[0], goal_loc[0]])
            for row in range(min_row + 1, max_row):
                if (row, agent_loc[1]) in holes:
                    path_clear = False
                    break
        else:
            path_clear = False

        return 1 if path_clear else 0

    def manhattanDistance(self, xy1, xy2):
        """
        Computes the Manhattan distance between two points.
        """
        return abs(xy1[0] - xy2[0]) + abs(xy1[1] - xy2[1])
