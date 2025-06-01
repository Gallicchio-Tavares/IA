import numpy as np
from feature_extractor import FeatureExtractor

class Actions:
    STICK = 0
    HIT = 1

class BlackjackFeatureExtractor(FeatureExtractor):
    
    __actions_one_hot_encoding = {
        Actions.STICK: np.array([1, 0]),
        Actions.HIT: np.array([0, 1])
    }

    def __init__(self, env):
        self.env = env
        self.features_list = []
        self.features_list.append(self.f0)  # termo de bias
        self.features_list.append(self.f1)  # soma total das cartas do jogador
        self.features_list.append(self.f2)  # sobre o uso de um ás
        self.features_list.append(self.f3)  # sobre a distância do dealer até 21
        self.features_list.append(self.f4)  # se a carta do dealer é um Ás
        self.features_list.append(self.f5)  # se a soma do jogador é pelo menos 17
        self.features_list.append(self.f6)  # se a soma do jogador é 10 ou 11
        self.features_list.append(self.f7)  # se o jogador tem um blackjack (som=21)
        self.features_list.append(self.f8)  # do impacto do uso de um Ás na distância do jogador até 21
        self.features_list.append(self.f9)  # probabilidade de estourar ao comprar carta
        self.features_list.append(self.f10) # se a carta do dealer é alta (10, J, Q, K)

    def get_num_features(self):
        '''
        Retorna o número de features extraídas pelo extrator de features.
        '''
        return len(self.features_list) + self.get_num_actions()

    def get_num_actions(self):
        '''
        Retorna o número de ações disponíveis no ambiente.
        '''
        return len(self.get_actions())

    def get_action_one_hot_encoded(self, action):
        '''
        Retorna a representação one-hot codificada de uma ação.
        '''
        return self.__actions_one_hot_encoding[action]

    def is_terminal_state(self, state):
        '''
        Verifica se o estado é terminal (fim do jogo).
        '''
        if state[2] == True:
            return True
        elif state[0] > 21:
            return True
        return False

    def get_actions(self):
        '''
        Retorna uma lista de ações disponíveis no ambiente.
        '''
        return [Actions.STICK, Actions.HIT]

    def get_features(self, state, action):
        '''
        Toma um estado e uma ação como entrada e retorna o vetor de features
        para aquele par estado-ação. Chama os métodos de extração de features.
        '''
        feature_vector = np.zeros(len(self.features_list))
        for index, feature in enumerate(self.features_list):
            feature_vector[index] = feature(state, action)

        action_vector = self.get_action_one_hot_encoded(action)
        feature_vector = np.concatenate([feature_vector, action_vector])

        return feature_vector

    def _map_state_to_position(self, state):
        grid_size = 12
        return [state // grid_size, state % grid_size]

    def f0(self, state, action):
        return 1.0

    def f1(self, state, action):
        player_sum, dealer_card, usable_ace = state
        distance_to_21 = 21 - player_sum
        if distance_to_21 < 0:
            return 0.0
        return 1 / (distance_to_21 + 1)

    def f2(self, state, action):
        player_sum, dealer_card, usable_ace = state
        return 1.0 if usable_ace else 0.0

    def f3(self, state, action):
        player_sum, dealer_card, usable_ace = state
        dealer_distance_to_21 = abs(21 - dealer_card)
        return 1 / (dealer_distance_to_21 + 1)

    def f4(self, state, action):
        player_sum, dealer_card, usable_ace = state
        if dealer_card == 1:
            return 1
        return 0

    def f5(self, state, action):
        player_sum, dealer_card, usable_ace = state
        return 1 if player_sum >= 17 else 0

    def f6(self, state, action):
        player_sum, dealer_card, usable_ace = state
        return 1 if player_sum == 10 or player_sum == 11 else 0

    def f7(self, state, action):
        player_sum, dealer_card, usable_ace = state
        if player_sum == 21:
            return 1
        return 0

    def f8(self, state, action):
        player_sum, dealer_card, usable_ace = state
        if not usable_ace:
            return 0
        player_distance_using_ace = 21 - player_sum - 11
        if player_distance_using_ace < 0:
            return 0
        return 1 / (player_distance_using_ace + 11)

    def f9(self, state, action):
        player_sum, dealer_card, usable_ace = state
        if player_sum >= 17:
            return (player_sum - 16) / 5
        return 0

    def f10(self, state, action):
        player_sum, dealer_card, usable_ace = state
        if dealer_card in [10, 11, 12, 13]:  # cartas altas
            return 1
        return 0
