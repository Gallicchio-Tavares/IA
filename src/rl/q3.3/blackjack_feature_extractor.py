import numpy as np
from feature_extractor import FeatureExtractor

# Definição de ações possíveis no Blackjack
class Actions:
    STICK = 0  # Parar
    HIT = 1    # Comprar carta

class BlackjackFeatureExtractor(FeatureExtractor):
    # Codificação one-hot para ações
    __actions_one_hot_encoding = {
        Actions.STICK: np.array([1, 0]),  # Codificação one-hot para "parar"
        Actions.HIT: np.array([0, 1])    # Codificação one-hot para "comprar carta"
    }

    def __init__(self, env):
        '''
        Inicializa o objeto BlackjackFeatureExtractor.
        Adiciona métodos de extração de features na lista de features.
        '''
        self.env = env
        self.features_list = []
        self.features_list.append(self.f0)  # Termo de viés (bias)
        self.features_list.append(self.f1)  # Característica relacionada à soma total das cartas do jogador
        self.features_list.append(self.f2)  # Característica sobre o uso de um ás
        self.features_list.append(self.f3)  # Característica sobre a distância do dealer até 21
        self.features_list.append(self.f4)  # Característica se a carta do dealer é um Ás
        self.features_list.append(self.f5)  # Característica se a soma do jogador é pelo menos 17
        self.features_list.append(self.f6)  # Característica se a soma do jogador é 10 ou 11
        self.features_list.append(self.f7)  # Característica se o jogador tem um blackjack (som=21)
        self.features_list.append(self.f8)  # Característica do impacto do uso de um Ás na distância do jogador até 21
        self.features_list.append(self.f9)  # Característica de probabilidade de estourar ao comprar carta
        self.features_list.append(self.f10) # Característica se a carta do dealer é alta (10, J, Q, K)

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
            return True  # Estado terminal se o jogo acabou
        elif state[0] > 21:
            return True  # Estado terminal se o jogador estourou
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
            feature_vector[index] = feature(state, action)  # Calcula cada feature para o estado-ação dado

        action_vector = self.get_action_one_hot_encoded(action)
        feature_vector = np.concatenate([feature_vector, action_vector])  # Combina features com a codificação one-hot da ação

        return feature_vector

    def _map_state_to_position(self, state):
        '''
        Mapeia o estado para uma posição na grade, usado para jogos com estados discretos.
        '''
        grid_size = 12
        return [state // grid_size, state % grid_size]

    def f0(self, state, action):
        '''
        Termo de viés (bias).
        '''
        return 1.0

    def f1(self, state, action):
        '''
        Feature: Soma total das cartas do jogador.
        Retorna uma inversa da distância para 21 para capturar quão perto o jogador está de estourar.
        '''
        player_sum, dealer_card, usable_ace = state
        distance_to_21 = 21 - player_sum
        if distance_to_21 < 0:
            return 0.0  # Se o jogador estourou, a feature é 0
        return 1 / (distance_to_21 + 1)  # Quanto mais perto de 21, maior a feature

    def f2(self, state, action):
        '''
        Feature: Ás utilizável.
        '''
        player_sum, dealer_card, usable_ace = state
        return 1.0 if usable_ace else 0.0

    def f3(self, state, action):
        '''
        Feature: Distância do dealer até 21.
        '''
        player_sum, dealer_card, usable_ace = state
        dealer_distance_to_21 = abs(21 - dealer_card)
        return 1 / (dealer_distance_to_21 + 1)  # Inversa da distância para 21 para o dealer

    def f4(self, state, action):
        '''
        Feature: A carta do dealer é um Ás.
        '''
        player_sum, dealer_card, usable_ace = state
        if dealer_card == 1:
            return 1
        return 0

    def f5(self, state, action):
        '''
        Feature: A soma do jogador é pelo menos 17.
        '''
        player_sum, dealer_card, usable_ace = state
        return 1 if player_sum >= 17 else 0

    def f6(self, state, action):
        '''
        Feature: A soma do jogador é 10 ou 11.
        '''
        player_sum, dealer_card, usable_ace = state
        return 1 if player_sum == 10 or player_sum == 11 else 0

    def f7(self, state, action):
        '''
        Feature: O jogador tem um blackjack (soma é 21).
        '''
        player_sum, dealer_card, usable_ace = state
        if player_sum == 21:
            return 1
        return 0

    def f8(self, state, action):
        '''
        Feature: Impacto do uso de um Ás na distância do jogador até 21.
        '''
        player_sum, dealer_card, usable_ace = state
        if not usable_ace:
            return 0  # Se o jogador não tem um Ás utilizável, a feature é 0
        player_distance_using_ace = 21 - player_sum - 11
        if player_distance_using_ace < 0:
            return 0
        return 1 / (player_distance_using_ace + 11)

    def f9(self, state, action):
        '''
        Feature: Probabilidade de estourar ao comprar carta.
        '''
        player_sum, dealer_card, usable_ace = state
        if player_sum >= 17:
            return (player_sum - 16) / 5  # Proporção da soma do jogador acima de 16
        return 0

    def f10(self, state, action):
        '''
        Feature: A carta do dealer é alta (10, J, Q, K).
        '''
        player_sum, dealer_card, usable_ace = state
        if dealer_card in [10, 11, 12, 13]:  # 10, J, Q, K são todas cartas altas
            return 1
        return 0
