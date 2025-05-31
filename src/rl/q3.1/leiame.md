# Instruçoes para rodar códigos da questão 3.1

## Rodar treinamentos

O script principal de treinamento é o tql_train.py. Ele aceita parâmetros de linha de comando:

| Parâmetro         | Descrição                            |  Valor padrão  |
| ----------------- | ------------------------------------ | ------------   |
| `--env_name`      | Nome do ambiente (`Gymnasium`)       | `Blackjack-v1` |
| `--num_episodes`  | Número de episódios para treinamento | `6000`         |
| `--decay_rate`    | Taxa de decaimento de epsilon        | `0.0001`       |
| `--learning_rate` | Taxa de aprendizado (alpha)          | `0.7`          |
| `--gamma`         | Fator de desconto (gamma)            | `0.618`        |

### Blackjack

``` bash
python tql_train.py --env_name Blackjack-v1 --num_episodes 50000 --learning_rate 0.1 --gamma 0.95
```

### Cliff Walking

``` bash
python tql_train.py --env_name CliffWalking-v0 --num_episodes 10000 --learning_rate 0.5 --gamma 0.99 --decay_rate 0.001
```

### Frozen Lake

``` bash
python tql_train.py --env_name FrozenLake-v1 --num_episodes 10000 --learning_rate 0.8 --gamma 0.95
```

## Rodar testes

O script de testes é o tql_test.py.

### Blackjack

``` bash
python tql_test.py --env_name Blackjack-v1 --num_episodes 1000
```

### Cliff Walking

``` bash
python tql_test.py --env_name CliffWalking-v0 --num_episodes 1000
```

### Frozen Lake

``` bash
python tql_test.py --env_name FrozenLake-v1 --num_episodes 1000
```
