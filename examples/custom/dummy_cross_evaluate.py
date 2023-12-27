import asyncio
import sys
import os

from tabulate import tabulate

from poke_env.player import RandomPlayer, cross_evaluate

from players.player_singles import (
    MaxBase,
    MaxDamage,
    DQNPlayer,
    TestDQNPlayer,
    # DQNPlayerObservation,
)

from poke_env.ps_client.account_configuration import AccountConfiguration

from rl.observation import (
    ObservationSimple,
    ObservationFull,
    ObservationIncremental,
    ObservationMatrix,
)

# from poke_env.player.gymnasium_env import _AsyncPlayer

# from rl.sb_example import SimpleRLPlayer


import logging

logging.basicConfig(level=logging.CRITICAL)

BATTLE_FORMAT = "gen1randombattle"
NUM_BATTLES = 1
MODEL_PATH = "rl/models/"
MODEL_NAME = "observation_0_2_1e5"


async def main():
    test_dqn_player = TestDQNPlayer(
        battle_format=BATTLE_FORMAT,
        model_path=MODEL_PATH,
        observation_class=ObservationMatrix,
    )

    # try:
    random_player = RandomPlayer(
        battle_format=BATTLE_FORMAT,
        max_concurrent_battles=10,
        account_configuration=AccountConfiguration("Random Player", ""),
    )

    players = [random_player, test_dqn_player]

    cross_evaluation = await cross_evaluate(players, n_challenges=NUM_BATTLES)

    # Defines a header for displaying results
    table = [["-"] + [p.username for p in players]]

    # Adds one line per player with corresponding results
    for p_1, results in cross_evaluation.items():
        table.append([p_1] + [cross_evaluation[p_1][p_2] for p_2 in results])

    # Displays results in a nicely formatted table.
    print(tabulate(table))
    # except Exception as e:
    #     print(f"Caught exception: {e}")
    #     # sys.exit(1)
    #     os._exit(1)


if __name__ == "__main__":
    asyncio.run(main())
