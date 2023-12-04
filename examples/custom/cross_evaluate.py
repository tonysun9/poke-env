import asyncio
import sys
import os

from tabulate import tabulate

from poke_env.player import RandomPlayer, cross_evaluate

from players.player_singles import MaxBase, MaxDamage, DQNPlayer

# from poke_env.player.gymnasium_env import _AsyncPlayer

# from rl.sb_example import SimpleRLPlayer


import logging

logging.basicConfig(level=logging.CRITICAL)

BATTLE_FORMAT = "gen1randombattle"
NUM_BATTLES = 100

MODEL_PATH = "rl/models/dqn_0_1"


async def main():
    dqn_player = DQNPlayer(
        battle_format=BATTLE_FORMAT,
        model_path=MODEL_PATH,
        # save_replays=True,
        # replay_folder="rl/replays/v0.1",
    )

    # try:
    random_player = RandomPlayer(battle_format=BATTLE_FORMAT, max_concurrent_battles=10)
    mb_player = MaxBase(battle_format=BATTLE_FORMAT, max_concurrent_battles=10)
    md_player = MaxDamage(
        battle_format=BATTLE_FORMAT, max_concurrent_battles=10, with_calcs=True
    )
    # players = [random_player, mb_player]
    # players = [mb_player, md_player]
    # players = [random_player, mb_player, md_player]

    # players = [random_player, dqn_player]
    # players = [mb_player, dqn_player]
    # players = [md_player, dqn_player]
    players = [random_player, mb_player, md_player, dqn_player]

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
    # asyncio.get_event_loop().run_until_complete(main())
    asyncio.run(main())
