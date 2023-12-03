import asyncio
import sys
import os

from tabulate import tabulate

from poke_env.player import RandomPlayer, cross_evaluate

from players.player_1v1 import MaxBase, MaxDamage

import logging

logging.basicConfig(level=logging.CRITICAL)

from functools import wraps

# def exit_on_exception(coro):
#     @wraps(coro)
#     async def wrapper(*args, **kwargs):
#         try:
#             return await coro(*args, **kwargs)
#         except Exception as e:
#             print(f"Caught exception: {e}")
#             os._exit(1)
#     return wrapper

FORMAT = "gen1randombattle"
NUM_BATTLES = 100


# @exit_on_exception
async def main():
    try:
        player1 = RandomPlayer(battle_format=FORMAT, max_concurrent_battles=10)
        player2 = MaxBase(battle_format=FORMAT, max_concurrent_battles=10)
        player3 = MaxDamage(
            battle_format=FORMAT, max_concurrent_battles=10, with_calcs=True
        )
        # players = [player1, player2]
        # players = [player2, player3]
        players = [player1, player2, player3]

        cross_evaluation = await cross_evaluate(players, n_challenges=NUM_BATTLES)

        # Defines a header for displaying results
        table = [["-"] + [p.username for p in players]]

        # Adds one line per player with corresponding results
        for p_1, results in cross_evaluation.items():
            table.append([p_1] + [cross_evaluation[p_1][p_2] for p_2 in results])

        # Displays results in a nicely formatted table.
        print(tabulate(table))
    except Exception as e:
        print(f"Caught exception: {e}")
        # sys.exit(1)
        os._exit(1)


if __name__ == "__main__":
    # asyncio.get_event_loop().run_until_complete(main())
    asyncio.run(main())
