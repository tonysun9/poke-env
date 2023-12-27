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

import argparse

# from poke_env.player.gymnasium_env import _AsyncPlayer

# from rl.sb_example import SimpleRLPlayer


import logging

logging.basicConfig(level=logging.CRITICAL)

BATTLE_FORMAT = "gen1randombattle"
NUM_BATTLES = 100
MODEL_PATH = "rl/models/"
MODEL_NAME = "obs_9_4"

OBSERVATION_TYPE = "matrix"

SAVE_REPLAYS = False


async def main(
    model_name: str,
    num_battles: int,
    save_replays: bool,
    obs_type: str,
):
    obs_class = None
    if obs_type == "full":
        obs_class = ObservationFull
    elif obs_type == "incremental":
        obs_class = ObservationIncremental
    elif obs_type == "matrix":
        obs_class = ObservationMatrix
    if not obs_class:
        raise ValueError(f"Observation type not set")

    dqn_player = DQNPlayer(
        battle_format=BATTLE_FORMAT,
        model_path=f"{MODEL_PATH}/{model_name}",
        observation_class=obs_class,
        save_replays=save_replays,
        replay_folder=f"rl/replays/{model_name}",
        account_configuration=AccountConfiguration(model_name, ""),
    )
    # dqn_player_2 = DQNPlayer(
    #     battle_format=BATTLE_FORMAT,
    #     model_path=f"{MODEL_PATH}/obs_9_3",
    #     observation_class=obs_class,
    #     save_replays=save_replays,
    #     replay_folder=f"rl/replays/{model_name}",
    #     account_configuration=AccountConfiguration("dqn93", ""),
    # )
    # test_dqn_player = TestDQNPlayer(
    #     battle_format=BATTLE_FORMAT,
    #     model_path=MODEL_PATH,
    #     observation_class=ObservationIncremental,
    # )

    # try:
    random_player = RandomPlayer(
        battle_format=BATTLE_FORMAT,
        max_concurrent_battles=10,
        account_configuration=AccountConfiguration("Random Player", ""),
    )
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

    # players = [random_player, test_dqn_player]

    cross_evaluation = await cross_evaluate(players, n_challenges=num_battles)

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
    parser = argparse.ArgumentParser(
        description="Evaluate a reinforcement learning model"
    )
    parser.add_argument(
        "--model_name",
        "-m",
        type=str,
        default=MODEL_NAME,
        help="Name of the model to be trained",
    )
    parser.add_argument(
        "--num_battles",
        "-n",
        type=int,
        default=NUM_BATTLES,
        help="Number of battles to evaluate",
    )
    parser.add_argument(
        "--save_replays",
        "-s",
        action="store_true",
        default=SAVE_REPLAYS,
        help="Whether to save replays of the battles",
    )
    parser.add_argument(
        "--observation_type",
        "-o",
        type=str,
        default=OBSERVATION_TYPE,
        help="Type of observation to use",
    )

    args = parser.parse_args()

    asyncio.run(
        main(
            model_name=args.model_name,
            num_battles=args.num_battles,
            save_replays=args.save_replays,
            obs_type=args.observation_type,
        )
    )
