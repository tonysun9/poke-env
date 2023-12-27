import asyncio

import json

import numpy as np
from typing import Optional

from poke_env.environment.move import Move
from poke_env.environment.pokemon import Pokemon


# from gym.spaces import Box, Space
# from gym.utils.env_checker import check_env
from gymnasium.spaces import Box, Space
from gymnasium.utils.env_checker import check_env
import torch
import torch.nn as nn

import torch.nn.functional as F

from tabulate import tabulate

from poke_env.environment.abstract_battle import AbstractBattle
from poke_env.player import (
    # Gen8EnvSinglePlayer,
    MaxBasePowerPlayer,
    RandomPlayer,
    SimpleHeuristicsPlayer,
    background_cross_evaluate,
    background_evaluate_player,
)
from poke_env.player.gymnasium_env_player import Gen1EnvSinglePlayer

from poke_env.data import GenData

import wandb
from wandb.integration.sb3 import WandbCallback

# from ..players.player_singles import TestDQNPlayer

from stable_baselines3 import DQN
from stable_baselines3.common.env_util import make_vec_env
from stable_baselines3.common.evaluation import evaluate_policy

from observation import ObservationFull, ObservationIncremental, ObservationMatrix

import sys

sys.path.append("..")

from players.player_singles import DQNPlayer

import argparse

BATTLE_FORMAT = "gen1randombattle"

MODEL_NAME = "obs_test"
TRAIN_STEPS = int(1e5)
BUFFER_SIZE = int(5e4)
BATCH_SIZE = 256
LEARNING_RATE = 0.00001
EXPLORATION_FRACTION = 1.0
LEARNING_STARTS = 0

DQN_OPPONENT = "random"


class SimpleRLPlayer(Gen1EnvSinglePlayer):
    def __init__(self, **kwargs):
        super().__init__(**kwargs)

    def calc_reward(self, last_battle, current_battle) -> float:
        # return self.reward_computing_helper(
        #     current_battle, fainted_value=2.0, hp_value=1.0, victory_value=30.0
        # )
        return self.reward_computing_helper(current_battle, victory_value=100.0)

    def embed_battle(self, battle: AbstractBattle):
        return ObservationMatrix.embed_battle(battle)

    def describe_embedding(self) -> Space:
        # return Box(
        #     low=np.full(ObservationFull.size, -1),
        #     high=np.full(ObservationFull.size, 6),
        #     shape=(ObservationFull.size,),
        #     dtype=np.float16,
        # )
        return Box(
            low=np.full((6, 3361), -1),
            high=np.full((6, 3361), 6),
            shape=(6, 3361),
            dtype=np.float16,
        )


async def main(
    model_name: str,
    buffer_size: int,
    batch_size: int,
    learning_rate: float,
    exploration_fraction: float,
    learning_starts: int,
    train_steps: int,
    self_play: str,
):
    # Create one environment for training and one for evaluation
    if self_play == "random":
        opponent = RandomPlayer(battle_format=BATTLE_FORMAT)
    else:
        # opponent = MaxBasePowerPlayer(battle_format=BATTLE_FORMAT)
        opponent = DQNPlayer(
            battle_format=BATTLE_FORMAT,
            model_path=f"/Users/tonysun/pokemon/poke-env/examples/custom/rl/models/{self_play}",
            observation_class=ObservationMatrix,
        )

    train_env = SimpleRLPlayer(
        battle_format=BATTLE_FORMAT,
        opponent=opponent,
        start_challenging=True,
    )

    if self_play == "random":
        model = DQN(
            "MlpPolicy",
            train_env,
            verbose=1,
            buffer_size=buffer_size,
            batch_size=batch_size,
            learning_rate=learning_rate,
            exploration_fraction=exploration_fraction,
            # exploration_initial_eps=0.75,
            learning_starts=learning_starts,
            tensorboard_log=f"tb_logs/{model_name}",
        )
    else:
        model = DQN.load(f"models/{self_play}", env=train_env, verbose=1)
    model.learn(
        total_timesteps=train_steps,
        callback=WandbCallback(
            gradient_save_freq=100,
            # model_save_path=f"models/{run.id}",
            verbose=2,
        ),
    )

    model.save(f"models/{model_name}")

    opponent = RandomPlayer(battle_format=BATTLE_FORMAT)
    eval_env = SimpleRLPlayer(
        battle_format=BATTLE_FORMAT, opponent=opponent, start_challenging=True
    )

    mean_reward, std_reward = evaluate_policy(model, eval_env, n_eval_episodes=10)

    print(f"Mean reward: {mean_reward} +/- {std_reward}")

    # Evaluating the model
    print("Results against random player:")
    # dqn.test(eval_env, nb_episodes=100, verbose=False, visualize=False)
    print(
        f"DQN Evaluation: {eval_env.n_won_battles} victories out of {eval_env.n_finished_battles} episodes"
    )


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description="Train a reinforcement learning model")
    parser.add_argument(
        "--model_name",
        "-m",
        type=str,
        default=MODEL_NAME,
        help="Name of the model to be trained",
    )
    parser.add_argument(
        "--buffer_size",
        "-b",
        type=int,
        default=BUFFER_SIZE,
        help="Size of the replay buffer",
    )
    parser.add_argument(
        "--batch_size",
        "-bs",
        type=int,
        default=BATCH_SIZE,
        help="Size of the training batch",
    )
    parser.add_argument(
        "--learning_rate",
        "-lr",
        type=float,
        default=LEARNING_RATE,
        help="Learning rate for the optimizer",
    )
    parser.add_argument(
        "--exploration_fraction",
        "-ef",
        type=float,
        default=EXPLORATION_FRACTION,
        help="Exploration fraction for DQN",
    )
    parser.add_argument(
        "--learning_starts",
        "-ls",
        type=float,
        default=LEARNING_STARTS,
        help="Learning starts for DQN",
    )
    parser.add_argument(
        "--train_steps",
        "-ts",
        type=float,
        default=TRAIN_STEPS,
        help="Train steps for DQN",
    )
    parser.add_argument(
        "--self_play",
        "-sp",
        type=str,
        default=DQN_OPPONENT,
        help="DQN self play opponent",
    )

    args = parser.parse_args()

    run = wandb.init(
        project="pokemon-rl",
        name=args.model_name,
        # config=config,
        sync_tensorboard=True,  # auto-upload sb3's tensorboard metrics
        save_code=True,  # optional
        tags=[
            f"model_{args.model_name}",
            f"buffer_{args.buffer_size}",
            f"lr_{args.learning_rate}",
            f"batch_{args.batch_size}",
            f"exploration_fraction_{args.exploration_fraction}",
            f"learning_starts_{args.learning_starts}",
            f"train_steps_{args.train_steps}",
            f"dqn_self_play_{args.self_play}",
        ],
        config={
            "model_name": args.model_name,
            "buffer_size": args.buffer_size,
            "batch_size": args.batch_size,
            "learning_rate": args.learning_rate,
            "exploration_fraction": args.exploration_fraction,
            "learning_starts": args.learning_starts,
            "train_steps": args.train_steps,
            "dqn_self_play": args.self_play,
        },
    )

    asyncio.run(
        main(
            model_name=args.model_name,
            buffer_size=args.buffer_size,
            batch_size=args.batch_size,
            learning_rate=args.learning_rate,
            exploration_fraction=args.exploration_fraction,
            learning_starts=args.learning_starts,
            train_steps=args.train_steps,
            self_play=args.self_play,
        )
    )
    # asyncio.run(main())
