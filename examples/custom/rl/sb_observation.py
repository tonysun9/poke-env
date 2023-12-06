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

BATTLE_FORMAT = "gen1randombattle"

MODEL_NAME = "observation_1_1_1e5"
TRAIN_STEPS = int(1e5)


# config = {
#     "observation": "imp",
# }
run = wandb.init(
    project="pokemon-rl",
    name=MODEL_NAME,
    # config=config,
    sync_tensorboard=True,  # auto-upload sb3's tensorboard metrics
    save_code=True,  # optional
)


class SimpleRLPlayer(Gen1EnvSinglePlayer):
    def __init__(self, **kwargs):
        super().__init__(**kwargs)
        with open("../pkmn_data/gen1randbats_dmg_calc.json", "r") as f:
            self.dmg_calc_results = json.load(f)
        with open("../pkmn_data/gen1randbats_cleaned.json", "r") as f:
            self.gen1data = json.load(f)

    def calc_reward(self, last_battle, current_battle) -> float:
        return self.reward_computing_helper(
            current_battle, fainted_value=2.0, hp_value=1.0, victory_value=30.0
        )

    @staticmethod
    def embed_side(pkmn):
        if pkmn.fainted:
            return [0] * 11

        boosts_order = ["atk", "spa", "def", "spd", "spe", "accuracy"]
        boosts = [0] * len(boosts_order)
        if pkmn.boosts:
            for stat, boost in pkmn.boosts.items():
                if stat == "evasion":  # evasion clause
                    continue
                boosts[boosts_order.index(stat)] = boost

        volatile_statuses = ["confusion", "substitute", "reflect"]
        volatile_status = [0] * len(volatile_statuses)
        for effect in pkmn.effects:
            if effect.name.lower() in volatile_statuses:
                volatile_status[volatile_statuses.index(effect.name.lower())] = 1

        battle_info = [int(pkmn.must_recharge), int(pkmn.preparing)]

        return boosts + volatile_status + battle_info

    @staticmethod
    def embed_pokemon(pkmn: Optional[Pokemon] = None, revealed=True, fainted=False):
        if not revealed:
            # return [0] * 9
            return [0] * 10

        if fainted:
            return [0] * 8 + [int(pkmn.active)] + [1]
            # return [0] * 8 + [1]

        stats = [
            pkmn.current_hp_fraction,
            pkmn.stats["spe"] / 200 if pkmn.stats["spe"] else 0,
        ]

        status_order = ["par", "slp", "frz", "brn", "psn"]
        status = [0] * len(status_order)
        if pkmn.status and pkmn.status.name.lower() != "fnt":
            status[status_order.index(str(pkmn.status.name).lower())] = 1

        crit_chance = pkmn.base_stats["spe"] / 512

        return stats + status + [crit_chance, int(pkmn.active), int(pkmn.revealed)]

    @staticmethod
    def embed_move(
        attacker: Optional[str] = None,
        defender: Optional[str] = None,
        move_str: Optional[str] = None,
        dmg_calc=None,
        fainted=False,
    ):
        if fainted:
            return [0] * 27

        if (
            defender not in dmg_calc[attacker]
            or move_str not in dmg_calc[attacker][defender]
        ):
            dmg = [0, 0]
            recoil = [0, 0]
            recovery = [0, 0]
        else:
            dmg = dmg_calc[attacker][defender][move_str].get("damage", [0, 0])
            recoil = dmg_calc[attacker][defender][move_str].get("recoil", [0, 0])
            recovery = dmg_calc[attacker][defender][move_str].get("recovery", [0, 0])

        recovery_moves = {"recover": 0.5, "softboiled": 0.5, "rest": 1}
        if move_str in recovery_moves:
            recovery = [recovery_moves[move_str], recovery_moves[move_str]]

        move = Move(move_str, gen=1)

        accuracy = move.accuracy
        crit_ratio = move.crit_ratio
        priority = move.priority

        move_info = [accuracy, crit_ratio, priority]

        move_categories = ["physical", "special", "status"]
        move_category = [0] * len(move_categories)
        move_category[move_categories.index(str(move.category.name).lower())] = 1

        boosts_order = ["atk", "spa", "def", "spd", "spe", "accuracy"]
        boosts = [0] * len(boosts_order)
        if move.boosts:
            for stat, boost in move.boosts.items():
                boosts[boosts_order.index(stat)] = boost

        status_order = ["par", "slp", "frz", "brn", "psn"]
        status = [0] * len(status_order)
        if move.status:
            status[status_order.index(str(move.status.name).lower())] = 1

        volatile_statuses = ["confusion", "substitute", "reflect", "flinch"]
        volatile_status = [0] * len(volatile_statuses)
        if move.volatile_status:
            volatile_status[
                volatile_statuses.index(str(move.volatile_status).lower())
            ] = 1

        for secondary in move.secondary:
            chance = float(secondary["chance"]) / 100
            if "boosts" in secondary:
                for stat, boost in secondary["boosts"].items():
                    boosts[boosts_order.index(stat)] = boost * chance
            if "status" in secondary:
                status[status_order.index(str(secondary["status"]).lower())] = chance
            if "volatileStatus" in secondary:
                volatile_status[
                    volatile_statuses.index(str(secondary["volatileStatus"]).lower())
                ] = chance

        unique_moves = ["hyperbeam", "skyattack", "rest", "counter", "mimic"]
        unique_move = [0] * len(unique_moves)
        if move_str in unique_moves:
            unique_move[unique_moves.index(move_str)] = 1

        # print(dmg, recoil, recovery)
        # print(move_category, move_info)
        # print(boosts, status, volatile_status)
        # print(unique_move)

        return (
            dmg
            + recoil
            + recovery
            + move_category
            + move_info
            + boosts
            + status
            + volatile_status
        )

    def embed_battle(self, battle: AbstractBattle):
        flatten_list = lambda nested_list: [
            item for sublist in nested_list for item in sublist
        ]

        our_team = [pkmn for pkmn in battle.team.values()]
        opp_team = [pkmn for pkmn in battle.opponent_team.values()]

        our_team_pkmn = [
            self.embed_pokemon(pkmn, fainted=pkmn.fainted) for pkmn in our_team
        ]
        opp_team_pkmn = [
            self.embed_pokemon(pkmn, fainted=pkmn.fainted) for pkmn in opp_team
        ]
        for i in range(6 - len(opp_team)):
            opp_team_pkmn.append(self.embed_pokemon(revealed=False))

        our_moves = []
        for our_atk in our_team:
            for opp_def in opp_team:
                our_moves.append(
                    [
                        self.embed_move(
                            our_atk.species,
                            opp_def.species,
                            move,
                            self.dmg_calc_results,
                            fainted=our_atk.fainted or opp_def.fainted,
                        )
                        for move in our_atk.moves
                    ]
                )
                if our_atk.species == "ditto":
                    our_moves.append(
                        [
                            self.embed_move(fainted=True)
                            for _ in range(4 - len(our_atk.moves))
                        ]
                    )
            for i in range(6 - len(opp_team)):
                our_moves.append(
                    [
                        self.embed_move(
                            fainted=True,
                        )
                        for _ in range(4)
                    ]
                )

        opp_moves = []
        for opp_atk in opp_team:
            for our_def in our_team:
                num_moves = len(self.gen1data[opp_atk.species])
                opp_moves.append(
                    [
                        self.embed_move(
                            opp_atk.species,
                            our_def.species,
                            move,
                            self.dmg_calc_results,
                            opp_atk.fainted or our_def.fainted,
                        )
                        + [chance]
                        for move, chance in self.gen1data[opp_atk.species].items()
                    ]
                )
                # dummy moves for padding
                opp_moves.append(
                    [
                        self.embed_move(fainted=True) + [0]
                        for _ in range(
                            10 - num_moves
                        )  # max number of moves is 10 by Pidgeot
                    ]
                )
        for j in range(6 - len(opp_team)):  # unrevealed
            for k in range(6):
                opp_moves.append(
                    [
                        self.embed_move(
                            fainted=True,  # treat unrevealed as fainted for moves
                        )
                        + [0]
                        for _ in range(10)
                    ]
                )

        return np.array(
            self.embed_side(battle.active_pokemon)
            + flatten_list(our_team_pkmn)
            + flatten_list(flatten_list(our_moves))
            + self.embed_side(battle.opponent_active_pokemon)
            + flatten_list(opp_team_pkmn)
            + flatten_list(flatten_list(opp_moves))
        )

    def describe_embedding(self) -> Space:
        # return Box(
        #     low=np.zeros(4030),
        #     high=np.full(4030, 2),
        #     shape=(4030,),
        #     dtype=np.float16,
        # )
        return Box(
            low=np.zeros(14110),
            high=np.full(14110, 2),
            shape=(14110,),
            dtype=np.float16,
        )


async def main():
    # Create one environment for training and one for evaluation
    opponent = RandomPlayer(battle_format=BATTLE_FORMAT)
    train_env = SimpleRLPlayer(
        battle_format=BATTLE_FORMAT, opponent=opponent, start_challenging=True
    )

    from stable_baselines3 import DQN
    from stable_baselines3.common.env_util import make_vec_env
    from stable_baselines3.common.evaluation import evaluate_policy

    model = DQN("MlpPolicy", train_env, verbose=1)
    # model = DQN.load("models/observation_0_1", env=train_env, verbose=1)
    model.learn(
        total_timesteps=TRAIN_STEPS,
        callback=WandbCallback(
            gradient_save_freq=100,
            # model_save_path=f"models/{run.id}",
            verbose=2,
        ),
    )

    model.save(f"models/{MODEL_NAME}")

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
    asyncio.run(main())
