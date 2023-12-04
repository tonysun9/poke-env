import asyncio

import numpy as np

# from gym.spaces import Box, Space
# from gym.utils.env_checker import check_env
from gymnasium.spaces import Box, Space
from gymnasium.utils.env_checker import check_env
import torch
import torch.nn as nn

import torch.nn.functional as F

from tabulate import tabulate

# from tensorflow.keras.layers import Dense, Flatten
# from tensorflow.keras.models import Sequential
# from tensorflow.keras.optimizers import Adam

# import tensorflow as tf
# from keras import __version__

# tf.keras.__version__ = __version__
# from rl.agents.dqn import DQNAgent
# from rl.memory import SequentialMemory
# from rl.policy import EpsGreedyQPolicy, LinearAnnealedPolicy

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

BATTLE_FORMAT = "gen1randombattle"


class SimpleRLPlayer(Gen1EnvSinglePlayer):
    def calc_reward(self, last_battle, current_battle) -> float:
        return self.reward_computing_helper(
            current_battle, fainted_value=2.0, hp_value=1.0, victory_value=30.0
        )

    # def embed_battle(self, battle: AbstractBattle) -> ObservationType:
    def embed_battle(self, battle: AbstractBattle):
        # -1 indicates that the move does not have a base power
        # or is not available
        moves_base_power = -np.ones(4)
        moves_dmg_multiplier = np.ones(4)
        for i, move in enumerate(battle.available_moves):
            moves_base_power[i] = (
                move.base_power / 100
            )  # Simple rescaling to facilitate learning
            if move.type:
                moves_dmg_multiplier[i] = move.type.damage_multiplier(
                    battle.opponent_active_pokemon.type_1,
                    battle.opponent_active_pokemon.type_2,
                    type_chart=GenData.from_gen(8).type_chart,
                )

        # We count how many pokemons have fainted in each team
        fainted_mon_team = len([mon for mon in battle.team.values() if mon.fainted]) / 6
        fainted_mon_opponent = (
            len([mon for mon in battle.opponent_team.values() if mon.fainted]) / 6
        )

        # Final vector with 10 components
        final_vector = np.concatenate(
            [
                moves_base_power,
                moves_dmg_multiplier,
                [fainted_mon_team, fainted_mon_opponent],
            ]
        )
        return np.float32(final_vector)

    def describe_embedding(self) -> Space:
        low = [-1, -1, -1, -1, 0, 0, 0, 0, 0, 0]
        high = [3, 3, 3, 3, 4, 4, 4, 4, 1, 1]
        return Box(
            low=np.array(low, dtype=np.float32),
            high=np.array(high, dtype=np.float32),
            shape=(10,),
            dtype=np.float32,
        )


async def main():
    # First test the environment to ensure the class is consistent
    # with the OpenAI API
    # opponent = RandomPlayer(battle_format=BATTLE_FORMAT)
    # test_env = SimpleRLPlayer(
    #     battle_format=BATTLE_FORMAT, start_challenging=True, opponent=opponent
    # )
    # check_env(test_env)
    # test_env.close()

    # Create one environment for training and one for evaluation
    opponent = RandomPlayer(battle_format=BATTLE_FORMAT)
    train_env = SimpleRLPlayer(
        battle_format=BATTLE_FORMAT, opponent=opponent, start_challenging=True
    )

    from stable_baselines3 import DQN
    from stable_baselines3.common.env_util import make_vec_env
    from stable_baselines3.common.evaluation import evaluate_policy

    model = DQN("MlpPolicy", train_env, verbose=1)
    model.learn(total_timesteps=int(1e5))

    model.save("models/pokemon_rl_dqn")

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
    # second_opponent = MaxBasePowerPlayer(battle_format=BATTLE_FORMAT)
    # eval_env.reset_env(restart=True, opponent=second_opponent)
    # print("Results against max base power player:")
    # # dqn.test(eval_env, nb_episodes=100, verbose=False, visualize=False)
    # # print(
    # #     f"DQN Evaluation: {eval_env.n_won_battles} victories out of {eval_env.n_finished_battles} episodes"
    # # )
    # eval_env.reset_env(restart=False)

    # # Evaluate the player with included util method
    # n_challenges = 250
    # placement_battles = 40
    # eval_task = background_evaluate_player(
    #     eval_env.agent, n_challenges, placement_battles
    # )
    # # dqn.test(eval_env, nb_episodes=n_challenges, verbose=False, visualize=False)
    # print("Evaluation with included method:", eval_task.result())
    # eval_env.reset_env(restart=False)

    # # Cross evaluate the player with included util method
    # n_challenges = 50
    # players = [
    #     eval_env.agent,
    #     RandomPlayer(battle_format=BATTLE_FORMAT),
    #     MaxBasePowerPlayer(battle_format=BATTLE_FORMAT),
    #     SimpleHeuristicsPlayer(battle_format=BATTLE_FORMAT),
    # ]
    # cross_eval_task = background_cross_evaluate(players, n_challenges)
    # dqn.test(
    #     eval_env,
    #     nb_episodes=n_challenges * (len(players) - 1),
    #     verbose=False,
    #     visualize=False,
    # )
    # cross_evaluation = cross_eval_task.result()
    # table = [["-"] + [p.username for p in players]]
    # for p_1, results in cross_evaluation.items():
    #     table.append([p_1] + [cross_evaluation[p_1][p_2] for p_2 in results])
    # print("Cross evaluation of DQN with baselines:")
    # print(tabulate(table))
    # eval_env.close()


if __name__ == "__main__":
    # asyncio.get_event_loop().run_until_complete(main())
    asyncio.run(main())
