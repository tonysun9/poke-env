import numpy as np

from poke_env.player import Player
from poke_env.environment import AbstractBattle

from stable_baselines3 import DQN
from poke_env.data import GenData

from poke_env.player.battle_order import BattleOrder, ForfeitBattleOrder


class PlayerSingles(Player):
    pass


class MaxBase(PlayerSingles):
    def choose_move(self, battle: AbstractBattle):
        # If the player can attack, it will
        if battle.available_moves:
            # Finds the best move among available ones
            best_move = max(battle.available_moves, key=lambda move: move.base_power)
            return self.create_order(best_move)

        # If no attack is available, a random switch will be made
        else:
            return self.choose_random_move(battle)


class MaxDamage(PlayerSingles):
    def choose_move(self, battle: AbstractBattle):
        try:
            # no moves probably means pokemon fainted
            if not battle.available_moves:
                return self.choose_random_move(battle)

            # If there's only one option, don't bother checking (e.g. recharge)
            if len(battle.available_moves) == 1:
                return self.create_order(battle.available_moves[0])

            if battle.active_pokemon.species == "ditto":
                # print("choosing move: ditto found!")
                # print("moves available: ", len(battle.available_moves))
                # print("moves: ", battle.available_moves)
                battle.add_new_calc(
                    battle.active_pokemon, battle.opponent_active_pokemon
                )
                # print("---")

            available_dmg_move = any(
                v.base_power > 0 and v in battle.available_moves
                for v in battle.active_pokemon.moves.values()
            )

            if available_dmg_move:
                moves_calc = battle.battle_calcs[
                    (
                        battle.active_pokemon.species,
                        battle.opponent_active_pokemon.species,
                    )
                ]
                available_moves_calc = [
                    move for move in moves_calc if move in set(battle.available_moves)
                ]
                max_dmg_move = max(
                    available_moves_calc, key=lambda move: max(moves_calc[move].damage)
                )

                return self.create_order(max_dmg_move)

            # If no attack is available, a random switch will be made
            else:
                return self.choose_random_move(battle)
        except Exception as e:
            print("Exception in choose_move: ", e)


class DQNPlayer(PlayerSingles):
    def __init__(self, model_path, **kwargs):
        super().__init__(**kwargs)
        self.model = DQN.load(model_path)

    @staticmethod
    def embed_battle(battle: AbstractBattle):
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

    def action_to_move(self, action: int, battle: AbstractBattle) -> BattleOrder:
        """Converts actions to move orders.

        The conversion is done as follows:

        action = -1:
            The battle will be forfeited.
        0 <= action < 4:
            The actionth available move in battle.available_moves is executed.
        4 <= action < 10
            The action - 4th available switch in battle.available_switches is executed.

        If the proposed action is illegal, a random legal move is performed.

        :param action: The action to convert.
        :type action: int
        :param battle: The battle in which to act.
        :type battle: Battle
        :return: the order to send to the server.
        :rtype: str
        """
        if action == -1:
            return ForfeitBattleOrder()
        elif (
            action < 4
            and action < len(battle.available_moves)
            and not battle.force_switch
        ):
            return self.create_order(battle.available_moves[action])
        elif 0 <= action - 4 < len(battle.available_switches):
            return self.create_order(battle.available_switches[action - 4])
        else:
            return self.choose_random_move(battle)

    def choose_move(self, battle: AbstractBattle):
        observation = self.embed_battle(battle)
        action, _ = self.model.predict(observation)
        order = self.action_to_move(action, battle)
        return order
