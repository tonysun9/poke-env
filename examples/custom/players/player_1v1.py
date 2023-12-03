import numpy as np

from poke_env.player import Player
from poke_env.environment import AbstractBattle


class Player1v1(Player):
    pass


class MaxBase(Player1v1):
    def choose_move(self, battle: AbstractBattle):
        # If the player can attack, it will
        if battle.available_moves:
            # Finds the best move among available ones
            best_move = max(battle.available_moves, key=lambda move: move.base_power)
            return self.create_order(best_move)

        # If no attack is available, a random switch will be made
        else:
            return self.choose_random_move(battle)


class MaxDamage(Player1v1):
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


#     def teampreview(self, battle):
#         mon_performance = {}

#         # For each of our pokemons
#         for i, mon in enumerate(battle.team.values()):
#             # We store their average performance against the opponent team
#             mon_performance[i] = np.mean(
#                 [
#                     teampreview_performance(mon, opp)
#                     for opp in battle.opponent_team.values()
#                 ]
#             )

#         # We sort our mons by performance
#         ordered_mons = sorted(mon_performance, key=lambda k: -mon_performance[k])

#         # We start with the one we consider best overall
#         # We use i + 1 as python indexes start from 0
#         #  but showdown's indexes start from 1
#         return "/team " + "".join([str(i + 1) for i in ordered_mons])


# def teampreview_performance(mon_a, mon_b):
#     # We evaluate the performance on mon_a against mon_b as its type advantage
#     a_on_b = b_on_a = -np.inf
#     for type_ in mon_a.types:
#         if type_:
#             a_on_b = max(a_on_b, type_.damage_multiplier(*mon_b.types))
#     # We do the same for mon_b over mon_a
#     for type_ in mon_b.types:
#         if type_:
#             b_on_a = max(b_on_a, type_.damage_multiplier(*mon_a.types))
#     # Our performance metric is the different between the two
#     return a_on_b - b_on_a
