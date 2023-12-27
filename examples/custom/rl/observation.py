import json
from typing import Optional

from poke_env.environment.move import Move
from poke_env.environment.pokemon import Pokemon
from poke_env.environment.pokemon_type import PokemonType
from poke_env.environment.move_category import MoveCategory

from poke_env.data import GenData

from poke_env.environment.abstract_battle import AbstractBattle

import numpy as np
import os


with open(
    "/Users/tonysun/pokemon/poke-env/examples/custom/pkmn_data/gen1randbats_dmg_calc.json",
    "r",
) as f:
    dmg_calc_results = json.load(f)
with open(
    "/Users/tonysun/pokemon/poke-env/examples/custom/pkmn_data/gen1randbats_cleaned.json",
    "r",
) as f:
    gen1data = json.load(f)


class ObservationSimple:
    size = 10

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


def embed_pokemon(pkmn: Optional[Pokemon] = None, revealed=True, fainted=False):
    if not revealed:
        # return [0] * 10
        return [0] * 31

    if fainted:
        # return [0] * 8 + [0 if not pkmn else int(pkmn.active)] + [1]
        return [0] * 29 + [0 if not pkmn else int(pkmn.active)] + [1]

    stats = [
        pkmn.current_hp_fraction,
        pkmn.stats["atk"] / 200 if pkmn.stats["atk"] else 0,
        pkmn.stats["spa"] / 200 if pkmn.stats["spa"] else 0,
        pkmn.stats["spe"] / 200 if pkmn.stats["spe"] else 0,
    ]

    status_order = ["par", "slp", "frz", "brn", "psn"]
    status = [0] * len(status_order)
    if pkmn.status and pkmn.status.name.lower() != "fnt":
        status[status_order.index(str(pkmn.status.name).lower())] = 1

    crit_chance = pkmn.base_stats["spe"] / 512

    # don't need all types for gen 1
    pkmn_type = [0] * len(PokemonType)
    index_1 = pkmn.type_1.value - 1
    pkmn_type[index_1] = 1
    if pkmn.type_2:
        index_2 = pkmn.type_2.value - 1
        pkmn_type[index_2] = 1

    other_info = [crit_chance, int(pkmn.active), int(pkmn.revealed)]

    pkmn_embed = stats + status + pkmn_type + other_info
    return pkmn_embed


def embed_move(
    attacker: Optional[str] = None,
    defender: Optional[str] = None,
    move_str: Optional[str] = None,
    dmg_calc=None,
    fainted=False,
):
    if fainted:
        # return [0] * 27
        return [0] * 47

    move = Move(move_str, gen=1)

    move_info = [move.accuracy, move.crit_ratio, move.priority, move.base_power]

    move_category = [0] * len(MoveCategory)
    move_category[move.category.value - 1] = 1

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
        volatile_status[volatile_statuses.index(str(move.volatile_status).lower())] = 1

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

    # don't need all types for gen 1
    move_type = [0] * len(PokemonType)
    index = move.type.value - 1
    move_type[index] = 1

    # damage calc
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

    return (
        move_category
        + move_type
        + move_info
        + boosts
        + status
        + volatile_status
        + dmg
        + recoil
        + recovery
    )


class ObservationFull:
    # size = 4030
    size = 14110

    # @staticmethod
    # def embed_side(pkmn):
    #     if pkmn.fainted:
    #         return [0] * 11

    #     boosts_order = ["atk", "spa", "def", "spd", "spe", "accuracy"]
    #     boosts = [0] * len(boosts_order)
    #     if pkmn.boosts:
    #         for stat, boost in pkmn.boosts.items():
    #             if stat == "evasion":  # evasion clause
    #                 continue
    #             boosts[boosts_order.index(stat)] = boost

    #     volatile_statuses = ["confusion", "substitute", "reflect"]
    #     volatile_status = [0] * len(volatile_statuses)
    #     for effect in pkmn.effects:
    #         if effect.name.lower() in volatile_statuses:
    #             volatile_status[volatile_statuses.index(effect.name.lower())] = 1

    #     battle_info = [int(pkmn.must_recharge), int(pkmn.preparing)]

    #     return boosts + volatile_status + battle_info

    # @statics+ status + [crit_chance, int(pkmn.active), int(pkmn.revealed)]

    # @staticmethod
    # def embed_move(
    #     attacker: Optional[str] = None,
    #     defender: Optional[str] = None,
    #     move_str: Optional[str] = None,
    #     dmg_calc=None,
    #     fainted=False,
    # ):
    #     if fainted:
    #         return [0] * 27

    #     if (
    #         defender not in dmg_calc[attacker]
    #         or move_str not in dmg_calc[attacker][defender]
    #     ):
    #         dmg = [0, 0]
    #         recoil = [0, 0]
    #         recovery = [0, 0]
    #     else:
    #         dmg = dmg_calc[attacker][defender][move_str].get("damage", [0, 0])
    #         recoil = dmg_calc[attacker][defender][move_str].get("recoil", [0, 0])
    #         recovery = dmg_calc[attacker][defender][move_str].get("recovery", [0, 0])

    #     recovery_moves = {"recover": 0.5, "softboiled": 0.5, "rest": 1}
    #     if move_str in recovery_moves:
    #         recovery = [recovery_moves[move_str], recovery_moves[move_str]]

    #     move = Move(move_str, gen=1)

    #     accuracy = move.accuracy
    #     crit_ratio = move.crit_ratio
    #     priority = move.priority

    #     move_info = [accuracy, crit_ratio, priority]

    #     move_categories = ["physical", "special", "status"]
    #     move_category = [0] * len(move_categories)
    #     move_category[move_categories.index(str(move.category.name).lower())] = 1

    #     boosts_order = ["atk", "spa", "def", "spd", "spe", "accuracy"]
    #     boosts = [0] * len(boosts_order)
    #     if move.boosts:
    #         for stat, boost in move.boosts.items():
    #             boosts[boosts_order.index(stat)] = boost

    #     status_order = ["par", "slp", "frz", "brn", "psn"]
    #     status = [0] * len(status_order)
    #     if move.status:
    #         status[status_order.index(str(move.status.name).lower())] = 1

    #     volatile_statuses = ["confusion", "substitute", "reflect", "flinch"]
    #     volatile_status = [0] * len(volatile_statuses)
    #     if move.volatile_status:
    #         volatile_status[
    #             volatile_statuses.index(str(move.volatile_status).lower())
    #         ] = 1

    #     for secondary in move.secondary:
    #         chance = float(secondary["chance"]) / 100
    #         if "boosts" in secondary:
    #             for stat, boost in secondary["boosts"].items():
    #                 boosts[boosts_order.index(stat)] = boost * chance
    #         if "status" in secondary:
    #             status[status_order.index(str(secondary["status"]).lower())] = chance
    #         if "volatileStatus" in secondary:
    #             volatile_status[
    #                 volatile_statuses.index(str(secondary["volatileStatus"]).lower())
    #             ] = chance

    #     unique_moves = ["hyperbeam", "skyattack", "rest", "counter", "mimic"]
    #     unique_move = [0] * len(unique_moves)
    #     if move_str in unique_moves:
    #         unique_move[unique_moves.index(move_str)] = 1

    #     # print(dmg, recoil, recovery)
    #     # print(move_category, move_info)
    #     # print(boosts, status, volatile_status)
    #     # print(unique_move)

    #     return (
    #         dmg
    #         + recoil
    #         + recovery
    #         + move_category
    #         + move_info
    #         + boosts
    #         + status
    #         + volatile_status
    #     )

    @staticmethod
    def embed_battle(battle: AbstractBattle):
        flatten_list = lambda nested_list: [
            item for sublist in nested_list for item in sublist
        ]

        # our alive pokemon in order of active, then switches, then fainted.
        our_team = (
            [battle.active_pokemon]
            + battle.available_switches
            + [
                pkmn
                for pkmn in battle.team.values()
                if pkmn.fainted and not pkmn.active
            ]
        )
        # pad with rest of team if < 6
        # (meaning can't switch out, recharge, preparing etc.)
        if len(our_team) < 6:
            our_team += [pkmn for pkmn in battle.team.values() if pkmn not in our_team]
        our_team_pkmn = [embed_pokemon(pkmn, fainted=pkmn.fainted) for pkmn in our_team]

        # opp pokemon in order of active, then switches. unrevealed pokemon are last
        opp_team = sorted(
            battle.opponent_team.values(),
            key=lambda pkmn: (not pkmn.active, pkmn.fainted),
        )

        opp_team_pkmn = [embed_pokemon(pkmn, fainted=pkmn.fainted) for pkmn in opp_team]
        num_unrevealed = 6 - len(opp_team)
        opp_team_pkmn += [embed_pokemon(revealed=False) for _ in range(num_unrevealed)]

        moves = []
        # attack / damage taken for each of our alive pokemon
        for our_pkmn in our_team:
            # attack moves vs seen / fainted pokemon on opp team
            for opp_def in opp_team:
                moves.append(
                    [
                        embed_move(
                            our_pkmn.species,
                            opp_def.species,
                            move,
                            dmg_calc_results,
                            fainted=our_pkmn.fainted or opp_def.fainted,
                        )
                        for move in our_pkmn.moves
                    ]
                )
                if our_pkmn.species == "ditto":
                    moves.append(
                        [
                            embed_move(fainted=True)
                            for _ in range(4 - len(our_pkmn.moves))
                        ]
                    )
            # attack moves vs unrevealed pokemon on opp team
            for _ in range(num_unrevealed):
                moves.append(
                    [
                        embed_move(
                            fainted=True,
                        )
                        for _ in range(4)
                    ]
                )
            # damage taken from opp team (seen / fainted)
            # move vector size + 1 for chance of having move
            for opp_atk in opp_team:
                # all possible attack moves
                num_moves = len(gen1data[opp_atk.species])
                moves.append(
                    [
                        embed_move(
                            opp_atk.species,
                            our_pkmn.species,
                            move,
                            dmg_calc_results,
                            opp_atk.fainted or our_pkmn.fainted,
                        )
                        + [chance]
                        for move, chance in gen1data[opp_atk.species].items()
                    ]
                )
                # pad remaining opp attack moves to 10 moves (max)
                moves.append(
                    [
                        embed_move(fainted=True) + [0]
                        for _ in range(
                            10 - num_moves
                        )  # max number of moves is 10 by Pidgeot
                    ]
                )
            # damage taken from unrevealed pokemon on opp team
            for _ in range(num_unrevealed):
                moves.append(
                    [
                        embed_move(
                            fainted=True,  # treat unrevealed as fainted for moves
                        )
                        + [0]
                        for _ in range(10)
                    ]
                )

        # print("our active", len(embed_side(battle.active_pokemon)))
        # print(
        #     "opp active",
        #     len(embed_side(battle.opponent_active_pokemon)),
        # )
        # print("our team  ", len(flatten_list(our_team_pkmn)))
        # print("opp team  ", len(flatten_list(opp_team_pkmn)))
        # print("total move", len(flatten_list(flatten_list(moves))))

        # obs = np.array(
        #     embed_side(battle.active_pokemon)
        #     + embed_side(battle.opponent_active_pokemon)
        #     + flatten_list(our_team_pkmn)
        #     + flatten_list(opp_team_pkmn)
        #     + flatten_list(flatten_list(moves))
        # )
        # assert len(obs) == 14110

        obs = np.array(
            embed_side(battle.active_pokemon)
            + embed_side(battle.opponent_active_pokemon)
            + flatten_list(our_team_pkmn)
            + flatten_list(opp_team_pkmn)
            + flatten_list(flatten_list(moves))
        )
        if len(obs) < 14110:
            obs = np.pad(obs, (0, 14110 - len(obs)), "constant")
            print("padded obs", len(obs))
            print(battle.active_pokemon)
            print([pkmn for pkmn in battle.team.values()])
            print(battle.opponent_active_pokemon)
            print([pkmn for pkmn in battle.opponent_team.values()])

        return

    # @staticmethod
    # def embed_battle_old(battle: AbstractBattle):
    #     flatten_list = lambda nested_list: [
    #         item for sublist in nested_list for item in sublist
    #     ]

    #     our_team = [pkmn for pkmn in battle.team.values()]
    #     opp_team = [pkmn for pkmn in battle.opponent_team.values()]

    #     our_team_pkmn = [
    #         ObservationFull.embed_pokemon(pkmn, fainted=pkmn.fainted)
    #         for pkmn in our_team
    #     ]
    #     opp_team_pkmn = [
    #         ObservationFull.embed_pokemon(pkmn, fainted=pkmn.fainted)
    #         for pkmn in opp_team
    #     ]
    #     for i in range(6 - len(opp_team)):
    #         opp_team_pkmn.append(ObservationFull.embed_pokemon(revealed=False))

    #     our_moves = []
    #     for our_atk in our_team:
    #         for opp_def in opp_team:
    #             our_moves.append(
    #                 [
    #                     ObservationFull.embed_move(
    #                         our_atk.species,
    #                         opp_def.species,
    #                         move,
    #                         dmg_calc_results,
    #                         fainted=our_atk.fainted or opp_def.fainted,
    #                     )
    #                     for move in our_atk.moves
    #                 ]
    #             )
    #             if our_atk.species == "ditto":
    #                 our_moves.append(
    #                     [
    #                         ObservationFull.embed_move(fainted=True)
    #                         for _ in range(4 - len(our_atk.moves))
    #                     ]
    #                 )
    #         for _ in range(6 - len(opp_team)):
    #             our_moves.append(
    #                 [
    #                     ObservationFull.embed_move(
    #                         fainted=True,
    #                     )
    #                     for _ in range(4)
    #                 ]
    #             )

    #     opp_moves = []
    #     for opp_atk in opp_team:
    #         for our_def in our_team:
    #             num_moves = len(gen1data[opp_atk.species])
    #             opp_moves.append(
    #                 [
    #                     ObservationFull.embed_move(
    #                         opp_atk.species,
    #                         our_def.species,
    #                         move,
    #                         dmg_calc_results,
    #                         opp_atk.fainted or our_def.fainted,
    #                     )
    #                     + [chance]
    #                     for move, chance in gen1data[opp_atk.species].items()
    #                 ]
    #             )
    #             # dummy moves for padding
    #             opp_moves.append(
    #                 [
    #                     ObservationFull.embed_move(fainted=True) + [0]
    #                     for _ in range(
    #                         10 - num_moves
    #                     )  # max number of moves is 10 by Pidgeot
    #                 ]
    #             )
    #     for j in range(6 - len(opp_team)):  # unrevealed
    #         for k in range(6):
    #             opp_moves.append(
    #                 [
    #                     ObservationFull.embed_move(
    #                         fainted=True,  # treat unrevealed as fainted for moves
    #                     )
    #                     + [0]
    #                     for _ in range(10)
    #                 ]
    #             )

    #     return np.array(
    #         ObservationFull.embed_side(battle.active_pokemon)
    #         + flatten_list(our_team_pkmn)
    #         + flatten_list(flatten_list(our_moves))
    #         + ObservationFull.embed_side(battle.opponent_active_pokemon)
    #         + flatten_list(opp_team_pkmn)
    #         + flatten_list(flatten_list(opp_moves))
    #     )


class ObservationIncremental:
    # size = 250
    # size = 4822
    size = 8062

    @staticmethod
    def embed_battle(battle: AbstractBattle):
        flatten_list = lambda nested_list: [
            item for sublist in nested_list for item in sublist
        ]

        # our_team = [pkmn for pkmn in battle.team.values()]
        # opp_team = [pkmn for pkmn in battle.opponent_team.values()]

        # our alive pokemon in order of active, then switches, then fainted.
        our_team = (
            [battle.active_pokemon]
            + battle.available_switches
            + [
                pkmn
                for pkmn in battle.team.values()
                if pkmn.fainted and not pkmn.active
            ]
        )
        # pad with rest of team if < 6
        # (meaning can't switch out, recharge, preparing etc.)
        if len(our_team) < 6:
            our_team += [pkmn for pkmn in battle.team.values() if pkmn not in our_team]
        our_team_pkmn = [embed_pokemon(pkmn, fainted=pkmn.fainted) for pkmn in our_team]
        # pad with dummy pokemon if only 1
        # (meaning can't switch out, recharge, preparing etc.)
        # treat as fainted
        # if len(our_team_pkmn) == 1:
        #     our_team_pkmn += [
        #         embed_pokemon(fainted=True) for _ in range(5)
        #     ]

        # opp pokemon in order of active, then switches. unrevealed pokemon are last
        opp_team = sorted(
            battle.team.values(), key=lambda pkmn: (not pkmn.active, pkmn.fainted)
        )

        opp_team_pkmn = [embed_pokemon(pkmn, fainted=pkmn.fainted) for pkmn in opp_team]
        num_unrevealed = 6 - len(opp_team)
        opp_team_pkmn += [embed_pokemon(revealed=False) for _ in range(num_unrevealed)]

        moves = []
        # attack / damage taken for each of our alive pokemon
        for our_pkmn in our_team[:]:
            # attack moves vs seen / fainted pokemon on opp team
            for opp_def in opp_team[:]:
                moves.append(
                    [
                        embed_move(
                            our_pkmn.species,
                            opp_def.species,
                            move,
                            dmg_calc_results,
                            fainted=our_pkmn.fainted or opp_def.fainted,
                        )
                        for move in our_pkmn.moves
                    ]
                )
                if our_pkmn.species == "ditto":
                    moves.append(
                        [
                            embed_move(fainted=True)
                            for _ in range(4 - len(our_pkmn.moves))
                        ]
                    )
            # attack moves vs unrevealed pokemon on opp team
            for _ in range(num_unrevealed):
                moves.append(
                    [
                        embed_move(
                            fainted=True,
                        )
                        for _ in range(4)
                    ]
                )
            # damage taken from opp team (seen / fainted)
            # move vector size + 1 for chance of having move
            for opp_atk in opp_team:
                # choose top 4 most likely moves
                num_moves = len(gen1data[opp_atk.species])
                moves.append(
                    [
                        embed_move(
                            opp_atk.species,
                            our_pkmn.species,
                            move,
                            dmg_calc_results,
                            opp_atk.fainted or our_pkmn.fainted,
                        )
                        + [chance]
                        for move, chance in sorted(
                            gen1data[opp_atk.species].items(),
                            key=lambda x: x[1],
                            reverse=True,
                        )[:4]
                    ]
                )
                # pad remaining opp attack moves to 4 moves (max)
                # e.g. ditto might only have 1 move
                moves.append(
                    [embed_move(fainted=True) + [0] for _ in range(4 - num_moves)]
                )
            # damage taken from unrevealed pokemon on opp team
            for _ in range(num_unrevealed):
                moves.append(
                    [
                        embed_move(
                            fainted=True,  # treat unrevealed as fainted for moves
                        )
                        + [0]
                        for _ in range(4)
                    ]
                )

        # print("our active", len(embed_side(battle.active_pokemon)))
        # print(
        #     "opp active",
        #     len(embed_side(battle.opponent_active_pokemon)),
        # )
        # print("our team  ", len(flatten_list(our_team_pkmn)))
        # print("opp team  ", len(flatten_list(opp_team_pkmn)))
        # print("total move", len(flatten_list(flatten_list(moves))))

        # obs = np.array(
        #     embed_side(battle.active_pokemon)
        #     + embed_side(battle.opponent_active_pokemon)
        #     + flatten_list(our_team_pkmn)
        #     + flatten_list(opp_team_pkmn)
        #     + flatten_list(flatten_list(moves))
        # )
        # assert len(obs) == 14110

        return np.array(
            embed_side(battle.active_pokemon)
            + embed_side(battle.opponent_active_pokemon)
            + flatten_list(our_team_pkmn)
            + flatten_list(opp_team_pkmn)
            + flatten_list(flatten_list(moves))
        )


class ObservationMatrix:
    # size = 8062

    @staticmethod
    def embed_battle(battle: AbstractBattle):
        flatten_list = lambda nested_list: [
            item for sublist in nested_list for item in sublist
        ]

        # our alive pokemon in order of active, then switches, then fainted.
        our_team = (
            [battle.active_pokemon]
            + battle.available_switches
            + [
                pkmn
                for pkmn in battle.team.values()
                if pkmn.fainted and not pkmn.active
            ]
        )
        # pad with rest of team if < 6
        # (meaning can't switch out, recharge, preparing etc.)
        if len(our_team) < 6:
            our_team += [pkmn for pkmn in battle.team.values() if pkmn not in our_team]
        # our_team_pkmn = [
        #     ObservationFull.embed_pokemon(pkmn, fainted=pkmn.fainted)
        #     for pkmn in our_team
        # ]
        # pad with dummy pokemon if only 1
        # (meaning can't switch out, recharge, preparing etc.)
        # treat as fainted
        # if len(our_team_pkmn) == 1:
        #     our_team_pkmn += [
        #         ObservationFull.embed_pokemon(fainted=True) for _ in range(5)
        #     ]

        # opp pokemon in order of active, then switches. unrevealed pokemon are last
        opp_team = sorted(
            battle.opponent_team.values(),
            key=lambda pkmn: (not pkmn.active, pkmn.fainted),
        )

        opp_team_pkmn = [embed_pokemon(pkmn, fainted=pkmn.fainted) for pkmn in opp_team]
        num_unrevealed = 6 - len(opp_team)
        opp_team_pkmn += [embed_pokemon(revealed=False) for _ in range(num_unrevealed)]

        num_opp_moves = 7

        # moves = []
        team_pkmn_embed = []
        # embed each pokemon
        for our_pkmn in our_team:
            our_pkmn_emb = []
            # embed our pokemon
            our_pkmn_emb.append([embed_pokemon(our_pkmn, fainted=our_pkmn.fainted)])
            our_pkmn_emb.append(opp_team_pkmn)

            # attack moves vs seen / fainted pokemon on opp team
            for opp_def in opp_team:
                our_pkmn_emb.append(
                    [
                        embed_move(
                            our_pkmn.species,
                            opp_def.species,
                            move,
                            dmg_calc_results,
                            fainted=our_pkmn.fainted or opp_def.fainted,
                        )
                        for move in our_pkmn.moves
                    ]
                )
                if our_pkmn.species == "ditto":
                    our_pkmn_emb.append(
                        [
                            embed_move(fainted=True)
                            for _ in range(4 - len(our_pkmn.moves))
                        ]
                    )
            # attack moves vs unrevealed pokemon on opp team
            for _ in range(num_unrevealed):
                our_pkmn_emb.append(
                    [
                        embed_move(
                            fainted=True,
                        )
                        for _ in range(4)
                    ]
                )
            # damage taken from opp team (seen / fainted)
            # move vector size + 1 for chance of having move
            for opp_atk in opp_team:
                # all possible moves opponent can have
                num_moves = len(gen1data[opp_atk.species])
                our_pkmn_emb.append(
                    [
                        embed_move(
                            opp_atk.species,
                            our_pkmn.species,
                            move,
                            dmg_calc_results,
                            opp_atk.fainted or our_pkmn.fainted,
                        )
                        + [chance]
                        for move, chance in sorted(
                            gen1data[opp_atk.species].items(),
                            key=lambda x: x[1],
                            reverse=True,
                        )[:num_opp_moves]
                    ]
                )
                # pad remaining opp attack moves to 10 moves (max)
                # e.g. ditto might only have 1 move
                our_pkmn_emb.append(
                    [
                        embed_move(fainted=True) + [0]
                        for _ in range(num_opp_moves - num_moves)
                    ]
                )
            # damage taken from unrevealed pokemon on opp team
            for _ in range(num_unrevealed):
                our_pkmn_emb.append(
                    [
                        embed_move(
                            fainted=True,  # treat unrevealed as fainted for moves
                        )
                        + [0]
                        for _ in range(num_opp_moves)
                    ]
                )

            team_pkmn_embed.append(flatten_list(flatten_list(our_pkmn_emb)))

        try:
            obs = np.array(team_pkmn_embed).flatten()

            if len(obs) < 15840:
                obs = np.pad(obs, (0, 15840 - len(obs)), "constant")
                print(
                    "observation length:", len(obs), "padding: " + str(15840 - len(obs))
                )
                print(battle.active_pokemon)
                print([pkmn for pkmn in battle.team.values()])
                print(battle.opponent_active_pokemon)
                print([pkmn for pkmn in battle.opponent_team.values()])

            # print(obs.shape)
            return obs.reshape(6, -1)
        except:
            return np.zeros((6, 3361))
