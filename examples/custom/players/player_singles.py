import numpy as np
import json
from typing import Optional

from poke_env.player import Player
from poke_env.environment import AbstractBattle

from stable_baselines3 import DQN
from poke_env.data import GenData

from poke_env.player.battle_order import BattleOrder, ForfeitBattleOrder

from poke_env.environment.move import Move
from poke_env.environment.pokemon import Pokemon


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


class DQNPlayer(PlayerSingles):
    def __init__(self, model_path, **kwargs):
        super().__init__(**kwargs)
        self.model = DQN.load(model_path)
        # self.model = DQN.load(model_path)
        # with open("pkmn_data/crit_chance.json", "r") as f:
        #     self.crit_chance = json.load(f)
        with open("pkmn_data/gen1randbats_dmg_calc.json", "r") as f:
            self.dmg_calc_results = json.load(f)
        with open("pkmn_data/gen1randbats_cleaned.json", "r") as f:
            self.gen1data = json.load(f)

    def choose_move(self, battle: AbstractBattle) -> BattleOrder:
        obs = self.test_embed_battle(battle)
        # print(battle.active_pokemon)
        # print(battle.opponent_active_pokemon)
        # print(
        #     "fainted",
        #     battle.active_pokemon.fainted,
        #     battle.opponent_active_pokemon.fainted,
        # )
        # print(self.embed_pokemon(battle.active_pokemon))
        # print(len([mon for mon in battle.team.values() if mon.fainted]))
        # print(len([mon for mon in battle.opponent_team.values() if mon.fainted]))
        # print(battle.team)
        # print(battle.opponent_team)
        # print(obs)
        print(len(obs))
        assert (len(obs)) == 4030
        return self.choose_random_move(battle)

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
            return [0] * 10

        if fainted:
            return [0] * 8 + [int(pkmn.active)] + [1]

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

    def test_embed_battle(self, battle: AbstractBattle):
        flatten_list = lambda nested_list: [
            item for sublist in nested_list for item in sublist
        ]

        # our_team = [battle.active_pokemon] + sorted(
        #     [pkmn for pkmn in battle.team.values() if pkmn != battle.active_pokemon],
        #     key=lambda pkmn: pkmn.species,
        # )
        # opp_team = [battle.opponent_active_pokemon] + [
        #     pkmn
        #     for pkmn in battle.opponent_team.values()
        #     if pkmn != battle.opponent_active_pokemon
        # ]
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

        # opp_moves = []
        # for opp_atk in opp_team:
        #     for our_def in our_team:
        #         num_moves = len(self.gen1data[opp_atk.species])
        #         opp_moves.append(
        #             [
        #                 self.embed_move(
        #                     opp_atk.species,
        #                     our_def.species,
        #                     move,
        #                     self.dmg_calc_results,
        #                     opp_atk.fainted or our_def.fainted,
        #                 )
        #                 + [chance]
        #                 for move, chance in self.gen1data[opp_atk.species].items()
        #             ]
        #         )
        #         # dummy moves for padding
        #         opp_moves.append(
        #             [
        #                 self.embed_move(fainted=True) + [0]
        #                 for _ in range(
        #                     10 - num_moves
        #                 )  # max number of moves is 10 by Pidgeot
        #             ]
        #         )
        # for j in range(6 - len(opp_team)):  # unrevealed
        #     for k in range(6):
        #         opp_moves.append(
        #             [
        #                 self.embed_move(
        #                     fainted=True,  # treat unrevealed as fainted for moves
        #                 )
        #                 + [0]
        #                 for _ in range(10)
        #             ]
        #         )

        # print("our moves: ", len(flatten_list(flatten_list(our_moves))))
        assert len(flatten_list(flatten_list(our_moves))) == 3888
        # print("opp moves: ", len(flatten_list(flatten_list(opp_moves))))
        # assert len(flatten_list(flatten_list(opp_moves))) == 10080

        return np.array(
            self.embed_side(battle.active_pokemon)
            + flatten_list(our_team_pkmn)
            + flatten_list(flatten_list(our_moves))
            + self.embed_side(battle.opponent_active_pokemon)
            + flatten_list(opp_team_pkmn)
            # + flatten_list(flatten_list(opp_moves))
        )


class DQNPlayerObservation(PlayerSingles):
    def __init__(self, model_path, **kwargs):
        super().__init__(**kwargs)
        self.model = DQN.load(model_path)
        with open("pkmn_data/gen1randbats_dmg_calc.json", "r") as f:
            self.dmg_calc_results = json.load(f)
        with open("pkmn_data/gen1randbats_cleaned.json", "r") as f:
            self.gen1data = json.load(f)

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

        # our_team = [battle.active_pokemon] + [
        #     pkmn for pkmn in battle.team.values() if pkmn != battle.active_pokemon
        # ]
        # opp_team = [battle.opponent_active_pokemon] + [
        #     pkmn
        #     for pkmn in battle.opponent_team.values()
        #     if pkmn != battle.opponent_active_pokemon
        # ]
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

        # opp_moves = []
        # for opp_atk in opp_team:
        #     for our_def in our_team:
        #         num_moves = len(self.gen1data[opp_atk.species])
        #         opp_moves.append(
        #             [
        #                 self.embed_move(
        #                     opp_atk.species,
        #                     our_def.species,
        #                     move,
        #                     self.dmg_calc_results,
        #                     opp_atk.fainted or our_def.fainted,
        #                 )
        #                 + [chance]
        #                 for move, chance in self.gen1data[opp_atk.species].items()
        #             ]
        #         )
        #         # dummy moves for padding
        #         opp_moves.append(
        #             [
        #                 self.embed_move(fainted=True) + [0]
        #                 for _ in range(
        #                     10 - num_moves
        #                 )  # max number of moves is 10 by Pidgeot
        #             ]
        #         )
        # for j in range(6 - len(opp_team)):  # unrevealed
        #     for k in range(6):
        #         opp_moves.append(
        #             [
        #                 self.embed_move(
        #                     fainted=True,  # treat unrevealed as fainted for moves
        #                 )
        #                 + [0]
        #                 for _ in range(10)
        #             ]
        #         )

        return np.array(
            self.embed_side(battle.active_pokemon)
            + flatten_list(our_team_pkmn)
            + flatten_list(flatten_list(our_moves))
            + self.embed_side(battle.opponent_active_pokemon)
            + flatten_list(opp_team_pkmn)
            # + flatten_list(flatten_list(opp_moves))
        )
