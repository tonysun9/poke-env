import requests
from typing import List, Optional
import json

from pydantic import BaseModel

from poke_env.environment.move import Move
from poke_env.data.gen_data import GenData
from poke_env.environment.pokemon import Pokemon


TYPE_CHART = GenData(1).load_type_chart(1)

# TODO: should do this programatically with  "selfdestruct": "always" in gen1moves.json
DESTRUCT_MOVES = {"selfdestruct": -1, "explosion": -1}

EXCLUDE_DMG_MOVES = {"counter"}


clean_move = (
    lambda move: move.replace(" ", "")
    .replace("-", "")
    .replace(".", "")
    .replace("’", "")
    .lower()
)


class PokemonInfo(BaseModel):
    name: str
    level: int
    ivs: Optional[dict] = None
    evs: Optional[dict] = None


def get_1v1_calc(
    p1_pkmn: PokemonInfo, p2_pkmn: PokemonInfo, moves: List[str], gen: int = 1
):
    dmg_moves, no_effect_moves = [], []
    def_pkmn = Pokemon(species=p2_pkmn.name, gen=1)
    for move in moves:
        move_type = Move(move, 1).type.name

        def_type_1 = def_pkmn.type_1.name
        if TYPE_CHART[def_type_1][move_type] == 0:
            no_effect_moves.append(move)
            continue

        if def_pkmn.type_2:
            def_type_2 = def_pkmn.type_2.name
            if TYPE_CHART[def_type_2][move_type] == 0:
                no_effect_moves.append(move)

                continue

        dmg_moves.append(move)

    data = {
        "gen": gen,
        "attackingPokemon": p1_pkmn.name,
        "attackerLevel": p1_pkmn.level,
        "attackerIVs": p1_pkmn.ivs,
        "attackerEVs": p1_pkmn.evs,
        "defendingPokemon": p2_pkmn.name,
        "defenderLevel": p2_pkmn.level,
        "defenderIVs": p2_pkmn.ivs,
        "defenderEVs": p2_pkmn.evs,
        "moveNames": dmg_moves,
    }

    {
        "damageRange": [0, 0],
        "recoilInfo": {"recoil": [0, 0], "text": ""},
        "recoveryInfo": {"recovery": [0, 0], "text": ""},
    }

    response = requests.post(
        "http://localhost:3000/calculatefull", json=data, timeout=3
    )
    json = response.json()
    try:
        for obj in json:
            if type(obj["damage"]) == int:
                obj["damage"] = [obj["damage"]]
    except:
        # print(json)
        pass
    # json_class = DamageCalcResult.model_validate(json)
    # return {
    #     move: DamageCalcResult.model_validate(dmg_result)
    #     for move, dmg_result in zip(dmg_moves.values(), json)
    # }
    return json, dmg_moves


def calc_dmg(data):
    # counter, inner_counter = 0, 0

    dmg_calc_results = {}

    for attacker, atk_data in data.items():
        attacker = clean_move(attacker)

        attacker_info = PokemonInfo(
            name=attacker,
            level=atk_data["level"],
            ivs=atk_data["ivs"] if "ivs" in atk_data else None,
            evs=atk_data["evs"] if "evs" in atk_data else None,
        )

        moves = list(atk_data["moves"].keys())
        moves = [clean_move(move) for move in moves]
        dmg_moves = list(
            filter(lambda move: Move(move_id=move, gen=1).base_power > 0, moves)
        )
        dmg_moves = list(filter(lambda move: move not in EXCLUDE_DMG_MOVES, dmg_moves))

        atk_stats, _ = get_1v1_calc(
            attacker_info, attacker_info, []
        )  # just to get stats
        atk_hp = atk_stats["attackerStats"]["hp"]
        dmg_calc_results[attacker] = {
            "hp": atk_hp,
            "spe": atk_stats["attackerStats"]["spe"],
        }

        for defender, def_data in data.items():
            defender = clean_move(defender)

            defender_info = PokemonInfo(
                name=defender,
                level=def_data["level"],
                ivs=def_data["ivs"] if "ivs" in def_data else None,
                evs=def_data["evs"] if "evs" in def_data else None,
            )

            dmg_calc, dmg_moves_filtered = get_1v1_calc(
                attacker_info, defender_info, dmg_moves
            )

            def_hp = dmg_calc["defenderStats"]["hp"]
            dmg_calc_percent = {move: {} for move in dmg_moves_filtered}
            for i, entry in enumerate(dmg_calc["results"]):
                dmg_calc_percent[dmg_moves_filtered[i]]["damage"] = (
                    entry["damageRange"][0] / def_hp,
                    entry["damageRange"][1] / def_hp,
                )

                if (
                    type(entry["recoilInfo"]["recoil"]) == float
                ):  # highjumpkick has fixed recoil of 1hp if miss
                    recoil_lower = entry["recoilInfo"]["recoil"]
                    recoil_higher = entry["recoilInfo"]["recoil"]
                else:
                    recoil_lower = entry["recoilInfo"]["recoil"][0]
                    recoil_higher = entry["recoilInfo"]["recoil"][1]
                recoil_lower *= -1  # mark recoil as negative
                recoil_higher *= -1
                if recoil_lower != 0 or recoil_higher != 0:
                    dmg_calc_percent[dmg_moves_filtered[i]]["recoil"] = (
                        recoil_lower / 100,
                        recoil_higher / 100,
                    )  # recoil is already returned as percentage from damage calc
                if dmg_moves_filtered[i] in DESTRUCT_MOVES:
                    dmg_calc_percent[dmg_moves_filtered[i]]["recoil"] = (
                        DESTRUCT_MOVES[dmg_moves_filtered[i]],
                        DESTRUCT_MOVES[dmg_moves_filtered[i]],
                    )  # count selfdestruct and explosion as -1 recoil (lose 100% health)

                recovery_lower = entry["recoveryInfo"]["recovery"][0]
                recovery_higher = entry["recoveryInfo"]["recovery"][1]
                if recovery_lower != 0 or recovery_higher != 0:
                    dmg_calc_percent[dmg_moves_filtered[i]]["recovery"] = (
                        recovery_lower / atk_hp,
                        recovery_higher / atk_hp,
                    )

            for move in dmg_calc_percent:
                for calc in dmg_calc_percent[move]:
                    if (
                        dmg_calc_percent[move][calc][0] == 0.0
                        and dmg_calc_percent[move][calc][1] == 0.0
                    ):
                        continue
                    dmg_calc_percent[move][calc] = (
                        round(dmg_calc_percent[move][calc][0], 4),
                        round(dmg_calc_percent[move][calc][1], 4),
                    )

            dmg_calc_results[attacker][defender] = dmg_calc_percent

        #     inner_counter += 1
        #     if inner_counter == 2:
        #         break

        # counter += 1
        # if counter == 1:
        #     break

    with open("gen1randbats_dmg_calc.json", "w") as f:
        json.dump(dmg_calc_results, f, indent=4)


# def postprocess_calc_dmg(data):
#     with open("gen1randbats_dmg_calc.json", "r") as f:
#         dmg_calc = json.load(f)

#     for attacker in dmg_calc:
#         for defender in data[attacker]:
#             if defender == "hp" or defender == "spe":
#                 continue

#             for move in data[attacker][defender]:
#                 if move == "damage":
#                     continue

#                 if "recoil" not in data[attacker][defender][move]:
#                     data[attacker][defender][move]["recoil"] = (0.0, 0.0)

#                 if "recovery" not in data[attacker][defender][move]:
#                     data[attacker][defender][move]["recovery"] = (0.0, 0.0)

#     with open("gen1randbats_dmg_calc.json", "w") as f:
#         json.dump(data, f, indent=4)


if __name__ == "__main__":
    with open("gen1randbats.json", "r") as f:
        data = json.load(f)

    calc_dmg(data)
