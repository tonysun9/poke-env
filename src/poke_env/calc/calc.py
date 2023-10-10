import requests
from typing import List

from pydantic import BaseModel

class DamageCalcResult(BaseModel):
    damage: List[int]
    # recoilInfo: dict
    # recoveryInfo: dict
    # kochanceInfo: dict


def get_active_calc(gen, battle):
    dmg_moves = {k: v for k, v in battle.active_pokemon.moves.items() if v.base_power > 0}
    
    data = {
        "gen": gen,
        "attackingPokemon": battle.active_pokemon.species,
        "defendingPokemon": battle.opponent_active_pokemon.species,
        "moveNames": list(dmg_moves.keys()),
    }

    response = requests.post("http://localhost:3000/calculate", json=data)
    json = response.json()
    try:
        for obj in json:
            if type(obj["damage"]) == int:
                obj["damage"] = [obj["damage"]]
    except:
        print(json)
    # json_class = DamageCalcResult.model_validate(json)
    return {move: DamageCalcResult.model_validate(dmg_result) for move, dmg_result in zip(dmg_moves.values(), json)}