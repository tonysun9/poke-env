import requests
from typing import List

from pydantic import BaseModel

class DamageCalcResult(BaseModel):
    damage: List[int]
    # recoilInfo: dict
    # recoveryInfo: dict
    # kochanceInfo: dict

def get_1v1_calc(gen, p1_pkmn, p2_pkmn):
    dmg_moves = {k: v for k, v in p1_pkmn.moves.items() if v.base_power > 0}
    
    data = {
        "gen": gen,
        "attackingPokemon": p1_pkmn.species,
        "defendingPokemon": p2_pkmn.species,
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