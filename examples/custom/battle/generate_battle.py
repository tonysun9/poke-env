from poke_env.environment.abstract_battle import AbstractBattle
from poke_env.environment.battle import Battle

from poke_env.player import RandomPlayer, cross_evaluate

from poke_env.ps_client.account_configuration import AccountConfiguration

import asyncio

BATTLE_FORMAT = "gen1randombattle"


random_player = RandomPlayer(
    battle_format=BATTLE_FORMAT,
    max_concurrent_battles=10,
    account_configuration=AccountConfiguration("RP Test", ""),
)

random_player_2 = RandomPlayer(
    battle_format=BATTLE_FORMAT,
    max_concurrent_battles=10,
    account_configuration=AccountConfiguration("RP Test 2", ""),
)


async def main():
    await cross_evaluate(
        [random_player, random_player_2],
        n_challenges=1,
    )


asyncio.run(main())
# battle = Battle(
#     battle_tag="test",
#     username="test",
#     logger=None,
#     gen=1,
# )

# print(battle)
# print(battle.team)
