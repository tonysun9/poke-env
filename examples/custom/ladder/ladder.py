import asyncio
import os

from poke_env import AccountConfiguration, ShowdownServerConfiguration
from poke_env.player import RandomPlayer

import sys

sys.path.append("..")

from players.player_1v1 import MaxBase, MaxDamage

FORMAT = "gen1randombattle"
USERNAME = "cm_md2_battler"
PASSWORD = "CalmMindPoop"

REPLAY_FOLDER = USERNAME

if not os.path.exists(f"replays/{REPLAY_FOLDER}"):
    os.makedirs(f"replays/{REPLAY_FOLDER}", exist_ok=True)


async def main():
    # We create a random player
    # player = RandomPlayer(
    #     account_configuration=AccountConfiguration(USERNAME, PASSWORD),
    #     server_configuration=ShowdownServerConfiguration,
    #     battle_format=FORMAT,
    #     max_concurrent_battles=1,
    #     save_replays=True,
    #     replay_folder=f"replays/{REPLAY_FOLDER}",
    # )

    # player = MaxBase(
    #     account_configuration=AccountConfiguration(USERNAME, PASSWORD),
    #     server_configuration=ShowdownServerConfiguration,
    #     battle_format=FORMAT,
    #     max_concurrent_battles=1,
    #     save_replays=True,
    #     replay_folder=f"replays/{REPLAY_FOLDER}",
    # )

    player = MaxDamage(
        account_configuration=AccountConfiguration(USERNAME, PASSWORD),
        server_configuration=ShowdownServerConfiguration,
        battle_format=FORMAT,
        max_concurrent_battles=1,
        save_replays=True,
        with_calcs=True,
        replay_folder=f"replays/{REPLAY_FOLDER}",
    )

    # Sending challenges to 'your_username'
    # await player.send_challenges("your_username", n_challenges=1)

    # Accepting one challenge from any user
    # await player.accept_challenges(None, 1)

    # Accepting three challenges from 'your_username'
    # await player.accept_challenges('your_username', 3)

    # Playing 5 games on the ladder
    await player.ladder(n_games=3, logs=f"logs/{USERNAME}.txt")

    # Print the rating of the player and its opponent after each battle
    # for battle in player.battles.values():
    #     print(battle.rating, battle.opponent_rating)


if __name__ == "__main__":
    # asyncio.get_event_loop().run_until_complete(main())
    asyncio.run(main())
