import requests

# URL of the endpoint
url = "http://localhost:3000/calculate"

# Sample request body
data = {
    "gen": 1,
    "attackingPokemon": "Pikachu",
    # "attackingPokemonOptions": {
    #     "level": 100,
    #     "nature": "Timid"
    # },
    "defendingPokemon": "Charizard",
    # "defendingPokemonOptions": {
    #     "level": 100,
    #     "nature": "Bold"
    # },
    "moveName": "Thunderbolt",
    # "field": {
    #     "weather": "Rain"
    # }
}

# Send GET request
response = requests.get(url, json=data)

# Print the response
print(response.status_code)
print(response.text)
# print(response.json())
