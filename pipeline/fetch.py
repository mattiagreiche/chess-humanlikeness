import requests
import zipfile
import os
import io

player_names = [
    "Adams", "Alekhine", "Anand", "Anderssen", "Aronian", "Bogoljubow", "Botvinnik",
    "Bronstein", "Capablanca", "Carlsen", "Caruana", "Chigorin", "DeLaBourdonnais",
    "Ding", "Euwe", "Fine", "Fischer", "Gashimov", "Gelfand", "Geller", "Giri",
    "Grischuk", "Hou", "Ivanchuk", "Karjakin", "Karpov", "Kasparov", "Keres",
    "Korchnoi", "Kosteniuk", "Kramnik", "Larsen", "Lasker", "Leko", "Marshall",
    "Morozevich", "Morphy", "Najdorf", "Nakamura", "Nepomniachtchi", "Nimzowitsch",
    "Petrosian", "Philidor", "Pillsbury", "PolgarJ", "Polugaevsky", "Radjabov",
    "Reshevsky", "Reti", "Rubinstein", "Schlechter", "Shirov", "Short", "Smyslov",
    "So", "Spassky", "Staunton", "Steinitz", "Svidler", "Tal", "Tarrasch", "Timman",
    "Topalov", "VachierLagrave", "Zukertort"
]


# Output directory
output_dir = 'data/pgn'
os.makedirs(output_dir, exist_ok=True)

for player in player_names:
    url = f"https://www.pgnmentor.com/players/{player}.zip"
    print(f"Downloading {player}'s games from {url}...")
    response = requests.get(url)

    if response.status_code == 200:
        with zipfile.ZipFile(io.BytesIO(response.content)) as z:
            z.extractall(output_dir)
            print(f"Extracted {player}.zip")
    else:
        print(f"Failed to download {player}.zip (status code {response.status_code})")
