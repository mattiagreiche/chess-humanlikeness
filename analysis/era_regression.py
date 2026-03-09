# Peak year data researched and compiled by Gemini 3.0 Pro. This is fine for these purposes:
# the task is purely data lookup (chess history / FIDE profiles), not anything that requires
# understanding the codebase or the Maia-2 methodology.

import os
import pandas as pd
import numpy as np
from scipy import stats
import matplotlib.pyplot as plt

# PC1 scores from the 3D PCA across opening/mid/endgame humanness dimensions.
player_values = {
    "Petrosian": -0.03655890218270808,
    "Botvinnik": -0.02831855159230781,
    "Larsen": -0.02551277208007713,
    "Bronstein": -0.025146020202104113,
    "Karpov": -0.0242086462436965,
    "Short": -0.019498964370279658,
    "Topalov": -0.018625322145810804,
    "Keres": -0.017671997492752563,
    "Adams": -0.016389960193036644,
    "Radjabov": -0.014555672403622807,
    "Spassky": -0.01185105858817262,
    "Reshevsky": -0.01182056491487498,
    "Timman": -0.011741817251244836,
    "Gashimov": -0.01125072047521251,
    "Korchnoi": -0.010186423014066366,
    "Geller": -0.009811115599331243,
    "Chigorin": -0.00884347205450352,
    "Najdorf": -0.007969660106609027,
    "Rubinstein": -0.0058477512635118165,
    "PolgarJ": -0.00580199777271659,
    "Leko": -0.004436495150711735,
    "Morozevich": -0.004161911007519098,
    "Kasparov": -0.004015318946768375,
    "Giri": -0.004014371222145277,
    "Caruana": -0.0034908103393619027,
    "Ivanchuk": -0.0032833667992704773,
    "Smyslov": -0.003020770834086747,
    "Polugaevsky": -0.0007114612122780771,
    "Nakamura": 0.00025624625433507,
    "Bogoljubow": 0.0004292244366493649,
    "Tarrasch": 0.000909627098224832,
    "Kramnik": 0.0013972555362924794,
    "Aronian": 0.0024898370298325743,
    "Anand": 0.0025458413459444828,
    "Karjakin": 0.0028907495791759266,
    "Lasker": 0.003443761712008518,
    "Gelfand": 0.0037216324234071864,
    "Alekhine": 0.005518017485004903,
    "Nepo": 0.007740945364834903,
    "Tal": 0.008249992719324528,
    "Hou": 0.010471888040029468,
    "Schlechter": 0.012670725325916938,
    "Grischuk": 0.014022184309009237,
    "Ding": 0.014057604573825034,
    "Capablanca": 0.015193709212340851,
    "Svidler": 0.016746062415144557,
    "Carlsen": 0.017353109913791683,
    "So": 0.017586506670664217,
    "Lagrave": 0.0188734864850403,
    "Fischer": 0.022272162136568626,
    "Marshall": 0.02281104471534848,
    "Reti": 0.026317655945614294,
    "Kosteniuk": 0.03130900032321924,
    "Euwe": 0.03190033898302095,
    "Shirov": 0.037567285424213545
}

# Year of peak rating (or peak activity for pre-rating-era players).
# Sources: FIDE profile data; standard chess history references.
peak_years = {
    "Petrosian": 1963, # World Championship win
    "Botvinnik": 1948, # WC
    "Larsen": 1971, # Candidates match vs Fischer
    "Bronstein": 1951, # WC match vs Botvinnik
    "Karpov": 1994, # FIDE peak 2780 (Linares)
    "Short": 2004,
    "Topalov": 2006, # WC
    "Keres": 1955,
    "Adams": 2000,
    "Radjabov": 2012,
    "Spassky": 1969, # WC
    "Reshevsky": 1953, # Zurich Candidates
    "Timman": 1990,
    "Gashimov": 2011,
    "Korchnoi": 1979,
    "Geller": 1976,
    "Chigorin": 1895, # Hastings
    "Najdorf": 1947,
    "Rubinstein": 1912,
    "PolgarJ": 2005,
    "Leko": 2005,
    "Morozevich": 2008,
    "Kasparov": 1999, # Peak 2851
    "Giri": 2015,
    "Caruana": 2014, # Peak 2844
    "Ivanchuk": 2007,
    "Smyslov": 1956,
    "Polugaevsky": 1980,
    "Nakamura": 2015,
    "Bogoljubow": 1925,
    "Tarrasch": 1894,
    "Kramnik": 2016,
    "Aronian": 2014,
    "Anand": 2011,
    "Karjakin": 2011,
    "Lasker": 1894, # Start of WC reign; active at elite level until ~1920
    "Gelfand": 2013,
    "Alekhine": 1930,
    "Nepo": 2023,
    "Tal": 1960, # WC; iconic peak year (rating peak was 1980, but style-wise 1960)
    "Hou": 2018,
    "Schlechter": 1910,
    "Grischuk": 2014,
    "Ding": 2018,
    "Capablanca": 1921,
    "Svidler": 2006,
    "Carlsen": 2014, # Peak 2882
    "So": 2017,
    "Lagrave": 2016,
    "Fischer": 1972,
    "Marshall": 1914,
    "Reti": 1924,
    "Kosteniuk": 2016,
    "Euwe": 1935,
    "Shirov": 1998,
}

df = pd.DataFrame(list(player_values.items()), columns=['Player', 'Value'])
df['Year'] = df['Player'].map(peak_years)

missing = df[df['Year'].isnull()]
print("Missing Years:", missing['Player'].tolist())

df_clean = df.dropna()

slope, intercept, r_value, p_value, std_err = stats.linregress(df_clean['Value'], df_clean['Year'])

print(f"R-squared: {r_value**2:.4f}")
print(f"Correlation: {r_value:.4f}")
print(f"P-value: {p_value:.4f}")
print(f"Slope: {slope:.4f}")
print(f"Intercept: {intercept:.4f}")

plt.figure(figsize=(10, 6))
plt.scatter(df_clean['Value'], df_clean['Year'], alpha=0.7)
plt.plot(df_clean['Value'], intercept + slope * df_clean['Value'], 'r', label='fitted line')
for i, txt in enumerate(df_clean['Player']):
    plt.annotate(txt, (df_clean['Value'].iloc[i], df_clean['Year'].iloc[i]), fontsize=8)
plt.xlabel('PC1 score')
plt.ylabel('Year of peak activity')
plt.title('Linear Regression: Player Value vs Year of Peak')
plt.legend()

os.makedirs('outputs', exist_ok=True)
plt.savefig('outputs/regression_era_plot.png')
