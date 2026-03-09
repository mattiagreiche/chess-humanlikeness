# ELO data researched and compiled by Gemini 3.0 Pro. This is fine for these purposes:
# the task is purely data lookup (Wikipedia / Chessmetrics), not anything that requires
# understanding the codebase or the Maia-2 methodology.

import os
import pandas as pd
import numpy as np
from scipy import stats
import matplotlib.pyplot as plt

# PC1 scores from the 3D PCA across opening/mid/endgame humanness dimensions.
player_values = {
  "Petrosian": -0.029867739452514554,
  "Bronstein": -0.029090668765969926,
  "Botvinnik": -0.025793476499218863,
  "Larsen": -0.021771939118902215,
  "Karpov": -0.021321996224841086,
  "Short": -0.019447373622596912,
  "Adams": -0.019127099486822455,
  "Keres": -0.017690439194166244,
  "Chigorin": -0.015112713562393494,
  "Topalov": -0.012061448356695937,
  "Korchnoi": -0.011050759949934137,
  "Timman": -0.008620458964200913,
  "Rubinstein": -0.007793218417440242,
  "Morozevich": -0.0068798771152101925,
  "Spassky": -0.006699834281442824,
  "Smyslov": -0.0056941082524894744,
  "Caruana": -0.005617362732762601,
  "Gashimov": -0.005201729478802777,
  "Giri": -0.0038387571230199236,
  "Aronian": -0.0037446747860010406,
  "Ivanchuk": -0.0037230081209816,
  "Najdorf": -0.0034170779076469425,
  "Radjabov": -0.0031705754446163968,
  "PolgarJ": -0.003075287678078096,
  "Nakamura": -0.0021801667859136144,
  "Reshevsky": -0.0017564943726231363,
  "Geller": -0.001184754990345327,
  "Kramnik": -0.0006671557444404665,
  "Tarrasch": 5.943395528137736e-05,
  "Bogoljubow": 0.0003261952737978034,
  "Leko": 0.0008207385310539086,
  "Kasparov": 0.001480098022445683,
  "Lasker": 0.0017925750134714644,
  "Karjakin": 0.0021124288469787545,
  "Alekhine": 0.0032771680196169413,
  "Anand": 0.003513298519146801,
  "Polugaevsky": 0.00567281847387948,
  "Nepo": 0.006756636469680274,
  "Gelfand": 0.007152356460727664,
  "Hou": 0.00964555427385956,
  "Tal": 0.010639518535555394,
  "Carlsen": 0.011797498941567468,
  "Schlechter": 0.012344387635296893,
  "Grischuk": 0.012801072746687169,
  "Svidler": 0.014114993085909114,
  "Ding": 0.014660585343892406,
  "So": 0.01553829197619289,
  "Marshall": 0.01862287172483794,
  "Lagrave": 0.020007147464150908,
  "Kosteniuk": 0.024871551809762648,
  "Euwe": 0.030343731915987382,
  "Fischer": 0.031412474427107095,
  "Shirov": 0.035836768963184884
}

# Peak ratings. Modern players (active post-1990): peak FIDE. Historical players: Chessmetrics.
# Sources: Wikipedia "List of chess players by peak FIDE rating";
#          Chessmetrics (chessmetrics.com) for pre-FIDE-era players.
elos = {
    "Carlsen": 2882, "Kasparov": 2851, "Caruana": 2844, "Aronian": 2830, "So": 2822,
    "Mamedyarov": 2820, "Vachier-Lagrave": 2819, "Lagrave": 2819,
    "Anand": 2817, "Kramnik": 2817, "Topalov": 2816, "Nakamura": 2816, "Ding": 2816,
    "Grischuk": 2810, "Giri": 2798, "Nepo": 2795, "Radjabov": 2793, "Karjakin": 2788,
    "Morozevich": 2788, "Ivanchuk": 2787, "Fischer": 2785, "Karpov": 2780,
    "Gelfand": 2777, "Leko": 2763, "Adams": 2761, "Svidler": 2769, "Shirov": 2755,
    "Short": 2718, "PolgarJ": 2735, "Hou": 2686, "Kosteniuk": 2540, "Gashimov": 2761,

    # Pre-FIDE / early-FIDE — Chessmetrics peaks
    "Petrosian": 2645, # FIDE peak (deflated; see elos_adjusted below)
    "Botvinnik": 2848, "Bronstein": 2792, "Keres": 2786, "Reshevsky": 2785,
    "Rubinstein": 2789, "Najdorf": 2766, "Bogoljubow": 2768, "Tarrasch": 2763,
    "Lasker": 2878, "Alekhine": 2860, "Schlechter": 2764, "Capablanca": 2877,
    "Marshall": 2726, "Reti": 2710, "Euwe": 2769, "Geller": 2765, "Chigorin": 2715,
    "Larsen": 2660, "Spassky": 2690, "Timman": 2680, # FIDE peaks (deflated)
    "Korchnoi": 2695, "Polugaevsky": 2640, "Smyslov": 2620, "Tal": 2705,
}

# FIDE ratings from the 1970s-80s are deflated by ~100-150 pts vs. Chessmetrics.
# Swap the worst offenders to CM peaks so the regression isn't skewed by era inflation.
elos_adjusted = elos.copy()
elos_adjusted.update({
    "Petrosian": 2796, "Larsen": 2755, "Spassky": 2773, "Korchnoi": 2814,
    "Smyslov": 2800, "Tal": 2799, "Polugaevsky": 2726, "Timman": 2720,
    "Karpov": 2848, "Fischer": 2881,
})

df = pd.DataFrame(list(player_values.items()), columns=['Player', 'Value'])
df['MaxELO'] = df['Player'].map(elos_adjusted)

missing = df[df['MaxELO'].isnull()]
print("Missing ELOs:", missing['Player'].tolist())

df_clean = df.dropna()

slope, intercept, r_value, p_value, std_err = stats.linregress(df_clean['Value'], df_clean['MaxELO'])

print(f"R-squared: {r_value**2:.4f}")
print(f"Correlation: {r_value:.4f}")
print(f"P-value: {p_value:.4f}")
print(f"Slope: {slope:.4f}")
print(f"Intercept: {intercept:.4f}")

plt.figure(figsize=(10, 6))
plt.scatter(df_clean['Value'], df_clean['MaxELO'], alpha=0.7)
plt.plot(df_clean['Value'], intercept + slope * df_clean['Value'], 'r', label=f'fit ($R^2 = {r_value**2:.4f}$)')
for i, txt in enumerate(df_clean['Player']):
    plt.annotate(txt, (df_clean['Value'].iloc[i], df_clean['MaxELO'].iloc[i]), fontsize=8)
plt.xlabel('PC1 score')
plt.ylabel('Peak ELO (adjusted)')
plt.title('Linear Regression: Player Value vs Max ELO')
plt.legend()

os.makedirs('outputs', exist_ok=True)
plt.savefig('outputs/regression_skill_plot.png')
