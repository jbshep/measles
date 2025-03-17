import numpy as np
import pandas as pd

# Read in both CSV files.
pop_df = pd.read_csv("data/bogus-pop.csv")
vax_df = pd.read_csv("data/bogus-vax.csv")

# Merge them into one data set.
all_df = pd.merge(pop_df, vax_df, left_on="Geography", right_on="State")
del all_df["Geography"]
print(all_df)

# Use new dataset to calculate population density and display
# state, vaccination rate, and population density.
all_df["Density"] = all_df["Population"] / all_df["Area"]
print(all_df[["State", "Vax Perc", "Density"]])