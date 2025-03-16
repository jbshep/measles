import numpy as np
import pandas as pd

# Read in both CSV files.
pop_df = pd.read_csv("data/bogus-pop.csv")
vax_df = pd.read_csv("data/bogus-vax.csv")

# Merge them into one data set.

# Use new dataset to calculate population density and display
# state, vaccination rate, and population density.