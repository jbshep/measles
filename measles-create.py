import numpy as np
import pandas as pd

def rename_col(df, old_col, new_col):
    df[new_col] = df[old_col]
    del df[old_col]

# Read in both CSV files.
vax_data = pd.read_csv("data/cdc-schoolvax.csv")
census_data = pd.read_csv("data/census.csv")
area_data = pd.read_csv("data/kaggle-state-area.csv")

# Rename the column.
rename_col(vax_data, "Vaccine/Exemption", "Vax")
rename_col(vax_data, "Geography", "State")
rename_col(vax_data, "Estimate (%)", "Perc")

# Trim the data to only include MMR vaccinations and current year.
vax_data = vax_data[
    (vax_data["Vax"] == "MMR") &
    (vax_data["School Year"] == "2023-24")
]

vax_data = pd.merge(vax_data, census_data, left_on="State", right_on="NAME")
del vax_data["NAME"]
rename_col(vax_data, "POPESTIMATE2024", "Population")

vax_data = pd.merge(vax_data, area_data, left_on="State", right_on="state")
del vax_data["state"]
rename_col(vax_data, "land_area_sq_mi", "Area")

vax_data.to_csv("data/mmr-vax.csv", index=False)

