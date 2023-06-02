import pandas as pd
import numpy as np

def dataexpl():
    main_data = pd.read_csv("merged_data.csv", index_col=[0, 1])

    time_lag = 5
    print(f"Time lag {time_lag} year(s):")
    for ctry, ctry_data in main_data.groupby(axis=0, level=0):
        main_data.loc[ctry, "Democracy change"] = main_data["Democracy"].shift(-time_lag) - main_data["Democracy"]

    for col in ["Political Contestation", "Eligible population"]:
        main_data[col] = main_data.groupby(level=0)[col].apply(
            lambda group: group.fillna(method="ffill", limit=10)).droplevel(0)

    for var in ["Political Contestation", "Eligible population", "FH_civ"]:
        print(f"Cross-section with {var}:")
        before = len(main_data)
        new = main_data.dropna(subset=["Democracy change", var]).loc[:, ["Democracy change"]]
        print(f"Dropped {before - len(new)} out of {before} rows, kept {len(new)}.")

        decrease = len(new[new["Democracy change"] > 0])
        increase = len(new[new["Democracy change"] < 0])
        stasis = len(new[new["Democracy change"] == 0])

        print("Decrease:", decrease, "or as %:", decrease / len(new))
        print("Increase:", increase, "or as %:", increase / len(new))
        print("Stasis", stasis, "or as %:", stasis / len(new))
        print("Provisional total:", decrease + increase + stasis)
        print("-----------------------------------------")
