import pandas as pd
import numpy as np

def dataexpl():
    main_data = pd.read_csv("merged_data.csv", index_col=[0, 1])

    for time_lag in range(1, 11):
        print(f"Time lag {time_lag} year(s):")
        for ctry, ctry_data in main_data.groupby(axis=0, level=0):
            main_data.loc[ctry, "Democracy change"] = main_data["Democracy"].shift(-time_lag) - main_data["Democracy"]
        # for (ctry, year) in main_data.index:
        #     demchg = main_data.loc[(ctry, year + time_lag), "Democracy"] -\
        #              main_data.loc[(ctry, year), "Democracy"] if year + time_lag \
        #              in main_data.index.get_level_values(1) else np.NaN
        #     main_data.loc[(ctry, year), "Democracy change"] = demchg

        before = len(main_data)
        new = main_data.loc[:, ["Democracy change"]].dropna(subset=["Democracy change"])
        print(f"Dropped {before - len(new)} out of {before} rows, kept {len(new)}.")

        decrease = len(new[new["Democracy change"] > 0])
        increase = len(new[new["Democracy change"] < 0])
        stasis = len(new[new["Democracy change"] == 0])

        print("Decrease:", decrease, "or as %:", decrease / len(new))
        print("Increase:", increase, "or as %:", increase / len(new))
        print("Stasis", stasis, "or as %:", stasis / len(new))
        print("Provisional total:", decrease + increase + stasis)
        print("-----------------------------------------")
