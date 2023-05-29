import pandas as pd
import numpy as np

def format_FH(in_data, in_index):
    out_data = pd.DataFrame(index=in_index, columns=["FH_pol", "FH_civ"])
    last_year = ""
    for column in in_data:
        var = column[1].strip()

        if var == "Status":
            continue
        colvals = in_data.loc[:, column]
        if "Unnamed" not in column[0]:
            last_year = column[0]
            # The columns with indicated years starting with "August" or "November"
            # are attributed to the following year
            if last_year[0] in ["A", "N"]:
                last_year = last_year[-4:]

        if last_year.isnumeric() and int(last_year) in in_index.get_level_values(1):
            for country, row in in_data.iterrows():
                if var == "PR":
                    out_data.loc[(country, int(last_year)), "FH_pol"] = row[column]
                elif var == "CL":
                    out_data.loc[(country, int(last_year)), "FH_civ"] = row[column]
        # This is the only column attributed to two years, as it encompasses the entirety of 1981
        # and the majority of 1982
        elif last_year == "Jan.1981-Aug. 1982":
            if var == "PR":
                out_data.loc[(country, 1981), "FH_pol"] = row[column] if '-' not in row[column] else np.nan
                out_data.loc[(country, 1982), "FH_pol"] = row[column] if '-' not in row[column] else np.nan
            elif var == "CL":
                out_data.loc[(country, 1981), "FH_civ"] = row[column] if '-' not in row[column] else np.nan
                out_data.loc[(country, 1982), "FH_civ"] = row[column] if '-' not in row[column] else np.nan
    out_data.replace(to_replace={"FH_civ": '-', "FH_pol": '-'}, value=np.nan, inplace=True)
    return out_data

def generic_list_transform(in_data, in_index, var_name, column_name=None, year_name="Year", ctry_name="Country"):
    print(var_name, ':')
    totalrows = len(in_data.index)
    done = 0
    print_freq = int(totalrows / 10)

    if column_name is None:
        column_name = var_name
    out = pd.DataFrame(index=in_index, columns=[var_name])
    for _, row in in_data.iterrows():
        if not done % print_freq:
            print(100 * done / totalrows, '%')
        ctry = row.loc["Country"]
        year = row.loc["Year"]
        slice1 = in_data.loc[in_data[year_name] == year, :]
        slice2 = slice1.loc[in_data[ctry_name] == ctry, :]
        # assert(len(slice2.loc[:, column_name].values <= 1) and f"Error in generic_list_transform(): more than one value detected for cell {ctry}, {year} in {var_name}")
        value = slice2.loc[:, column_name].values[0]
        out.loc[(ctry, year), var_name] = value
        done += 1
    return out

def dataprep():
    path_all = "datasets_input/"
    path_dem = path_all + "dur_dem.csv"
    path_FH = path_all + "FH_data.csv"
    path_elec = path_all + "GlobalElections_Election_results.csv"

    raw_dem = pd.read_csv(path_dem).loc[:, ["country", "year", "polity2"]].rename(columns={"polity2": "Democracy"}).rename(str.capitalize, axis="columns")
    raw_FH = pd.read_csv(path_FH, header=[0, 1], index_col=0, encoding="cp1252")
    raw_elec = pd.read_csv(path_elec)

    print(raw_elec)
