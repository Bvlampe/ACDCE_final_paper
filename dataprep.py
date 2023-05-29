import pandas as pd
import numpy as np


def query_yn(query="Write to file? y/n: "):
    answer = ""
    while answer not in ["y", "n"]:
        answer = input(query)
    return True if answer == "y" else False


def list_countries_per_set(in_dataset, name_dataset, io_list, name_column="Country", in_index=False, in_header=False):
    assert((not(in_header and in_index)) and "Error: list_countries_per_set() was called with both in_index and in_header set as True")
    i = 0
    countries = []
    if in_index:
        countries = in_dataset.index.unique()
    elif in_header:
        countries = in_dataset.columns.unique()
    else:
        countries = in_dataset.loc[:, name_column].unique()
    for country in countries:
        io_list.loc[i, name_dataset] = country
        i += 1


# Takes the country names in the main dataset (as list-like variable) and a DataFrame of country names in
# the other datasets (each column being a list of all countries from one specific dataset, with the column
# name being the name of the dataset) and exports an Excel file with those countries not matching up between the
# datasets with the goal to create a concordance table from it


def create_country_table(main, in_data, write=False):
    first = list(set(main))
    first.sort()
    out = pd.DataFrame(index=range(len(list(first))), columns=["Main"], data=list(first))
    diff_countries = []

    for column in in_data.columns:
        second = set(in_data.loc[:, column])
        both = set(first).intersection(second)
        # Necessary for some reason because a nan value managed to slip through in the Fragility dataset
        unique = [x for x in list(second - both) if type(x) == str]
        unique.sort()
        diff_countries.extend(unique)

    diff_countries = list(set(diff_countries))
    diff_countries.sort()
    i = 0
    for ctry in diff_countries:
        out.loc[i, "Non-matching"] = ctry
        i += 1
    if write:
        out.loc[:, ["Non-matching", "Main"]].to_csv("country_names.csv", index=False)


def update_table(main, in_data):
    old_table = pd.read_csv("concordance_table.csv")
    first = list(set(main))
    first.sort()
    out = pd.DataFrame(index=range(len(list(first))), columns=["Main", "Non-matching"])
    out.loc[:, "Main"] = list(first)
    diff_countries = []

    for column in in_data.columns:
        second = set(in_data.loc[:, column])
        both = set(first).intersection(second)
        # Necessary for some reason because a nan value managed to slip through in the Fragility dataset
        unique = [x for x in list(second - both) if type(x) == str]
        unique.sort()
        for newctry in unique:
            if newctry not in old_table.loc[:, "Non-matching"].values:
                diff_countries.append(newctry)

    diff_countries = list(set(diff_countries))
    diff_countries.sort()
    i = 0
    for ctry in diff_countries:
        out.loc[i, "Non-matching"] = ctry
        i += 1
    print(out)
    print("Number of new non-matching:", len(out.loc[:,"Non-matching"].unique()) - 1)
    if query_yn():
        out.loc[:, ["Non-matching", "Main"]].to_csv("updated_country_names.csv", index=False)


def country_dict():
    source = pd.read_csv("concordance_table.csv")
    out = {}
    for _, row in source.iterrows():
        out[row.loc["Non-matching"]] = row.loc["Rename"]
    return out


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


def generic_table_transform(in_data, in_index, var_name, ctry_name="Country"):
    print(var_name, ':')
    totalrows = len(in_data.index)
    done = 0
    print_freq = int(totalrows / 10)

    out = pd.DataFrame(index=in_index, columns=[var_name])
    years = [col for col in in_data.columns if col.isdigit()]

    for _, row in in_data.iterrows():
        if not done % print_freq:
            print(100 * done / totalrows, '%')

        country = row.loc[ctry_name]
        for year in in_index.get_level_values(1):
            if str(year) in years:
                out.loc[(country, year), var_name] = row.loc[str(year)]
        done += 1
    return out


def rename_countries(io_data, in_dict, ctry_name="Country", in_index=False, in_header=False,
                     drop_missing=True, leave_groups=False):
    assert ((not (
                in_header and in_index)) and "Error: rename_countries() was called with both in_index and in_header set as True")
    if in_index:
        for country in io_data.index:
            if country in in_dict.keys():
                newname = in_dict[country]
                if newname == "None" and drop_missing:
                    io_data.drop(country, inplace=True)
                elif newname == "Region" and drop_missing:
                    io_data.drop(country, inplace=True)
                else:
                    io_data.rename(index={country: newname}, inplace=True)
    elif in_header:
        for country in io_data.columns:
            if country in in_dict.keys():
                newname = in_dict[country]
                if newname == "None" and drop_missing:
                    io_data.drop(country, inplace=True, axis=1)
                elif newname == "Region" and drop_missing:
                    io_data.drop(country, inplace=True, axis=1)
                else:
                    io_data.rename(columns={country: newname}, inplace=True)

    else:
        for i in range(len(io_data.index)):
            country = io_data.loc[i, ctry_name]
            if leave_groups and (country in ["Arab League", "AU", "CIS", "ECOWAS", "EU", "NATO", "UN"]):
                continue
            if country in in_dict.keys():
                newname = in_dict[country]
                if newname == "None" and drop_missing:
                    io_data.loc[i, ctry_name] = None
                elif newname == "Region" and drop_missing:
                    io_data.loc[i, ctry_name] = None
                else:
                    io_data.loc[i, ctry_name] = newname
        if drop_missing:
            io_data.dropna(subset=ctry_name, inplace=True)


def calc_elec_frag(in_elecdata, in_index):
    out_data = pd.DataFrame(index=in_index, columns=["Political Contestation"])
    in_elecdata.loc[:, "Year"] = in_elecdata.loc[:, "Year"].astype(int)
    column_list = ['p' + str(num + 1) + 'v' for num in range(715)]
    for _, row in in_elecdata.iterrows():
        found_any = False
        frag = 1
        for col in column_list:
            if not np.isnan(row.loc[col]):
                found_any = True
                vote_share = row.loc[col] / row.loc["ElectionTotalVotes"]
                frag -= vote_share ** 2
        ctry = row.loc["Country"]
        year = row.loc["Year"]
        if found_any:
            out_data.loc[(ctry, year), "Political Contestation"] = frag
    return out_data


def dataprep(step="merge"):
    path_all = "datasets_input/"
    path_dem = path_all + "dur_dem.csv"
    path_FH = path_all + "FH_data.csv"
    path_elec = path_all + "GlobalElections_Election_results.csv"
    path_turnout = path_all + "turnout.csv"
    path_pop = path_all + "population.csv"

    raw_dem = pd.read_csv(path_dem).loc[:, ["country", "year", "polity2"]].rename(columns={"polity2": "Democracy"}).rename(str.capitalize, axis="columns")
    raw_FH = pd.read_csv(path_FH, header=[0, 1], index_col=0, encoding="cp1252")
    raw_elec = pd.read_csv(path_elec).rename(columns={"cnty": "Country", "year": "Year"})
    raw_turnout = pd.read_csv(path_turnout)
    raw_pop = pd.read_csv(path_pop).rename(columns={"Country Name": "Country"})

    main_index_ctry = raw_dem.loc[:, "Country"].unique()
    main_index_ctry.sort()
    main_index_year = range(raw_dem.loc[:, "Year"].min(), raw_dem.loc[:, "Year"].max() + 1)
    main_index = pd.MultiIndex.from_product([main_index_ctry, main_index_year], names=["Country", "Year"])

    if step == "create_dict":
        cntry_names = pd.DataFrame()
        cntry_names["Elec"] = raw_elec.loc[:, "Country"].unique()
        list_countries_per_set(raw_FH, "FreedomHouse", cntry_names, in_index=True)
        list_countries_per_set(raw_turnout, "Turnout", cntry_names)
        list_countries_per_set(raw_pop, "Population", cntry_names)
        create_country_table(main_index.get_level_values(0), cntry_names, write=query_yn())

    elif step == "update_dict":
        cntry_names = pd.DataFrame()
        cntry_names["Population"] = raw_pop.loc[:, "Country"].unique()
        update_table(main_index.get_level_values(0), cntry_names)

    elif step == "merge":
        concordance_table = country_dict()
        rename_countries(raw_dem, concordance_table)
        rename_countries(raw_elec, concordance_table)
        rename_countries(raw_FH, concordance_table, in_index=True)
        rename_countries(raw_turnout, concordance_table)
        rename_countries(raw_pop, concordance_table)

        main_data = generic_list_transform(raw_dem, main_index, "Democracy")
        slice_elec = calc_elec_frag(raw_elec, main_index)
        slice_FH = format_FH(raw_FH, main_index)
        slice_turnout = generic_list_transform(raw_turnout, main_index, "Turnout", column_name="Voter Turnout")
        slice_votes = generic_list_transform(raw_turnout, main_index, "Votes", column_name="Total vote")
        slice_pop = generic_table_transform(raw_pop, main_index, "Population")
        print(slice_elec.dropna())