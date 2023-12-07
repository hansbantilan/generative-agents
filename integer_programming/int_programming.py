import pandas as pd

from integer_programming.solve import IntegerProgrammingSolver


def load_data():
    # tltv data
    df = pd.read_csv("tmp/tltv.csv")
    # gl profiles
    tours = pd.read_json("tmp/MASTER_tours_data.json")
    # recommendations by gpt
    recs = pd.read_csv("tmp/tour_recommendations_matrix.csv")

    return df, tours, recs


def process_data(df, tours, recs):
    tours = tours.T["GROUP_SIZE"].reset_index().rename(columns={"index": "Tour Family"})
    tours = tours[tours["GROUP_SIZE"] != "Not Available"]
    df = df.merge(tours, on="Tour Family", how="inner")
    df["GROUP_SIZE"] = df["GROUP_SIZE"].apply(lambda x: float(x))
    df["new_capacity"] = round(df["June Capacity"] * 900 / df["GROUP_SIZE"])
    df = df[df[" TLTV "] != "  "]
    df[" TLTV "] = df[" TLTV "].apply(lambda x: float(x.strip().replace(",", ".")))
    df = df.rename(columns={" TLTV ": "TLTV"})
    df["TLTV"] = df["TLTV"] * 1000
    recs = recs[0:100]  # drop the extra customer gpt produces at the end

    # take the tours that having tltv + GPT recommended
    tours_to_take = recs.columns[(recs.columns.isin(df["Tour Family"]))]
    recs = recs[["id"] + list(tours_to_take)]
    recs_matrix = recs.drop(columns="id")
    recs_list = recs_matrix.values.tolist()

    # sort tours by the same index in recommendations. This is crucial for not messing the indices
    df = df[df["Tour Family"].isin(recs.columns)]
    df["Tour Family"] = pd.Categorical(df["Tour Family"], recs.columns)
    df = df.sort_values("Tour Family")
    return df, recs, recs_list


if __name__ == "__main__":
    df, tours, recs = load_data()
    df, recs, recs_list = process_data(df, tours, recs)

    solver = IntegerProgrammingSolver(df, recs, recs_list)
    solver.solve()
