import pandas as pd

from integer_programming.solve import IntegerProgrammingSolver

df = pd.read_csv("../../Downloads/tltv.csv")
tours = pd.read_json("../../Downloads/MASTER_tours_data.json")
tours = tours.T["GROUP_SIZE"].reset_index().rename(columns={"index": "Tour Family"})
tours = tours[tours["GROUP_SIZE"] != "Not Available"]
df = df.merge(tours, on="Tour Family", how="inner")
df["GROUP_SIZE"] = df["GROUP_SIZE"].apply(lambda x: float(x))
df["new_capacity"] = round(df["June Capacity"] * 900 / df["GROUP_SIZE"])
df = df[df[" TLTV "] != "  "]
df[" TLTV "] = df[" TLTV "].apply(lambda x: float(x.strip().replace(",", ".")))
df = df.rename(columns={" TLTV ": "TLTV"})
# read recommendations from gpt
recs = pd.read_csv("../../Downloads/tour_recommendations_matrix.csv")
recs = recs[0:100]  # drop the last customer gpt produces for extra

tours_to_take = recs.columns.drop("id")[
    (recs.columns.drop("id").isin(df["Tour Family"]))
]
recs = recs[["id"] + list(tours_to_take)]
recs_matrix = recs.drop(columns="id")
recs_list = recs_matrix.values.tolist()


# sort tours by the same index in recommendations
df = df[df["Tour Family"].isin(recs.columns)]
df["Tour Family"] = pd.Categorical(df["Tour Family"], recs.columns)
df = df.sort_values("Tour Family")


solver = IntegerProgrammingSolver(df, recs, recs_list)
solver.solve()
