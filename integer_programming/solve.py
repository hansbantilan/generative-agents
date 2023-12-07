import math

import numpy as np
import pandas as pd
from scipy.optimize import linprog


class IntegerProgrammingSolver:
    def __init__(self, df, recs, recs_list):
        self.df = df
        self.recs = recs
        self.recs_list = recs_list
        self.num_tours = df.shape[0]
        self.num_gl = recs.shape[0]
        self.A_ub = []
        self.A_eq = np.zeros((self.num_gl, self.num_tours * self.num_gl))
        self.bounds_list = []
        self.coefs = df["TLTV"]
        self.c_array = np.tile(np.array(self.coefs), self.num_gl)
        self.b_eq = np.mat([1] * 100)
        self.b_ub = self.df["new_capacity"].to_list()

    # fill upper bound matrix
    def fill_A_ub(self):
        for k in range(self.num_tours):
            self.A_ub.append(
                [
                    1
                    if i in [self.num_tours * i + k for i in range(self.num_tours)]
                    else 0
                    for i in range(self.num_gl * self.num_tours)
                ]
            )

    # fill equality matrix
    def fill_A_eq(self):
        for i in range(0, self.num_gl):
            for j in range(0, self.num_tours):
                self.A_eq[i, i * self.num_tours + j] = self.recs_list[i][j]

    # fill bounds

    def fill_bounds(self):
        for i in range(0, self.num_gl):
            for j in range(0, self.num_tours):
                if self.A_eq[i, i * self.num_tours + j] == 1:
                    self.bounds_list.append(((0, 1)))
                else:
                    self.bounds_list.append(((0, 0)))

    def solve(self):
        print("solve")
        self.fill_A_ub()
        self.fill_A_eq()
        self.fill_bounds()
        results = linprog(
            c=self.c_array,
            A_eq=self.A_eq,
            b_eq=self.b_eq,
            A_ub=self.A_ub,
            b_ub=self.b_ub,
            bounds=self.bounds_list,
            method="simplex",
        )
        print(f"Objective value: z* = {results.fun}")
        print(results.x)

        recs_dic = {}
        for k, v in enumerate(results.x):
            if v == 1:
                recs_dic[math.floor((k / self.num_tours))] = self.df.iloc[
                    k % self.num_tours
                ]["Tour Family"]

        # check if you haven't messed up with the indices
        for k in range(0, self.num_gl):
            assert self.recs.loc[k][recs_dic[k]] == 1

        modified_recs_dic = {f"id_{k}": v for k, v in recs_dic.items()}
        pd.DataFrame([modified_recs_dic]).to_csv("tmp/results_of_ip.csv")
