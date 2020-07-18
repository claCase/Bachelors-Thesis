import pandas as pd
import os
import numpy as np


def load_data():
    path = os.path.join(os.getcwd(), "CSV_data.csv")
    data = pd.read_csv(path, sep=";", header=[0, 1])  # , index_col = [0])
    print(data.columns)
    games = []
    for i in range(12):
        games.append(
            (
                data["Game {}".format(i + 1)][:200]
                .set_index("Round")["mean(Up)"]
                .values,
                data["Game {}".format(i + 1)][:200]
                .set_index("Round")["mean(Left)"]
                .values,
            )
        )

    games = np.asarray(games)
    np.save(os.path.join(os.getcwd(), "numpy_data"), games)


def return_data():
    return np.load(
        os.path.join(os.getcwd(), "data", "numpy_data.npy"), allow_pickle=True
    )
