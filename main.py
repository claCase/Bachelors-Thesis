import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
from pandas.plotting import table
from src.agent import Agent
from src.environment import Environment
from src.analyzer import (
    averages,
    plot_data,
    grid_search,
    mean_squared_error,
    show_params_space,
    plot_avg_distance,
    plot_average_run,
    open_results,
    load_obj,
)
import argparse
import os
import pickle

parser = argparse.ArgumentParser()
parser.add_argument("--simulate", type=bool, help="Run simulation :bool", default=False)
parser.add_argument("--iter", type=int, help="number of iterations :int", default=200)
parser.add_argument(
    "--agents", type=int, help="number of agents per player type :int", default=2
)
parser.add_argument(
    "--strategy",
    type=str,
    help='strategy of agent: "max"| "boltzmann" | "entropy" :string',
    default="entropy",
)
parser.add_argument(
    "--d_expl", type=bool, help="Decrease Exploration :bool", default=True
)
parser.add_argument(
    "--exp",
    type=bool,
    help="Whether to decrease exploration exponentially or linearly :bool",
    default=False,
)
parser.add_argument(
    "--decay_rate",
    type=float,
    help="Rate of exploration decrease of exponential or linear decay :float",
    default=0.005,
)
parser.add_argument(
    "--save", type=bool, help="save game histories :bool", default=False
)
parser.add_argument("--runs", type=int, help="number of simulation runs", default=1)
parser.add_argument(
    "--save_fig",
    type=bool,
    help="whether to save simulations images :bool",
    default=False,
)
parser.add_argument(
    "--show_fig",
    type=bool,
    help="whether to show simulations images :bool",
    default=False,
)
parser.add_argument(
    "--sum_rewards",
    type=bool,
    help="whether to plot sum of rewards or rewards per iteration :bool",
    default=False,
)
parser.add_argument(
    "--grid_search",
    type=bool,
    help="grid search of optimal parameter :bool",
    default=False,
)
parser.add_argument(
    "--plot_exp",
    type=bool,
    help="whether to show experimental games players' probabilities :bool",
)
parser.add_argument(
    "--t", type=float, help="temperature parameter of agents :float", default=2
)
parser.add_argument(
    "--a", type=float, help="alpha parameter of agents :float", default=0.1
)
parser.add_argument(
    "--open",
    type=int,
    help="show best parameter runs in folder n of games histories :int",
)
parser.add_argument("--intervals", type=int, help="search intervals :int", default=10)
parser.add_argument(
    "--begin", type=int, help="beginning time from which to calculate MSD", default=0
)
parser.add_argument(
    "--mse",
    type=int,
    help="Calculate mean square deviation of histories in file config.pkl in folder n :int",
)
parser.add_argument(
    "--from_game", type=int, help="If mse then begin from this game", default=0
)
parser.add_argument(
    "--to_game", type=int, help="If mse then end to this game :int", default=11
)
parser.add_argument(
    "--equilibria", type=bool, help="Show all equilibrium in plot :bool", default=False
)
parser.add_argument(
    "--compare",
    type=int,
    help="MSD comparison of constant and non-constant sum games of file config.pkl in folder n :int",
    default=False,
)
parser.add_argument("--plot_avg", type=int, help="Plot equilibrium results")

args = parser.parse_args()
simulate = args.simulate
iterations = args.iter
agents = args.agents
strategy = args.strategy
d_exp = args.d_expl
exp_decay = args.exp
decay_rate = args.decay_rate
save = args.save
runs = args.runs
save_fig = args.save_fig
show_fig = args.show_fig
grid = args.grid_search
sum_rewards = args.sum_rewards
experimental = args.plot_exp
open_ = args.open
intervals = args.intervals
begin = args.begin
mse = args.mse
from_game = args.from_game
to_game = args.to_game
equilibria = args.equilibria
compare_games = args.compare
plot_avg = args.plot_avg
t = args.t if args.t > 0 else 1.6
a = args.a if 0 < args.a < 1 else 0.4

strategy_ = {"entropy": 2, "boltzmann": 1, "max": 0}
strategy = strategy_[strategy]


def main():
    games = [
        [[[10, 0], [9, 10]], [[8, 18], [9, 8]]],
        [[[9, 0], [6, 8]], [[4, 13], [7, 5]]],
        [[[8, 0], [7, 10]], [[6, 14], [7, 4]]],
        [[[7, 0], [5, 9]], [[4, 11], [6, 2]]],
        [[[7, 0], [4, 8]], [[2, 9], [5, 1]]],
        [[[7, 1], [3, 8]], [[1, 7], [5, 0]]],
        [[[10, 4], [9, 14]], [[12, 22], [9, 8]]],
        [[[9, 3], [6, 11]], [[7, 16], [7, 5]]],
        [[[8, 3], [7, 13]], [[9, 17], [7, 4]]],
        [[[7, 2], [5, 11]], [[6, 13], [6, 2]]],
        [[[7, 2], [4, 10]], [[4, 11], [5, 1]]],
        [[[7, 3], [3, 10]], [[3, 9], [5, 0]]],
    ]
    # add Bimatrix Games
    env = Environment()
    env.iterations = iterations
    for game in games:
        env.add_game(np.asarray(game))

    if save:
        n = 1
        folder = os.path.join(os.getcwd(), "data", "game_histories")
        directory = os.path.join(folder, str(n))
        path = os.path.exists(directory)
        while path:
            n += 1
            directory = os.path.join(folder, str(n))
            path = os.path.exists(directory)
        os.mkdir(directory)
        print("saving directory: {}".format(directory))
    if grid:
        if exp_decay:
            range_decay = (0.9999, 0.94)
        else:
            range_decay = (0.001, 0.01)
        t_init = (0.5, 1.8)
        t_final = (0.1, 1)
        points_t_final = 3
        points_t_init = intervals
        points_decay = 1
        a_range = (0.01, 0.65)
        best_fit, parameter_space = grid_search(
            t_init,
            points_t_init,
            a_range,
            intervals,
            range_decay,
            points_decay,
            t_final,
            points_t_final,
            env,
            runs,
            agents,
            kwargs_agent={
                "d_exp": d_exp,
                "exponential_decay": exp_decay,
                "strategy": strategy,
            },
            kwargs_averages={"begin": begin},
        )
        mse = best_fit["mse"]
        print("Mean Squared Error {}".format(mse))
        print("params {}".format(best_fit["params"]))
        histories = best_fit["history"]
        if save:
            with open(os.path.join(directory, "config.pkl"), "wb") as f:
                pickle.dump(best_fit, f, pickle.HIGHEST_PROTOCOL)
            np.save(os.path.join(directory, "params.npy"), parameter_space)
            print("history saved to {}".format(os.path.join(directory, "params.npy")))
            open_results(directory)
        if show_fig:
            plot_average_run(histories, save_fig, directory)
            plt.show()
    elif simulate:
        for _ in range(agents):
            env.add_player(
                Agent(
                    strategy=strategy,
                    t=t,
                    alpha=a,
                    player_type=0,
                    exponential_decay=exp_decay,
                    d_exp=d_exp,
                    decay_rate=decay_rate,
                )
            )
            env.add_player(
                Agent(
                    strategy=strategy,
                    t=t,
                    alpha=a,
                    player_type=1,
                    exponential_decay=exp_decay,
                    d_exp=d_exp,
                    decay_rate=decay_rate,
                )
            )
        histories = []
        for _ in range(runs):
            history = env.play()
            histories.append(history)
        mse = mean_squared_error(histories)
        print("MSD {}".format(mse))
        if show_fig:
            plot_average_run(histories, save_fig)
            plt.show()
        # save history
        if save:
            np.save(os.path.join(directory, "histories.npy"), histories)
            print(
                "history saved to {}".format(os.path.join(directory, "histories.npy"))
            )
        if save_fig:
            directory = os.path.join(os.getcwd(), "figures")

    if experimental:
        plot_data()

    if save:
        return directory


if __name__ == "__main__":
    print("START")
    directory = main()
    if open_:
        open_results(open_, compare=equilibria)
        print("open")

    if plot_avg:
        directory = os.path.join("data", "game_histories", str(plot_avg))
        histories = load_obj(directory)["history"]
        if save_fig:
            save_dir = directory
        else:
            save_dir = None
        plot_avg_distance(histories, compare=equilibria, save_dir=save_dir)
        avg1, avg2, _, _ = averages(histories, 2, begin=begin, average_runs=True)
        avg1 = np.mean(avg1, axis=1)
        avg2 = np.mean(avg2, axis=1)
        from data.load_csv import return_data

        data_exp = return_data()
        avg_data_exp = np.mean(data_exp, axis=2)
        avg_ = np.empty((12, 4))
        print(avg1.shape, avg_data_exp.shape)
        avg_[:, 0] = avg1
        avg_[:, 1] = avg_data_exp[:, 0]
        avg_[:, 2] = avg2
        avg_[:, 3] = avg_data_exp[:, 1]
        df1 = pd.DataFrame(
            data=avg_,
            index=["game" + str(i + 1) for i in range(12)],
            columns=[
                "Player1 simulated",
                "Player1 experimental",
                "Player2 simulated",
                "Player2 experimental",
            ],
        )
        print(df1.head())
        if save_fig:
            df1 = df1.round(3)
            fig, ax = plt.subplots(figsize=(10, 5))  # set size frame
            ax.xaxis.set_visible(False)  # hide the x axis
            ax.yaxis.set_visible(False)  # hide the y axis
            ax.set_frame_on(False)  # no visible frame, uncomment if size is ok
            tabla = table(
                ax, df1, loc="center", colWidths=[0.20] * len(df1.columns)
            )  # where df is your data frame
            tabla.auto_set_font_size(False)  # Activate set fontsize manually
            tabla.set_fontsize(12)  # if ++fontsize is necessary ++colWidths
            tabla.scale(1.2, 1.2)  # change size table
            plt.savefig(
                os.path.join(directory, "games_equilibrium_begin-{}.png".format(begin))
            )
            print(
                "Equilibriums probabilities table saved to {}".format(
                    os.path.join(
                        directory, "games_equilibrium_begin-{}.png".format(begin)
                    )
                )
            )
            np.save(
                os.path.join(directory, "games_equilibrium_begin-{}.npy".format(begin)),
                df1.to_numpy,
            )

    if mse:
        directory = os.path.join("data", "game_histories", str(mse))
        histories = load_obj(directory)["history"]
        mse_mat = np.empty((12, 3))
        for i in range(12):
            if i == 11:
                j = None
            else:
                j = i + 1
            mse, T, S = mean_squared_error(
                histories, from_game=i, to_game=j, begin=begin
            )
            mse_mat[i] = [mse, T, S]
        df1 = pd.DataFrame(
            data=mse_mat,
            index=["game" + str(i + 1) for i in range(12)],
            columns=["MSD", "T", "S"],
        )
        print(df1.head())
        print("mse game{}: {:.4f} {:.4f} {:.4f}".format(i, mse, T, S))
        if save_fig:
            df1 = df1.round(5)
            fig, ax = plt.subplots(figsize=(5, 5))  # set size frame
            ax.xaxis.set_visible(False)  # hide the x axis
            ax.yaxis.set_visible(False)  # hide the y axis
            ax.set_frame_on(False)  # no visible frame, uncomment if size is ok
            tabla = table(
                ax, df1, loc="center", colWidths=[0.17] * len(df1.columns)
            )  # where df is your data frame
            tabla.auto_set_font_size(False)  # Activate set fontsize manually
            tabla.set_fontsize(12)  # if ++fontsize is necessary ++colWidths
            tabla.scale(1.2, 1.2)  # change size table
            plt.savefig(
                os.path.join(directory, "games_total_MSD_begin-{}.png".format(begin))
            )
            print(
                "MSD figure saved to {}".format(
                    os.path.join(
                        directory, "games_total_MSD_begin-{}.png".format(begin)
                    )
                )
            )
            np.save(os.path.join(directory, "games_MSD.npy"), df1.to_numpy)
        if show_fig:
            plt.show()

    if compare_games:
        import pandas as pd
        from pandas.plotting import table

        directory = os.path.join("data", "game_histories", str(compare_games))
        histories = load_obj(directory)["history"]
        mse_c, T_c, S_c = mean_squared_error(
            histories, from_game=0, to_game=5, begin=begin
        )
        mse_n, T_n, S_n = mean_squared_error(
            histories, from_game=5, to_game=11, begin=begin
        )
        mse_t, T_t, S_t = mean_squared_error(histories, begin=begin)
        plt.figure()
        tcn = plt.bar([0, 1], [S_c + T_c, S_n + T_n])
        scn = plt.bar([0, 1], [S_c, S_n])
        plt.xticks([0, 1], ("Constant Games", "Non-constant Games"))
        plt.legend([scn, tcn], ("Sampling Variance", "Theory Specific Component"))
        if save_fig:
            plt.savefig(
                os.path.join(
                    directory, "games_const_non_bar_begin-{}.png".format(begin)
                )
            )

        df1 = pd.DataFrame(
            {"MSD": [mse_c, mse_n, mse_t], "T": [T_c, T_n, T_t], "S": [S_c, S_n, S_t]}
        )
        df1 = df1.round(5)
        df1.index = ["Constant", "Nonconstant", "Total"]
        fig, ax = plt.subplots(figsize=(5, 2))  # set size frame
        ax.xaxis.set_visible(False)  # hide the x axis
        ax.yaxis.set_visible(False)  # hide the y axis
        ax.set_frame_on(False)  # no visible frame, uncomment if size is ok
        tabla = table(
            ax, df1, loc="upper right", colWidths=[0.17] * len(df1.columns)
        )  # where df is your data frame
        tabla.auto_set_font_size(False)  # Activate set fontsize manually
        tabla.set_fontsize(12)  # if ++fontsize is necessary ++colWidths
        tabla.scale(1.2, 1.2)  # change size table
        if save_fig:
            plt.savefig(os.path.join(directory, "games_MSD_begin-{}.png".format(begin)))
            print(
                "MSD figure saved to {}".format(
                    os.path.join(directory, "games_MSD_begin-{}.png".format(begin))
                )
            )

    if show_fig:
        plt.show()
    print("END")
