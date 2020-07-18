import numpy as np
import matplotlib.pyplot as plt
import os
from data.load_csv import return_data
import tqdm

data_path = os.path.join(os.getcwd(), "data", "numpy_data.npy")


def averages(
    histories, what, begin=0, end=None, average_runs=True, from_game=0, to_game=None
):
    """
		histories: list of games history from Environment.play()
		what: 0|1|2|3 = actions|rewards|probabilities|Qvalues
		begin: begin iteration index
		end: end iteration index
		average_runs: True|False, returns average of runs
	"""
    average_runs1, average_runs2 = [], []
    for run in histories:
        gamePlayersAvg1, gamePlayersAvg2 = [], []
        for game in run[from_game:to_game]:
            avgPlayers1, avgPlayers2 = [], []
            for player in game[what]:
                avgPlayers1.append(player[0][begin:end])
                avgPlayers2.append(player[1][begin:end])
            gamePlayersAvg1.append(np.mean(avgPlayers1, axis=0))
            gamePlayersAvg2.append(np.mean(avgPlayers2, axis=0))
        average_runs1.append(gamePlayersAvg1)
        average_runs2.append(gamePlayersAvg2)
    variance_runs1 = np.std(average_runs1, axis=0)
    variance_runs2 = np.std(average_runs2, axis=0)
    if average_runs:
        average_runs1 = np.mean(average_runs1, axis=0)
        average_runs2 = np.mean(average_runs2, axis=0)
    return average_runs1, average_runs2, variance_runs1, variance_runs2


def expected_trajectory(histories, **kwargs):
    avg = averages(histories, what=2, **kwargs)
    return np.mean(avg, axis=1)


def mean_squared_error(histories, from_game=0, to_game=None, **kwargs):
    data_games = return_data()[from_game:to_game]
    avg_data_games = np.mean(data_games, axis=2)[np.newaxis, :]
    avg1, avg2, _, _ = averages(
        histories,
        from_game=from_game,
        to_game=to_game,
        what=2,
        average_runs=False,
        **kwargs
    )
    runs = len(avg1)
    avg1 = np.mean(avg1, axis=2)
    avg2 = np.mean(avg2, axis=2)
    avg_run1, avg_run2, _, _ = averages(
        histories,
        from_game=from_game,
        to_game=to_game,
        what=2,
        average_runs=True,
        **kwargs
    )
    avg_run1 = np.mean(avg_run1, axis=1)[np.newaxis, :]
    avg_run2 = np.mean(avg_run2, axis=1)[np.newaxis, :]

    var_theory_spec1 = np.square(avg_run1 - avg_data_games[:, :, 0]).flatten()
    var_theory_spec2 = np.square(avg_run2 - avg_data_games[:, :, 1]).flatten()

    sampling_variance1 = np.square(avg1 - avg_run1).flatten()
    sampling_variance2 = np.square(avg2 - avg_run2).flatten()

    T = np.sum((var_theory_spec1, var_theory_spec2))
    S = np.sum((sampling_variance1, sampling_variance2)) / runs
    mse = T + S
    return mse, T, S


def grid_search(
    range_t,
    points_t,
    range_a,
    points_a,
    range_decrease,
    points_decrease,
    range_final,
    points_final,
    environment,
    runs,
    agents,
    kwargs_agent={},
    kwargs_averages={},
):
    from src.agent import Agent
    from time import time

    best_fit = {
        "mse": 100000,
        "T": 100000,
        "S": 0,
        "history": None,
        "params": {},
        "agents": agents,
        "range_t": range_t,
        "range_a": range_a,
        "points_a": points_a,
        "points_t": points_t,
        "points_decrease": points_decrease,
        "range_decrease": range_decrease,
        "range_final": range_final,
        "points_final": points_final,
        "runs": runs,
        "kwargs_agent": kwargs_agent,
        "kwargs_averages": kwargs_averages,
    }
    interval_a = np.linspace(range_a[0], range_a[1], points_a)
    interval_t = np.linspace(range_t[0], range_t[1], points_t)
    interval_e = np.linspace(range_decrease[0], range_decrease[1], points_decrease)
    interval_final_t = np.linspace(range_final[0], range_final[1], points_final)
    parameter_space = np.empty((points_t, points_a, points_decrease, points_final))
    begin_time = time()
    progress = tqdm.tqdm(
        total=points_a * points_t * points_decrease * points_final * runs
    )
    for i, t in enumerate(interval_t):  # initial t
        for j, a in enumerate(interval_a):  # alpha
            for z, e in enumerate(interval_e):  # exploration decay
                for k, final_t in enumerate(interval_final_t):  # final t
                    for _ in range(agents):
                        environment.add_player(
                            Agent(
                                player_type=0,
                                decay_rate=e,
                                alpha=a,
                                t=t,
                                final_t=final_t,
                                **kwargs_agent
                            )
                        )
                        environment.add_player(
                            Agent(
                                player_type=1,
                                decay_rate=e,
                                alpha=a,
                                t=t,
                                final_t=final_t,
                                **kwargs_agent
                            )
                        )
                    histories = []
                    for _ in range(runs):
                        history = environment.play()
                        histories.append(history)
                        progress.update(n=1)
                    mse, T, S = mean_squared_error(histories, **kwargs_averages)
                    parameter_space[i, j, z, k] = mse
                    if best_fit["T"] > T:
                        best_fit["mse"] = mse
                        best_fit["T"] = T
                        best_fit["S"] = S
                        best_fit["history"] = histories
                        best_fit["params"] = {
                            "t": t,
                            "a": a,
                            "e": e,
                            "final_t": environment.players1[0].final_t,
                            "ijzk": [i, j, z, k],
                        }
                    environment.reset_agents()

    progress.close()
    end_time = time()
    delta = end_time - begin_time
    print("\nTime elapsed {:.2f} s".format(delta))
    return best_fit, parameter_space


def plot_data():
    data = return_data()
    for i in range(12):
        p = plt.subplot(12, 1, i + 1)
        plt.plot(np.arange(200), data[i, 0, :], np.arange(200), data[i, 1, :])
        plt.xlim(0, 200)
        plt.ylim(0, 1)
        if i != 11:
            p.set_xticks([])
        plt.ylabel("Game {}".format(i + 1))
        if i == 0:
            plt.title("Experimental Probabilities")
            plt.legend(
                bbox_to_anchor=(0.9, 2),
                loc="upper left",
                borderaxespad=0.0,
                labels=["Player1", "Player2"],
            )


def show_params_space(
    directory,
    range_x,
    range_y,
    labels=["\u03C4", "\u03B1"],
    slice_=[0, slice(0, None), 0, slice(0, None)],
):
    img = np.load(os.path.join(directory, "params.npy"))[slice_]
    x_ticks = np.linspace(range_x[0], range_x[1], img.shape[0])
    x_ticks = ["%.3f" % x for x in x_ticks]
    y_ticks = np.linspace(range_y[0], range_y[1], img.shape[1])
    y_ticks = ["%.3f" % y for y in y_ticks]
    plt.figure(figsize=(10, 10))
    plt.imshow(img[:, ::-1].T)
    plt.colorbar()
    plt.title("Heat map of MSD")
    plt.xlabel(labels[0], fontsize=15)
    plt.xticks(np.arange(img.shape[0]), x_ticks)
    plt.ylabel(labels[1], fontsize=15)
    plt.yticks(np.arange(img.shape[1])[::-1], y_ticks)
    plt.xlim()
    path = os.path.join(
        directory, "params_space" + str(labels[0]).replace(" ","") + str(labels[1]).replace(" ","") + ".png"
    )
    plt.savefig(path)
    print("image saved to {}".format(path))


def plot_avg_distance(histories, save_dir=None, compare=True, **kwargs):
    avg1, avg2, _, _ = averages(histories, what=2, **kwargs)
    avg1 = np.mean(avg1, axis=1)
    avg2 = np.mean(avg2, axis=1)
    data_games = return_data()
    data_games = np.mean(data_games, axis=2)
    fg, ax_ls = plt.subplots(4, 3, figsize=(7 * 1.5, 9 * 1.5))
    NE = np.asarray(
        [
            [0.091, 0.909],
            [0.181, 0.727],
            [0.273, 0.909],
            [0.364, 0.818],
            [0.364, 0.727],
            [0.455, 0.636],
        ]
    )
    NE = np.vstack((NE, NE))
    QE = np.asarray(
        [
            [0.07, 0.882],
            [0.172, 0.711],
            [0.25, 0.898],
            [0.348, 0.812],
            [0.354, 0.721],
            [0.449, 0.634],
        ]
    )
    QE = np.vstack((QE, QE))
    AS = np.asarray(
        [
            [0.057, 0.664],
            [0.185, 0.619],
            [0.137, 0.753],
            [0.283, 0.679],
            [0.286, 0.679],
            [0.448, 0.613],
        ]
    )
    AS = np.vstack((AS, AS))
    IB = np.asarray(
        [
            [0.104, 0.634],
            [0.258, 0.561],
            [0.188, 0.764],
            [0.304, 0.724],
            [0.354, 0.646],
            [0.466, 0.604],
        ]
    )
    IB = np.vstack((IB, IB))

    for (
        i,
        ((ea1, ea2), (ne1, ne2), (qe1, qe2), (as1, as2), (ib1, ib2), a1, a2, ax),
    ) in enumerate(zip(data_games, NE, QE, AS, IB, avg1, avg2, ax_ls.flat)):
        ax.set_aspect("equal", "box")
        ax.set_title("Game {}".format(i + 1), fontsize=12)
        ax.set_xlim(0, 1)
        ax.set_ylim(0, 1)
        ax.scatter(a1, a2, color="blue", label="SE")
        ax.scatter(ea1, ea2, color="orange", label="EE")
        if compare:
            ax.scatter(ne1, ne2, color="green", s=14.5)
            ax.scatter(qe1, qe2, color="red", s=14.5)
            ax.scatter(as1, as2, color="black", s=14.5)
            ax.scatter(ib1, ib2, color="purple", s=14.5)
        ax.set_xlabel("player 1", fontsize=10)
        ax.set_ylabel("player 2", fontsize=10)

    ax_ls[0, 1].legend(
        loc="lower left",
        bbox_to_anchor=(0, 1.2),
        borderaxespad=0.0,
        labels=[
            "Simulated",
            "Experimental",
            "Nash",
            "Quantal Response",
            "Action Sampling",
            "Impulse Balance",
        ],
        fontsize=11,
    )
    fg.tight_layout()
    if save_dir:
        if compare:
            name = "avg_dist_compare.png"
        else:
            name = "avg_dist.png"
        plt.savefig(os.path.join(save_dir, name))
        print("figure saved in {}".format(os.path.join(save_dir, name)))


def plot_average_run(histories, save_fig, directory=None):
    avgP1, avgP2, varP1, varP2 = averages(histories=histories, what=2)
    # print(avgP1[0].shape)
    # avgR1, avgR2, varR1, varR2 = averages(histories = histories, what = 1, average_runs=True)

    fg, ax_ls = plt.subplots(len(histories[0]), 1, figsize=(15, 12))
    for index, axis in enumerate(ax_ls):
        if index < 11:
            axis.set_xticklabels([])

    ax_ls[0].set_title("Average Players Rewards")
    ax_ls[0].set_title("Average Players Probabilities")
    ax_ls[0].set_xlabel("iterations")
    ax_ls[0].set_xlabel("iterations")

    """
	#plot rewards
	for index, (game_p1, game_p2) in enumerate(zip(avgR1, avgR2)):
		if sum_rewards:
			ax_ls[index,0].plot(np.arange(len(game_p1)), np.cumsum(game_p1) , np.arange(len(game_p2)), np.cumsum(game_p2))
		else:
			ax_ls[index,0].plot(np.arange(len(game_p1)), game_p1, np.arange(len(game_p2)), game_p2)
		ax_ls[index, 0].set_xlim(0,200)
		ax_ls[index,0].set_ylabel("Game{}".format(index+1))
	"""
    # plot probabilities
    for index, (game_p1, game_p2, var1, var2) in enumerate(
        zip(avgP1, avgP2, varP1, varP2)
    ):
        ax_ls[index].set_xlim(0, 200)
        ax_ls[index].set_ylim(bottom=-0.1, top=1.1)
        ax_ls[index].plot(
            np.arange(len(game_p1)), game_p1, np.arange(len(game_p2)), game_p2
        )
        ax_ls[index].fill_between(
            np.arange(len(game_p1)), game_p1 - var1, game_p1 + var1, alpha=0.3
        )
        ax_ls[index].fill_between(
            np.arange(len(game_p1)), game_p2 - var2, game_p2 + var2, alpha=0.3
        )
        ax_ls[index].set_ylabel("Game{}".format(index + 1))
    ax_ls[0].legend(
        bbox_to_anchor=(0.85, 1.5),
        loc="lower left",
        borderaxespad=0.0,
        labels=["Player1", "Player2"],
    )

    if save_fig and directory:
        plt.savefig(os.path.join(directory, "RP.png"))
    else:
        n = 0
        folder = os.path.join(os.getcwd(), "figures")
        path = os.path.exists(
            os.path.join(
                folder,
                "RP_run"
                + str(len(histories))
                + "_ag"
                + str(len(histories[0][0][0]))
                + "_"
                + str(n)
                + ".png",
            )
        )
        while path:
            n += 1
            path = os.path.exists(
                os.path.join(
                    folder,
                    "RP_run",
                    str(len(histories))
                    + "_ag"
                    + str(len(histories[0][0][0]))
                    + "_"
                    + str(n)
                    + ".png",
                )
            )
        plt.savefig(
            os.path.join(
                folder,
                "RP_run"
                + str(len(histories))
                + "_ag"
                + str(len(histories[0][0][0]))
                + "_"
                + str(n)
                + ".png",
            )
        )

    plt.figure(1)
    fg, ax_ls = plt.subplots(4, 3, figsize=(8, 11))
    # fg.suptitle("Probability Dynamics")
    fg.tight_layout()
    indexes = np.arange(12).reshape(4, 3)
    for i in range(4):
        for j in range(3):
            ax_ls[i, j].set_aspect("equal", "box")
            ax_ls[i, j].set_ylim(bottom=0, top=1)
            ax_ls[i, j].set_xlim(left=0, right=1)
            ax_ls[i, j].plot(
                avgP1[indexes[i, j]], avgP2[indexes[i, j]], "--", linewidth=0.5
            )
            ax_ls[i, j].set_ylabel("player 2")
            ax_ls[i, j].set_xlabel("player 1")
            ax_ls[i, j].set_title("Game{}".format(indexes[i, j] + 1))
            # ax_ls[i, j].scatter(data[0,:], data[1,:], s=0.5, color="r")
            # ax_ls[i, j].plot(data[0,:], data[1,:], linewidth=0.5, color="r")
            """
			length = len(avgP1[indexes[i,j]])
			avp1 = np.asarray(avgP1[indexes[i,j]]).reshape(-1,1)
			avp2 = np.asarray(avgP2[indexes[i,j]]).reshape(-1,1)
			for t, (avp1, avp2) in enumerate(zip(avp1, avp2)):
				ax_ls[i,j].scatter( avp1, avp2, c = np.asarray(t).reshape(1,), s = 10, alpha = 1-t/lenght)
			"""
    if save_fig and directory:
        plt.savefig(os.path.join(directory, "Dy.png"))
    else:
        n = 0
        folder = os.path.join(os.getcwd(), "figures")
        path = os.path.exists(
            os.path.join(
                folder,
                "Dy_run"
                + str(len(histories))
                + "_ag"
                + str(len(histories[0][0][0]))
                + "_"
                + str(n)
                + ".png",
            )
        )
        while path:
            n += 1
            path = os.path.exists(
                os.path.join(
                    folder,
                    "Dy_run"
                    + str(len(histories))
                    + "_ag"
                    + str(len(histories[0][0][0]))
                    + "_"
                    + str(n)
                    + ".png",
                )
            )
        plt.savefig(
            os.path.join(
                folder,
                "Dy_run"
                + str(len(histories))
                + "_ag"
                + str(len(histories[0][0][0]))
                + "_"
                + str(n)
                + ".png",
            )
        )


def load_obj(directory):
    import pickle

    with open(os.path.join(directory, "config.pkl"), "rb") as f:
        return pickle.load(f)


def open_results(open_, **kwargs):
    print("open")
    if type(open_) == int:
        directory = os.path.join("data", "game_histories", str(open_))
    else:
        directory = open_
    obj = load_obj(directory)
    print(
        "params {} mse {:.5f} T {:.5f} S {:.5f}  agents {} runs {}".format(
            obj["params"], obj["mse"], obj["T"], obj["S"], obj["agents"], obj["runs"]
        )
    )
    histories = obj["history"]
    plot_avg_distance(histories, directory, **kwargs, begin=100)
    i = obj["params"]["ijzk"][0]
    j = obj["params"]["ijzk"][1]
    z = obj["params"]["ijzk"][2]
    k = obj["params"]["ijzk"][3]
    t = obj["params"]["t"]
    e = obj["params"]["e"]
    a = obj["params"]["a"]
    t_f = obj["params"]["final_t"]
    final_t = obj["params"]["final_t"]
    i_len = obj["points_t"]
    j_len = obj["points_a"]
    z_len = obj["points_decrease"]
    k_len = obj["points_final"]

    slice_init_t_alpha = (slice(0, None), slice(0, None), z, k)
    slice_alpha_final_t = (i, slice(0, None), z, slice(0, None))
    slice_e_final_t = (i, j, slice(0, None), slice(0, None))
    range_e = obj["range_decrease"]
    range_final_t = obj["range_final"]
    range_initial_t = obj["range_t"]
    range_a = obj["range_a"]
    range_t = obj["range_t"]
    labels_decay_final_t = ["decay coeff", "final \u03C4"]
    labels_t_init_a = ["initial \u03C4", "\u03B1"]
    labels_a_t_final = ["\u03B1", "final \u03C4"]
    show_params_space(
        directory,
        range_x=range_a,
        range_y=range_final_t,
        slice_=slice_alpha_final_t,
        labels=labels_a_t_final,
    )
    show_params_space(
        directory,
        range_x=range_initial_t,
        range_y=range_a,
        slice_=slice_init_t_alpha,
        labels=labels_t_init_a,
    )
    show_params_space(
        directory,
        range_x=range_e,
        range_y=range_final_t,
        slice_=slice_e_final_t,
        labels=labels_decay_final_t,
    )
    return histories
