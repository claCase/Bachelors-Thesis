# Thesis
## Abstract
Reinforcement Learning is a framework into which many real-world problems can be casted. It is a learning paradigm which is based on iterative learning. It has shown great promise in the single agent setting, with applications to many fields and much of the research is focused on this setting (Kapoor, 2018). In this thesis we will present the multi-agent case, within which two agents play against each other and try to maximize their own reward. This is more challenging than the single agent setting because agents play against a non-stationary opponent. We apply this framework to the class of games called repeated normal form games. We first describe the general game setting and the requirements of the algorithm. We then introduce the theory behind the algorithm and its derivation. We then investigate how well this algorithm can approximate human learning and the resulting equilibrium that arises from those human interactions by comparing the simulation results to gameplays played by human players.
## Install 
to install required packages with pip: 
```console
pip install -r requirements.txt
```
to run the program: 
```console
python3 main.py
```
## Usage Examples: 
Run and show simulation 
```console
python3 .\main.py --simulate True --agents 4 --runs 2 --show_fig True
```
Search for optimal parameters and save best history and simulation parameters
```console
python3 .\main.py --grid_search True --save True --save_fig True --show_fig True --intervals 4 --agents 4 --runs 2 
```
## Optional commands: 
### Simulate Games 
```console
--simulate, type=bool, help="Run simulation :bool", default=False
--iter, type=int, help="number of iterations :int", default=200
--runs, type=int, number of simulation runs", default=1
--strategy, type=str, help="strategy of agent: "max"| "boltzmann"| "entropy" :string", default="entropy"
--agents, type=int, help="number of agents per player type :int", default=2
--t, type=float, temperature parameter of agents :float, default=2
--a, type=float, alpha parameter of agents :float, default=.1
--d_exp, type=bool, help="Decrease Exploration :bool", default=True
--exp, type=bool, Whether to decrease exploration exponentially or linearly :bool, default=False
--decay_rate, type=float, Rate of exploration decrease of exponential or linear decay :float
```
### Grid Search
```console
--grid_search, type=bool, grid search of optimal parameter :bool, default=False
--intervals, type=int, search intervals :int, default=10
```
### Analyze 
```console
--mse, type=int, help="Calculate mean square deviation of histories in file config.pkl in folder n :int"
--plot_exp, type=bool, whether to show experimental games players' probabilities :bool"
--open, type=int, show best parameter runs in folder n of games histories :int
--equilibria, type=bool, help="Show all equilibrium in plot :bool", default=False
--compare, type=int, help="MSD comparison of constant and non-constant sum games of file config.pkl in folder n :int", default=False
--begin, type=int, help="beginning iteration, default=0
--from_game, type=int, help="If mse then begin from this game", default=0
--to_game, type=int, help="If mse then end to this game :int", default=11
```
### Saving and Plotting
```console
--save, type=bool, save games histories :bool", default=False
--save_fig, type=bool, whether to save simulations images :bool", default=False
--show_fig, type=bool, whether to show simulations images :bool", default=False
--sum_rewards, type=bool, whether to plot sum of rewards or rewards per iteration :bool
--plot_avg, type=int, help="Plot equilibrium results"
```
## Equilibrium Results of 8 runs with 4 agents
![alt text](https://github.com/claCase97/Thesis/blob/master/data/game_histories/1/RP.png?raw=true)
![alt text](https://github.com/claCase97/Thesis/blob/master/data/game_histories/1/Dy.png?raw=true)
![alt text](https://github.com/claCase97/Thesis/blob/master/data/game_histories/1/avg_dist.png?raw=true)
![alt text](https://github.com/claCase97/Thesis/blob/master/data/game_histories/1/avg_dist_compare.png?raw=true)
![alt text](https://github.com/claCase97/Thesis/blob/master/data/game_histories/1/params_spaceinitialτα.png?raw=true)
![alt text](https://github.com/claCase97/Thesis/blob/master/data/game_histories/1/games_const_non_bar_begin-150.png?raw=true)
