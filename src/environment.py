import numpy as np


class Environment:
    def __init__(self, games=[], players1=[], players2=[], iterations=200):
        self.players1 = players1
        self.players2 = players2
        self.games = games
        self.agentsRewards = []
        self.agentsActions = []
        self.agentsProbabilities = []
        self.iterations = iterations

    def reset_agents(self):
        self.players1 = []
        self.players2 = []

    def reset_games(self):
        self.games = []

    def add_game(self, game):
        self.games.append(game)

    def add_player(self, player):
        if player.player_type == 0:
            self.players1.append(player)
        elif player.player_type == 1:
            self.players2.append(player)

    def reward(self, game, player, action_player1, action_player2):
        r = game[player, action_player1, action_player2]
        return r

    def random_matching(self):
        if len(self.players1) == len(self.players2):
            players2idx = [i for i in range(len(self.players2))]
            np.random.shuffle(players2idx)
            playersList = []
            for i in range(len(self.players1)):
                playersList.append((self.players1[i], self.players2[players2idx[i]]))
            return playersList

    def play2_agents(self, game, player1, player2):
        action1 = player1.take_action()
        action2 = player2.take_action()
        R1 = self.reward(game, 0, action1, action2)
        R2 = self.reward(game, 1, action1, action2)
        player1.get_reward_for_action(R1)
        player2.get_reward_for_action(R2)

    def step(self, game):
        random_matched_pair_list = self.random_matching()
        for player1, player2 in random_matched_pair_list:
            self.play2_agents(game, player1, player2)

    def play(self):
        # np.random.seed(1)
        games_history = []
        if self.games and self.players1 and self.players2:
            for game in self.games:
                agents_rewards = []
                agents_actions = []
                agents_probabilities = []
                agentsQ = []
                for _ in range(self.iterations):
                    self.step(game)
                for i, j in zip(self.players1, self.players2):
                    i.final_t = i.t
                    j.final_t = j.t
                    agents_rewards.append((i.reward_history, j.reward_history))
                    agents_actions.append((i.action_history, j.action_history))
                    agents_probabilities.append(
                        (i.probability_history, j.probability_history)
                    )
                    agentsQ.append((i.Qhistory, j.Qhistory))
                    i.reset()
                    j.reset()
                games_history.append(
                    (agents_actions, agents_rewards, agents_probabilities, agentsQ)
                )
        return games_history
