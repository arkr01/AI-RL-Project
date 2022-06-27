import time
from random import random, choice
from game_state import GameState
from itertools import product
from numpy import cumsum

"""
solution.py

Template file for you to implement your solution to Assignment 3.

You must implement the following method stubs, which will be invoked by the simulator during 
testing:
    __init__(game_env)
    run_training()
    select_action()
    
To ensure compatibility with the autograder, please avoid using try-except blocks for Exception 
or OSError exception types. Try-except blocks with concrete exception types other than OSError
(e.g. try: ... except ValueError) are allowed.

COMP3702 2021 Assignment 3 Support Code

Last updated by njc 10/10/21

REFERENCES:
- Tutorial 9 Solution Code (For Q-Learning Implementation)
"""


def dict_argmax(d):
    """ Returns argmax from dictionary. FROM TUTORIAL 9 SOLUTION CODE. """
    return max(d, key=d.get)


def moving_average(a, n=50):
    """ FROM STACK OVERFLOW - Link: https://stackoverflow.com/questions/14313510/how-to-calculate
    -rolling-moving-average-using-python-numpy-scipy"""
    ret = cumsum(a, dtype=float)
    ret[n:] = ret[n:] - ret[:-n]
    return ret[n - 1:] / n


class RLAgent:

    def __init__(self, game_env, learning_rate=0.05, use_sarsa=True):
        """
        Constructor for your solver class.

        Any additional instance variables you require can be initialised here.

        Computationally expensive operations should not be included in the constructor,
        and should be placed in the plan_offline() method instead.

        This method has an allowed run time of 1 second, and will be terminated by the simulator if
        not completed within the limit.
        """
        self.game_env = game_env
        self.states = []
        self.learning_rate = learning_rate#0.05 if "a3-t2.txt" not in self.game_env.filename else 0.3
        self.discount = 0.9999
        self.epsilon = 0.1

        # For Q3/4
        self.num_episodes = 0
        self.total_rewards = [0 for _ in range(100000)]
        self.use_sarsa = use_sarsa

    def run_training(self):
        """
        This method will be called once at the beginning of each episode.

        You can use this method to perform training (e.g. via Q-Learning or SARSA).

        The allowed run time for this method is given by 'game_env.training_time'. The method
        will be terminated by the simulator if it does not complete within this limit - you
        should design your algorithm to ensure this method exits before the time limit is exceeded.
        """
        t0 = time.time()
        self.possible_gem_statuses = list(product((0, 1), repeat=self.game_env.n_gems))

        # Initialise list of states
        for r in range(1, self.game_env.n_rows - 1):
            for c in range(1, self.game_env.n_cols - 1):
                for gem_status in self.possible_gem_statuses:
                    if self.game_env.grid_data[r][c] not in self.game_env.COLLISION_TILES:
                        self.states.append(GameState(r, c, gem_status))

        # Find all valid actions for each state
        self.valid_actions = {state: [] for state in self.states}

        for state in self.states:
            for action in self.game_env.ACTIONS:
                if action in {self.game_env.WALK_LEFT, self.game_env.WALK_RIGHT,
                              self.game_env.JUMP}:
                    # check walkable ground prerequisite if action is walk or jump
                    if self.game_env.grid_data[state.row + 1][state.col] not in \
                            self.game_env.WALK_JUMP_ALLOWED_TILES:
                        continue
                    self.valid_actions[state].append(action)
                else:
                    # check permeable ground prerequisite if action is glide or drop
                    if self.game_env.grid_data[state.row + 1][state.col] not in \
                            self.game_env.GLIDE_DROP_ALLOWED_TILES:
                        continue
                    self.valid_actions[state].append(action)

        self.q_values = {state: {a: 0.0 for a in self.valid_actions[state]} for state in
                         self.states}
        self.persistent_state = self.random_restart()
        self.a = self.eps_greedy()  # For SARSA

        # optional: loop for ensuring your code exits before exceeding the reward target or time
        # limit
        while self.game_env.get_total_reward() > self.game_env.training_reward_tgt and \
                self.num_episodes < 100000:#time.time() - t0 < self.game_env.training_time - 1:
            #self.sarsa() if "a3-t2.txt" in self.game_env.filename else \
            # self.q_learning() if "a3-t2.txt" not in self.game_env.filename else self.sarsa()
            # self.q_learning()# if not self.use_sarsa else self.sarsa()
            self.sarsa() if self.use_sarsa else self.q_learning()
        self.rolling_averages = moving_average(self.total_rewards)

    def select_action(self, state):
        """
        This method will be called each time the agent is called upon to decide which action to
        perform (once for each step of the episode).

        You can use this method to select an action based on the Q-value estimates learned during
        training.

        The allowed run time for this method is 1 second. The method will be terminated by the
        simulator if it does not complete within this limit - you should design your algorithm to
        ensure this method exits before the time limit is exceeded.

        :param state: the current state, a GameState instance
        :return: action, the selected action to be performed for the current state
        """
        return dict_argmax(self.q_values[state])

    def q_learning(self):
        """ Implements a single iteration of Q-learning. HEAVILY BASED OFF OF TUTORIAL 9 CODE. """
        a = self.eps_greedy()
        _, r, next_state, terminal = self.game_env.perform_action(self.persistent_state, a)
        self.total_rewards[self.num_episodes] += r

        old_q = self.q_values[self.persistent_state][a]
        # old_q = q_s[a] if a in q_s else 0

        # Q(s', a') dict
        next_s_q = {}
        for action in self.valid_actions[next_state]:
            next_s_q[action] = 0.0
            if action in self.q_values[next_state]:
                next_s_q[action] = self.q_values[next_state][action]

        best_next_q = next_s_q[dict_argmax(next_s_q)] if not terminal else 0

        self.q_values[self.persistent_state][a] = old_q + self.learning_rate * (r + self.discount
                                                                                * best_next_q -
                                                                                old_q)
        self.persistent_state = next_state

        if terminal:
            # self.q_values[next_state] = {a: 0.0 for a in self.valid_actions[next_state]}
            self.persistent_state = self.random_restart()
            self.num_episodes += 1

    def sarsa(self):
        _, r, next_state, terminal = self.game_env.perform_action(self.persistent_state, self.a)
        self.total_rewards[self.num_episodes] += r

        # if terminal:
        #     self.q_values[next_state] = {a: 0 for a in self.valid_actions[next_state]}
        #     self.persistent_state = self.random_restart()
        #     self.a = self.eps_greedy()
        #     return

        old_q = self.q_values[self.persistent_state][self.a]
        # old_q = q_s[self.a] if self.a in q_s else 0

        # Q(s', a') dict
        # next_s_q = {}
        # for action in self.valid_actions[next_state]:
        #     next_s_q[action] = 0.0
        #     if action in self.q_values[next_state]:
        #         next_s_q[action] = self.q_values[next_state][action]

        current_state = self.persistent_state
        self.persistent_state = next_state
        next_a = self.eps_greedy()
        best_next_q = 0.0 if terminal else self.q_values[next_state][next_a]

        self.q_values[current_state][self.a] = old_q + self.learning_rate * (r + self.discount *
                                                                             best_next_q - old_q)
        self.a = next_a
        if terminal:
            # self.q_values[next_state] = {a: 0 for a in self.valid_actions[next_state]}
            self.persistent_state = self.random_restart()
            self.a = self.eps_greedy()
            self.num_episodes += 1

    def eps_greedy(self):
        if random() <= self.epsilon:
            return choice(self.valid_actions[self.persistent_state])
        else:
            return dict_argmax(self.q_values[self.persistent_state])

    def random_restart(self):
        """
        Restart the agent in a random (valid), non-terminal state
        """
        while True:
            next_state = choice(self.states)
            if self.game_env.is_game_over(next_state) or self.game_env.is_solved(next_state):
                continue
            return next_state
