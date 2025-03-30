import random

class RandomAgent:
    def __str__(self):
        return "Random Agent"

    def getMove(self, problem):
        return None  # No moves to make

class RuleBasedAgent:
    def __str__(self):
        return "Cheating Agent"

    def getMove(self, problem):
        problem.player_rolls[problem.current_player] = 6
        return None

class MinimaxAgent:
    def __str__(self):
        return "Predicting Agent"

    def getMove(self, problem):
        if problem.rolls_made == 1:
            problem.player_rolls[1] = random.randint(1,6) + 1
            if problem.player_rolls[1] > 6:
                problem.player_rolls[1] = 6
        return None

class QLearningAgent:
    def __init__(self, learning_rate=0.1, discount_factor=0.9, exploration_rate=1.0):
        self.learning_rate = learning_rate
        self.discount_factor = discount_factor
        self.exploration_rate = exploration_rate
        self.q_table = {}

    def __str__(self):
        return "Q-Learning Agent"

    def getMove(self, problem):
        state = str(problem.rolls_made)
        if state not in self.q_table:
            self.q_table[state] = [0] * 6

        if random.random() < self.exploration_rate:
            roll = random.randint(1, 6)
        else:
            roll = self.q_table[state].index(max(self.q_table[state])) + 1

        problem.player_rolls[problem.current_player] = roll

        if problem.rolls_made == 1:
            reward = problem.getWinner()
            self.q_table[state][roll - 1] = (1 - self.learning_rate) * self.q_table[state][roll - 1] + self.learning_rate * (reward + self.discount_factor * max(self.q_table[str(problem.rolls_made)]))

        self.exploration_rate *= 0.995 

        return None