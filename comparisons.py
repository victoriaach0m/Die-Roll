import random
import numpy as np
import time
from collections import defaultdict

random.seed(42)
np.random.seed(42)
num_games = 1000
# Game Defs
class DieRollGame:
    def __init__(self, target=10):
        self.target = target

    def reset(self):
        return (0, 0)

    def step(self, state, action):
        my_score, opp_score = state
        roll = random.randint(1, 6)
        my_score += roll
        done = my_score >= self.target
        reward = 1 if done else 0
        next_state = (opp_score, my_score)
        return next_state, reward, done
        
class BlackjackGame:
    def __init__(self):
        self.cards = [1,2,3,4,5,6,7,8,9,10,10,10,10]

    def draw(self):
        return random.choice(self.cards)

    def reset(self):
        player = [self.draw(), self.draw()]
        dealer = [self.draw()]
        return (sum(player), dealer[0])

    def step(self, state, action):
        player_sum, dealer_up = state
        if action == 1:  
            player_sum += self.draw()
            if player_sum > 21:
                return None, -1, True
            return (player_sum, dealer_up), 0, False
        dealer_total = dealer_up
        while dealer_total < 17:
            dealer_total += self.draw()
        if dealer_total > 21 or player_sum > dealer_total:
            return None, 1, True
        elif player_sum < dealer_total:
            return None, -1, True
        return None, 0, True
        
# Tabular Q-Learning Agent
class QLearningAgent:
    def __init__(self, actions, alpha=0.1, gamma=0.99, epsilon=0.1):
        self.q = {}
        self.actions = actions
        self.alpha = alpha
        self.gamma = gamma
        self.epsilon = epsilon

    def get_q(self, state, action):
        return self.q.get((state, action), 0.0)

    def select_action(self, state):
        if random.random() < self.epsilon:
            return random.choice(self.actions)
        q_vals = [self.get_q(state, a) for a in self.actions]
        return self.actions[int(np.argmax(q_vals))]

    def update(self, state, action, reward, next_state, done):
        max_next = 0 if done else max(self.get_q(next_state, a) for a in self.actions)
        old = self.get_q(state, action)
        self.q[(state, action)] = old + self.alpha * (reward + self.gamma * max_next - old)


if 0 in compsToRun:
    wins_random = 0
    draws_random = 0
    losses_random = 0
    for _ in range(100):
        game = prb.Game(prb.DieRollGame(), alg.RandomAgent(), alg.RandomAgent(), False)
        winner = game.playGame()
        if winner == 0:
            wins_random += 1
        elif winner == 1:
            losses_random += 1
        else:
            draws_random += 1
    print(f"Random vs. Random - Wins: {wins_random}, Losses: {losses_random}, Draws: {draws_random}")

if 1 in compsToRun:
    agent = alg.RuleBasedAgent()
    wins_rule = 0
    draws_rule = 0
    losses_rule = 0
    for _ in range(100):
        game = prb.Game(prb.DieRollGame(), agent, alg.RandomAgent(), False)
        winner = game.playGame()
        if winner == 0:
            wins_rule += 1
        elif winner == 1:
            losses_rule += 1
        else:
            draws_rule += 1
        agent.update_results(winner)
    print(f"Rule vs. Random - Wins: {wins_rule}, Losses: {losses_rule}, Draws: {draws_rule}")
    print(f"Rule based agent total wins: {agent.wins}, losses: {agent.losses}, draws: {agent.draws}")
