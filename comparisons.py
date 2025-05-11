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

class BlackjackFeatureExtractor:
    def __init__(self):
        self.num_features = 3

    def extract(self, state, action):
        player_sum, dealer_up = state
        return np.array([
            player_sum / 21.0,
            dealer_up / 10.0,
            1.0 if player_sum > 17 else 0.0
        ])
# Approximate Q-Learning Agent
class ApproxQLearningAgent(QLearningAgent):
    def __init__(self, actions, feature_extractor, alpha=0.01, gamma=0.99, epsilon=0.1):
        super().__init__(actions, alpha, gamma, epsilon)
        self.feat = feature_extractor
        self.weights = np.zeros(self.feat.num_features)

    def get_q(self, state, action):
        features = self.feat.extract(state, action)
        return np.dot(self.weights, features)

    def update(self, state, action, reward, next_state, done):
        pred = self.get_q(state, action)
        max_next = 0 if done else max(self.get_q(next_state, a) for a in self.actions)
        target = reward + self.gamma * max_next
        error = target - pred
        features = self.feat.extract(state, action)
        self.weights += self.alpha * error * features

# Minimax Agent 
class MinimaxAgent:
    def __init__(self, eval_fn, max_depth=2):
        self.eval_fn = eval_fn
        self.max_depth = max_depth

    def select_action(self, game, state):
        _, action = self._minimax(game, state, 0, -np.inf, np.inf, True)
        return action

    def _minimax(self, game, state, depth, alpha, beta, maximizing):
        if depth == self.max_depth or getattr(game, 'is_terminal', lambda s: False)(state):
            return self.eval_fn(state), None

        best_action = None
        if maximizing:
            value = -np.inf
            for a in [0,1] if isinstance(game, BlackjackGame) else [0]:
                next_s, r, done = game.step(state, a)
                v, _ = (r, None) if done else self._minimax(game, next_s, depth+1, alpha, beta, False)
                if v > value:
                    value, best_action = v, a
                alpha = max(alpha, value)
                if alpha >= beta:
                    break
            return value, best_action
        else:
            value = np.inf
            for a in [0,1] if isinstance(game, BlackjackGame) else [0]:
                next_s, r, done = game.step(state, a)
                v, _ = (r, None) if done else self._minimax(game, next_s, depth+1, alpha, beta, True)
                if v < value:
                    value, best_action = v, a
                beta = min(beta, value)
                if beta <= alpha:
                    break
            return value, best_action

def blackjack_eval(state):
    player_sum, dealer_up = state
    return player_sum - dealer_up


HYPOTHESES = [
    "In small state spaces (die-roll), tabular Q-learning converges faster than minimax with hand-tuned eval.",
    "In moderate state spaces (Blackjack), approximate Q-learning with a better feature set outperforms tabular Q-learning given fixed training.",
    "Minimax evaluation functions approach tabular Q-learning performance only after multiple revision iterations."
]


def train_tabular_q(game, episodes=5000, track_metrics=False):
    actions = [0,1] if isinstance(game, BlackjackGame) else [0]
    agent = QLearningAgent(actions)
    
    total_steps = 0
    total_time = 0
    results = []

    for _ in range(episodes):
        steps = 0
        start = time.time()
        
        state = game.reset()
        done = False
        while not done:
            a = agent.select_action(state)
            next_s, r, done = game.step(state, a)
            agent.update(state, a, r, next_s, done)
            state = next_s
            steps += 1

        total_steps += steps
        total_time += time.time() - start
        results.append(1 if r > 0 else 0)

    if track_metrics:
        print(f"Avg steps per episode: {total_steps / episodes:.2f}")
        print(f"Avg time per episode: {total_time / episodes:.4f} sec")
        return agent, results
    else:
        return agent

def train_approx_q(game, feature_extractor, episodes=5000):
    actions = [0,1]
    agent = ApproxQLearningAgent(actions, feature_extractor)
    for _ in range(episodes):
        state = game.reset()
        done = False
        while not done:
            a = agent.select_action(state)
            next_s, r, done = game.step(state, a)
            agent.update(state, a, r, next_s, done)
            state = next_s
    return agent


def evaluate_agent(game, agent, trials=1000):
    wins = 0
    for _ in range(trials):
        state = game.reset()
        done = False
        while not done:
            if isinstance(agent, MinimaxAgent):
                a = agent.select_action(game, state)
            else:
                a = agent.select_action(state)
            next_s, r, done = game.step(state, a)
            state = next_s
        if r == 1:
            wins += 1
    return wins / trials

# Comparisons
if __name__ == "__main__":
    games = {"die": DieRollGame(), "bj": BlackjackGame()}

    for name, game in games.items():
        print(f"{name.upper()} GAME: ")

        # Tabular Q-learning with metrics
        tq, results = train_tabular_q(game, track_metrics=True)
        win_tq = evaluate_agent(game, tq)
        print(f"Tabular Q-Learning win rate: {win_tq:.2f}")
        
        

        if name == "bj":
            # Approximate Q-learning
            feat = BlackjackFeatureExtractor()
            aq = train_approx_q(game, feat)
            win_aq = evaluate_agent(game, aq)
            print(f"Approx Q-Learning win rate: {win_aq:.2f}")

            # Minimax
            mm = MinimaxAgent(eval_fn=blackjack_eval, max_depth=2)
            win_mm = evaluate_agent(game, mm)
            print(f"MinimaxAgent win rate: {win_mm:.2f}")

    print("\nHypotheses:")
    for h in HYPOTHESES:
        print(f"- {h}")

