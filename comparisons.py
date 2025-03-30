import problems as prb
import algorithms as alg

compsToRun = [0, 1]

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