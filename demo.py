
import problems as prb
import algorithms as alg

print("demo.py is running!")
demoGame = prb.Game(prb.DieRollGame(), alg.RuleBasedAgent(), alg.RandomAgent())
demoGame.playGame()