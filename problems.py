import random

class DieRollGame:
    def __init__(self):
        self.player_rolls = [0, 0]
        self.rolls_made = 0
        self.current_player = 0

    def getLegalMoves(self):
        return []

    def doMove(self, player):
        if self.player_rolls[player] == 0:
            self.player_rolls[player] = random.randint(1, 6)
        self.rolls_made += 1
        self.current_player = 1

    def isTerminal(self):
        return self.rolls_made == 2

    def getWinner(self):
        if not self.isTerminal():
            return -1
        if self.player_rolls[0] > self.player_rolls[1]:
            return 0
        elif self.player_rolls[1] > self.player_rolls[0]:
            return 1
        else:
            return -1

    def showState(self, ms=1000):
        print(f"Player 1 rolled: {self.player_rolls[0]}")
        print(f"Player 2 rolled: {self.player_rolls[1]}")
        if self.isTerminal():
            winner = self.getWinner()
            if winner == 0:
                print("Player 1 wins!")
            elif winner == 1:
                print("Player 2 wins!")
            else:
                print("It's a draw!")

class Game:
    def __init__(self, problem, pZero, pOne, verbose=True):
        self.problem = problem
        self.players = [pZero, pOne]
        self.verbose = verbose
        if self.verbose:
            self.problem.showState()

    def playGame(self):
        pCur = 0
        while self.problem.isTerminal() == False:
            self.problem.doMove(pCur)
            if self.verbose:
                self.problem.showState()
            pCur = 1

        wIndex = self.problem.getWinner()
        winner = self.players[wIndex]
        if wIndex == -1:
            winner = "DRAW"
        print(f"The Winner is {winner} ({wIndex})!")
        if self.verbose:
            self.problem.showState(4000)
        return wIndex 