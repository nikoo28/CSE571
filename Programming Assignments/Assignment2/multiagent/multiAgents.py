# multiAgents.py
# --------------
# Licensing Information:  You are free to use or extend these projects for
# educational purposes provided that (1) you do not distribute or publish
# solutions, (2) you retain this notice, and (3) you provide clear
# attribution to UC Berkeley, including a link to http://ai.berkeley.edu.
#
# Attribution Information: The Pacman AI projects were developed at UC Berkeley.
# The core projects and autograders were primarily created by John DeNero
# (denero@cs.berkeley.edu) and Dan Klein (klein@cs.berkeley.edu).
# Student side autograding was added by Brad Miller, Nick Hay, and
# Pieter Abbeel (pabbeel@cs.berkeley.edu).


from util import manhattanDistance
from game import Directions
import random, util

from game import Agent


class ReflexAgent(Agent):
    """
      A reflex agent chooses an action at each choice point by examining
      its alternatives via a state evaluation function.

      The code below is provided as a guide.  You are welcome to change
      it in any way you see fit, so long as you don't touch our method
      headers.
    """

    def getAction(self, gameState):
        """
        You do not need to change this method, but you're welcome to.

        getAction chooses among the best options according to the evaluation function.

        Just like in the previous project, getAction takes a GameState and returns
        some Directions.X for some X in the set {North, South, West, East, Stop}
        """
        # Collect legal moves and successor states
        legalMoves = gameState.getLegalActions()

        # Choose one of the best actions
        scores = [self.evaluationFunction(gameState, action) for action in legalMoves]
        bestScore = max(scores)
        bestIndices = [index for index in range(len(scores)) if scores[index] == bestScore]
        chosenIndex = random.choice(bestIndices)  # Pick randomly among the best

        "Add more of your code here if you want to"

        return legalMoves[chosenIndex]

    def evaluationFunction(self, currentGameState, action):
        """
        Design a better evaluation function here.

        The evaluation function takes in the current and proposed successor
        GameStates (pacman.py) and returns a number, where higher numbers are better.

        The code below extracts some useful information from the state, like the
        remaining food (newFood) and Pacman position after moving (newPos).
        newScaredTimes holds the number of moves that each ghost will remain
        scared because of Pacman having eaten a power pellet.

        Print out these variables to see what you're getting, then combine them
        to create a masterful evaluation function.
        """
        # Useful information you can extract from a GameState (pacman.py)
        successorGameState = currentGameState.generatePacmanSuccessor(action)
        newPos = successorGameState.getPacmanPosition()
        newFood = successorGameState.getFood()
        newGhostStates = successorGameState.getGhostStates()
        newScaredTimes = [ghostState.scaredTimer for ghostState in newGhostStates]

        "*** YOUR CODE HERE ***"

        # If game state is win return a very high score
        if successorGameState.isWin():
            return float('inf') - 10

        score = successorGameState.getScore()
        # if food is eaten increase score by 10
        if currentGameState.getNumFood() - successorGameState.getNumFood() == 1:
            score = score + 100

        # find distance between ghost pos and current state and  update score
        maxDist = 5
        ghostList = currentGameState.getNumAgents() - 1
        for ghost in range(1, ghostList):
            ghostPos = currentGameState.getGhostPosition(ghost)
            dist = manhattanDistance(newPos, ghostPos)
            score = score + max(dist, maxDist)

        # get food positions and move close to food positions
        newFood = newFood.asList()
        minDist = 10
        for foodPos in newFood:
            dist = manhattanDistance(newPos, foodPos)
            minDist = min(minDist, dist)

        # Score increases with decreasing distance from food pellet
        score = score - (minDist * 5)

        # if Power pellet is eaten increase score
        if len(currentGameState.getCapsules()) - len(successorGameState.getCapsules()) == 1:
            score = score + 200

        # if dead end reached
        if action == Directions.STOP:
            score = score - 10

        return score


def scoreEvaluationFunction(currentGameState):
    """
      This default evaluation function just returns the score of the state.
      The score is the same one displayed in the Pacman GUI.

      This evaluation function is meant for use with adversarial search agents
      (not reflex agents).
    """
    return currentGameState.getScore()


class MultiAgentSearchAgent(Agent):
    """
      This class provides some common elements to all of your
      multi-agent searchers.  Any methods defined here will be available
      to the MinimaxPacmanAgent, AlphaBetaPacmanAgent & ExpectimaxPacmanAgent.

      You *do not* need to make any changes here, but you can if you want to
      add functionality to all your adversarial search agents.  Please do not
      remove anything, however.

      Note: this is an abstract class: one that should not be instantiated.  It's
      only partially specified, and designed to be extended.  Agent (game.py)
      is another abstract class.
    """

    def __init__(self, evalFn='scoreEvaluationFunction', depth='2'):
        self.index = 0  # Pacman is always agent index 0
        self.evaluationFunction = util.lookup(evalFn, globals())
        self.depth = int(depth)


class MinimaxAgent(MultiAgentSearchAgent):
    """
      Your minimax agent (question 2)
    """

    def getAction(self, gameState):
        """
          Returns the minimax action from the current gameState using self.depth
          and self.evaluationFunction.

          Here are some method calls that might be useful when implementing minimax.

          gameState.getLegalActions(agentIndex):
            Returns a list of legal actions for an agent
            agentIndex=0 means Pacman, ghosts are >= 1

          gameState.generateSuccessor(agentIndex, action):
            Returns the successor game state after an agent takes an action

          gameState.getNumAgents():
            Returns the total number of agents in the game
        """
        "*** YOUR CODE HERE ***"

        def maxScore(gameState, depth):

            # the function returns the nest action at root node else the max score.
            if gameState.isWin() or gameState.isLose():
                return self.evaluationFunction(gameState)

            if depth == 0:
                ultimateStateFlag = True
            else:
                ultimateStateFlag = False

            legalActions = gameState.getLegalActions(0)
            maxscore = float("-inf")
            bestAction = ""
            for action in legalActions:
                score = minScore(gameState.generateSuccessor(0, action), depth, 1)
                if score > maxscore:
                    maxscore = score
                    if ultimateStateFlag:
                        bestAction = action

            if ultimateStateFlag:
                return bestAction
            return maxscore

        def minScore(gameState, depth, ghostNumber):

            # the function returns minValue or evaluation function value
            # if the current depth of game reaches final Depth(self.depth)
            if gameState.isWin() or gameState.isLose():
                return self.evaluationFunction(gameState)

            legalActions = gameState.getLegalActions(ghostNumber)
            numberOfAgents = gameState.getNumAgents()
            minscore = float("inf")

            for action in legalActions:
                # if this is the last ghost, then the next level will be for pacman of depth=depth+1
                if numberOfAgents - 1 == ghostNumber:
                    if depth == self.depth - 1:
                        # this is the last depth so return the value associated with evaluation function
                        minscore = min(minscore,
                                       self.evaluationFunction(gameState.generateSuccessor(ghostNumber, action)))
                    else:
                        minscore = min(minscore, maxScore(gameState.generateSuccessor(ghostNumber, action), depth + 1))
                else:
                    minscore = min(minscore,
                                   minScore(gameState.generateSuccessor(ghostNumber, action), depth, ghostNumber + 1))
            return minscore

        return maxScore(gameState, 0)


class AlphaBetaAgent(MultiAgentSearchAgent):
    """
      Your minimax agent with alpha-beta pruning (question 3)
    """

    def getAction(self, gameState):
        """
          Returns the minimax action using self.depth and self.evaluationFunction
        """
        "*** YOUR CODE HERE ***"

        def maxScore(gameState, depth, alpha, beta):

            # the function returns the nest action at root node else the max score.
            if gameState.isWin() or gameState.isLose():
                return self.evaluationFunction(gameState)

            if depth == 0:
                ultimateStateFlag = True
            else:
                ultimateStateFlag = False

            legalActions = gameState.getLegalActions(0)
            maxscore = float("-inf")
            bestAction = ""

            for action in legalActions:
                score = minScore(gameState.generateSuccessor(0, action), depth, 1, alpha, beta)
                if score > maxscore:
                    maxscore = score
                    if ultimateStateFlag:
                        bestAction = action

                if maxscore > beta:
                    return maxscore

                alpha = max(alpha, maxscore)

            if ultimateStateFlag:
                return bestAction
            return maxscore

        def minScore(gameState, depth, ghostNumber, alpha, beta):

            # the function returns minValue or evaluation function value
            # if the current depth of game reaches final Depth(self.depth)
            if gameState.isWin() or gameState.isLose():
                return self.evaluationFunction(gameState)

            legalActions = gameState.getLegalActions(ghostNumber)
            numberOfAgents = gameState.getNumAgents()
            minscore = float("inf")

            for action in legalActions:
                # if this is the last ghost, then the next level will be for pacman of depth=depth+1
                if numberOfAgents - 1 == ghostNumber:
                    # this is the last depth so return the value associated with evaluation function
                    if depth == self.depth - 1:
                        minscore = min(minscore,
                                       self.evaluationFunction(gameState.generateSuccessor(ghostNumber, action)))
                    else:
                        minscore = min(minscore,
                                       maxScore(gameState.generateSuccessor(ghostNumber, action), depth + 1, alpha,
                                                beta))
                else:
                    minscore = min(minscore,
                                   minScore(gameState.generateSuccessor(ghostNumber, action), depth, ghostNumber + 1,
                                            alpha, beta))
                # if minScore is lesser
                if minscore < alpha:
                    return minscore
                beta = min(beta, minscore)
            return minscore

        return maxScore(gameState, 0, float("-inf"), float("inf"))


class ExpectimaxAgent(MultiAgentSearchAgent):
    """
      Your expectimax agent (question 4)
    """

    def getAction(self, gameState):
        """
          Returns the expectimax action using self.depth and self.evaluationFunction

          All ghosts should be modeled as choosing uniformly at random from their
          legal moves.
        """
        "*** YOUR CODE HERE ***"

        def maxScore(gameState, depth, ghosts):

            if gameState.isWin() or gameState.isLose() or (depth == 0):
                return self.evaluationFunction(gameState)

            totalLegalActions = gameState.getLegalActions(0)
            maxscore = float("-inf")

            for action in totalLegalActions:
                maxscore = max(maxscore, expectedScore(gameState.generateSuccessor(0, action), depth, 1, ghosts))
            return maxscore

        def expectedScore(gameState, depth, agentNumber, ghosts):

            if gameState.isWin() or gameState.isLose() or depth == 0:
                return self.evaluationFunction(gameState)

            score = 0
            totalLegalActions = gameState.getLegalActions(agentNumber)

            for action in totalLegalActions:
                if agentNumber == ghosts:
                    score += maxScore(gameState.generateSuccessor(agentNumber, action),
                                      depth - 1, ghosts)
                else:
                    score += expectedScore(gameState.generateSuccessor(agentNumber, action),
                                           depth, agentNumber + 1, ghosts)
            return score / len(totalLegalActions)

        # main function
        if gameState.isWin() or gameState.isLose():
            return self.evaluationFunction(gameState)

        ghosts = gameState.getNumAgents() - 1
        totalLegalActions = gameState.getLegalActions(0)
        priorityQueue = util.PriorityQueue()

        bestAction = ""

        for action in totalLegalActions:
            successor = gameState.generateSuccessor(0, action)
            score = expectedScore(successor, self.depth, 1, ghosts)
            priorityQueue.push(action, score)

        while not priorityQueue.isEmpty():
            bestAction = priorityQueue.pop()

        return bestAction


def betterEvaluationFunction(currentGameState):
    """
      Your extreme ghost-hunting, pellet-nabbing, food-gobbling, unstoppable
      evaluation function (question 5).

      DESCRIPTION: <The logic has been written in comments>
    """
    "*** YOUR CODE HERE ***"

    # Return a high score if we win the game
    if currentGameState.isWin():
        return float('inf') - 10

    if currentGameState.isLose():
        return float('-inf') + 10

    score = currentGameState.getScore()

    currentPos = currentGameState.getPacmanPosition()

    # find distance between ghost pos and current state and  update score
    maxDist = 5
    ghostList = currentGameState.getNumAgents() - 1
    for ghost in range(1, ghostList):
        ghostPos = currentGameState.getGhostPosition(ghost)
        dist = util.manhattanDistance(currentPos, ghostPos)
        score = score + max(dist, maxDist)

    # get food positions and move close to food positions
    newFood = currentGameState.getFood().asList()
    minDist = 100
    for foodPos in newFood:
        dist = util.manhattanDistance(currentPos, foodPos)
        minDist = min(minDist, dist)

    # Score increases with decreasing distance from food pellet
    # Move to the closest food, hence minDist
    score = score - (minDist * 2)

    # if power capsule eaten, increase score
    # We must boost the score if pacman eats a power capsule
    foodCapsules = currentGameState.getCapsules()
    if currentGameState.getPacmanPosition() in foodCapsules:
        score = score + 400
    score = score - (10 * len(foodCapsules))

    return score

# Abbreviation
better = betterEvaluationFunction
