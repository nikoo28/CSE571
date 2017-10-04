# valueIterationAgents.py
# -----------------------
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


import mdp, util

from learningAgents import ValueEstimationAgent


class ValueIterationAgent(ValueEstimationAgent):
    """
        * Please read learningAgents.py before reading this.*

        A ValueIterationAgent takes a Markov decision process
        (see mdp.py) on initialization and runs value iteration
        for a given number of iterations using the supplied
        discount factor.
    """

    def __init__(self, mdp, discount=0.9, iterations=100):
        """
          Your value iteration agent should take an mdp on
          construction, run the indicated number of iterations
          and then act according to the resulting policy.

          Some useful mdp methods you will use:
              mdp.getStates()
              mdp.getPossibleActions(state)
              mdp.getTransitionStatesAndProbs(state, action)
              mdp.getReward(state, action, nextState)
              mdp.isTerminal(state)
        """
        self.mdp = mdp
        self.discount = discount
        self.iterations = iterations
        self.values = util.Counter()  # A Counter is a dict with default 0

        # Write value iteration code here
        "*** YOUR CODE HERE ***"

        # Get all states
        states = self.mdp.getStates()

        for i in range(self.iterations):

            # NOT modify the original values, but create a copy
            newQValues = self.values.copy()

            # Iterate over each of the possible state in each iteration
            for state in states:

                if not self.mdp.isTerminal(state):
                    actionsAtState = self.mdp.getPossibleActions(state)
                    maxValue = float('-inf')
                    for action in actionsAtState:
                        qValue = self.getQValue(state, action)
                        if qValue > maxValue:
                            maxValue = qValue
                    newQValues[state] = maxValue

            self.values = newQValues

    def getValue(self, state):
        """
          Return the value of the state (computed in __init__).
        """
        return self.values[state]

    def computeQValueFromValues(self, state, action):
        """
          Compute the Q-value of action in state from the
          value function stored in self.values.
        """
        "*** YOUR CODE HERE ***"

        qValue = 0

        for transition in self.mdp.getTransitionStatesAndProbs(state, action):
            # Get the nextState and probability
            nextState = transition[0]
            probability = transition[1]

            rewardAtState = self.mdp.getReward(state, action, nextState)
            valueAtState = self.getValue(nextState)
            discountAtState = self.discount

            qValue += probability * (rewardAtState + discountAtState * valueAtState)

        return qValue

    def computeActionFromValues(self, state):
        """
          The policy is the best action in the given state
          according to the values currently stored in self.values.

          You may break ties any way you see fit.  Note that if
          there are no legal actions, which is the case at the
          terminal state, you should return None.
        """

        "*** YOUR CODE HERE ***"
        # We need to get the best possible action from the states
        legalActions = self.mdp.getPossibleActions(state)
        if len(legalActions) == 0:
            return None

        qValue = float('-inf')
        bestAction = None

        for legalAction in legalActions:

            # Get the Q-value for each action
            legalActionQValue = self.computeQValueFromValues(state, legalAction)

            # If this action is better than our chosen one, update it
            if legalActionQValue > qValue:
                qValue = legalActionQValue
                bestAction = legalAction

        # Return the best action
        return bestAction

    def getPolicy(self, state):
        return self.computeActionFromValues(state)

    def getAction(self, state):
        "Returns the policy at the state (no exploration)."
        return self.computeActionFromValues(state)

    def getQValue(self, state, action):
        return self.computeQValueFromValues(state, action)
