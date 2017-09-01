# search.py
# ---------
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


"""
In search.py, you will implement generic search algorithms which are called by
Pacman agents (in searchAgents.py).
"""

import util


class SearchProblem:
    """
    This class outlines the structure of a search problem, but doesn't implement
    any of the methods (in object-oriented terminology: an abstract class).

    You do not need to change anything in this class, ever.
    """

    def getStartState(self):
        """
        Returns the start state for the search problem.
        """
        util.raiseNotDefined()

    def isGoalState(self, state):
        """
          state: Search state

        Returns True if and only if the state is a valid goal state.
        """
        util.raiseNotDefined()

    def getSuccessors(self, state):
        """
          state: Search state

        For a given state, this should return a list of triples, (successor,
        action, stepCost), where 'successor' is a successor to the current
        state, 'action' is the action required to get there, and 'stepCost' is
        the incremental cost of expanding to that successor.
        """
        util.raiseNotDefined()

    def getCostOfActions(self, actions):
        """
         actions: A list of actions to take

        This method returns the total cost of a particular sequence of actions.
        The sequence must be composed of legal moves.
        """
        util.raiseNotDefined()


def tinyMazeSearch(problem):
    """
    Returns a sequence of moves that solves tinyMaze.  For any other maze, the
    sequence of moves will be incorrect, so only use this for tinyMaze.
    """
    from game import Directions
    s = Directions.SOUTH
    w = Directions.WEST
    return [s, s, w, s, w, w, s, w]


def depthFirstSearch(problem):
    """
    Search the deepest nodes in the search tree first.

    Your search algorithm needs to return a list of actions that reaches the
    goal. Make sure to implement a graph search algorithm.

    To get started, you might want to try some of these simple commands to
    understand the search problem that is being passed in:

    print "Start:", problem.getStartState()
    print "Is the start a goal?", problem.isGoalState(problem.getStartState())
    print "Start's successors:", problem.getSuccessors(problem.getStartState())
    """
    "*** YOUR CODE HERE ***"

    # Taking fringe as a LIFO stack
    fringe = util.Stack()
    startState = (problem.getStartState(), {})
    fringe.push(startState)

    # Maintain a set of all the visited nodes
    visitedNodes = set([])

    # Start a loop to find a possible path
    while True:

        # If we are out of fringes, no path exists
        if fringe.isEmpty():
            return []

        # Removing that vertex from queue, whose neighbour will be
        # visited now
        nextNode = fringe.pop()

        # Check if the goal state is reached in the next node
        if problem.isGoalState(nextNode[0]):
            return nextNode[1]

        # If the node has not been visited, add all its successors to the
        # fringe stack
        if not nextNode[0] in visitedNodes:
            visitedNodes.add(nextNode[0])
            successors = problem.getSuccessors(nextNode[0])

            for idx in range(len(successors)):
                currentPath = list(nextNode[1])
                directionOfNextNode = (successors[idx])[1]
                currentPath.append(directionOfNextNode)
                nextNodeState = (successors[idx])[0]
                fringe.push((nextNodeState, currentPath))

    util.raiseNotDefined()


def breadthFirstSearch(problem):
    """Search the shallowest nodes in the search tree first."""
    "*** YOUR CODE HERE ***"

    # Taking fringe as a FIFO queue
    fringe = util.Queue()
    startState = (problem.getStartState(), {})
    fringe.push(startState)

    # Maintain a set of all the visited nodes
    visitedNodes = set([])
    visitedNodes.add(startState[0])

    # Start a loop to find a possible path
    while True:

        # If we are out of fringes, no path exists
        if fringe.isEmpty():
            return []

        # Removing that vertex from queue, whose neighbour will be
        # visited now
        nextNode = fringe.pop()

        # Check if the goal state is reached in the next node
        if problem.isGoalState(nextNode[0]):
            return nextNode[1]

        # Processing all neighbors of vertex
        neighbors = problem.getSuccessors(nextNode[0])
        for idx in range(len(neighbors)):
            if not (neighbors[idx])[0] in visitedNodes:
                visitedNodes.add((neighbors[idx])[0])
                currentPath = list(nextNode[1])
                directionOfNextNode = (neighbors[idx])[1]
                currentPath.append(directionOfNextNode)
                nextNodeState = (neighbors[idx])[0]
                fringe.push((nextNodeState, currentPath))

    util.raiseNotDefined()


def uniformCostSearch(problem):
    """Search the node of least total cost first."""
    "*** YOUR CODE HERE ***"

    fringe = util.PriorityQueue()
    startState = (problem.getStartState(), {})
    fringe.push(startState, problem.getCostOfActions(startState[1]))

    # Maintain a set of all the visited nodes
    visitedNodes = set([])
    visitedNodes.add(startState[0])

    cost = 0
    # Start a loop to find a possible path
    while True:

        # If we are out of fringes, no path exists
        if fringe.isEmpty():
            return []

        # Removing that vertex from queue, whose neighbour will be
        # visited now
        nextNode = fringe.pop()

        # Check if the goal state is reached in the next node
        if problem.isGoalState(nextNode[0]):
            return nextNode[1]

        # Processing all neighbors of vertex
        neighbors = problem.getSuccessors(nextNode[0])
        cost += 1
        for idx in range(len(neighbors)):
            if not (neighbors[idx])[0] in visitedNodes:
                visitedNodes.add((neighbors[idx])[0])
                currentPath = list(nextNode[1])
                directionOfNextNode = (neighbors[idx])[1]
                currentPath.append(directionOfNextNode)
                nextNodeState = (neighbors[idx])[0]
                fringe.push((nextNodeState, currentPath), problem.getCostOfActions(currentPath))


    util.raiseNotDefined()


def nullHeuristic(state, problem=None):
    """
    A heuristic function estimates the cost from the current state to the nearest
    goal in the provided SearchProblem.  This heuristic is trivial.
    """
    return 0


def aStarSearch(problem, heuristic=nullHeuristic):
    """Search the node that has the lowest combined cost and heuristic first."""
    "*** YOUR CODE HERE ***"
    util.raiseNotDefined()


# Abbreviations
bfs = breadthFirstSearch
dfs = depthFirstSearch
astar = aStarSearch
ucs = uniformCostSearch
