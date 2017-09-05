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
    start_state = problem.getStartState()
    start_path = []
    start_cost = 0

    # Taking fringe as a LIFO stack
    fringe = util.Stack()
    start_state_tuple = (start_state, start_path, start_cost)
    fringe.push(start_state_tuple)

    # Maintain a set of all the visited nodes
    visited_states = set()

    # Start a loop to find a possible path
    while True:

        # If we are out of fringes, no path exists
        if fringe.isEmpty():
            return []

        # Removing that vertex from queue, whose neighbour will be
        # visited now
        next_state_tuple = fringe.pop()
        next_state = next_state_tuple[0]
        next_path = next_state_tuple[1]
        next_cost = next_state_tuple[2]

        # Check if the goal state is reached in the next node
        if problem.isGoalState(next_state):
            return next_path

        # If the node has not been visited, add all its successors to the
        # fringe stack
        if not next_state in visited_states:
            visited_states.add(next_state)

            # Get all possible successor states
            successor_state_tuples = problem.getSuccessors(next_state)

            # Insert the states in the fringes
            for idx in range(len(successor_state_tuples)):
                successor_state_tuple = successor_state_tuples[idx]
                successor_state = successor_state_tuple[0]
                successor_path = successor_state_tuple[1]
                successor_cost = successor_state_tuple[2]

                if not successor_state in visited_states:
                    new_path = next_path + [successor_path]
                    new_tuple = (successor_state, new_path, successor_cost)
                    fringe.push(new_tuple)


def breadthFirstSearch(problem):
    """Search the shallowest nodes in the search tree first."""

    start_state = problem.getStartState()
    start_path = []
    start_cost = 0

    # Taking fringe as a FIFO queue
    fringe = util.Queue()
    start_state_tuple = (start_state, start_path, start_cost)
    fringe.push(start_state_tuple)

    # Maintain a set of all the visited nodes
    visited_states = set()

    # Start a loop to find a possible path
    while True:

        # If we are out of fringes, no path exists
        if fringe.isEmpty():
            return []

        # Removing that vertex from queue, whose neighbour will be
        # visited now
        next_state_tuple = fringe.pop()
        next_state = next_state_tuple[0]
        next_path = next_state_tuple[1]
        next_cost = next_state_tuple[2]

        # Check if the goal state is reached in the next node
        if problem.isGoalState(next_state):
            return next_path

        # If the node has not been visited, add all its successors to the
        # fringe stack
        if not next_state in visited_states:
            visited_states.add(next_state)

            # Get all possible successor states
            successor_state_tuples = problem.getSuccessors(next_state)

            # Insert the states in the fringes
            for idx in range(len(successor_state_tuples)):
                successor_state_tuple = successor_state_tuples[idx]
                successor_state = successor_state_tuple[0]
                successor_path = successor_state_tuple[1]
                successor_cost = successor_state_tuple[2]

                if not successor_state in visited_states:
                    new_path = next_path + [successor_path]
                    new_tuple = (successor_state, new_path, successor_cost)
                    fringe.push(new_tuple)


def uniformCostSearch(problem):
    """Search the node of least total cost first."""

    start_state = problem.getStartState()
    start_path = []
    start_cost = 0

    # Taking fringe as a priority queue
    fringe = util.PriorityQueue()
    start_state_tuple = (start_state, start_path, start_cost)
    fringe.push(start_state_tuple, problem.getCostOfActions(start_path))

    # Maintain a set of all the visited nodes
    visited_states = set()

    # Start a loop to find a possible path
    while True:

        # If we are out of fringes, no path exists
        if fringe.isEmpty():
            return []

        # Removing that vertex from queue, whose neighbour will be
        # visited now
        next_state_tuple = fringe.pop()
        next_state = next_state_tuple[0]
        next_path = next_state_tuple[1]
        next_cost = next_state_tuple[2]

        # Check if the goal state is reached in the next node
        if problem.isGoalState(next_state):
            return next_path

        # If the node has not been visited, add all its successors to the
        # fringe stack
        if not next_state in visited_states:
            visited_states.add(next_state)

            # Get all possible successor states
            successor_state_tuples = problem.getSuccessors(next_state)

            # Insert the states in the fringes
            for idx in range(len(successor_state_tuples)):
                successor_state_tuple = successor_state_tuples[idx]
                successor_state = successor_state_tuple[0]
                successor_path = successor_state_tuple[1]
                successor_cost = successor_state_tuple[2]

                if not successor_state in visited_states:
                    new_path = next_path + [successor_path]
                    new_cost = problem.getCostOfActions(new_path)
                    new_tuple = (successor_state, new_path, new_cost)
                    fringe.update(new_tuple, new_cost)


def nullHeuristic(state, problem=None):
    """
    A heuristic function estimates the cost from the current state to the nearest
    goal in the provided SearchProblem.  This heuristic is trivial.
    """
    return 0


def aStarSearch(problem, heuristic=nullHeuristic):
    """Search the node of least total cost first."""

    start_state = problem.getStartState()
    start_path = []
    start_cost = 0

    # Taking fringe as a priority queue
    fringe = util.PriorityQueue()
    start_state_tuple = (start_state, start_path, start_cost)
    fringe.push(start_state_tuple, problem.getCostOfActions(start_path))

    # Maintain a set of all the visited nodes
    visited_states = set()

    # Start a loop to find a possible path
    while True:

        # If we are out of fringes, no path exists
        if fringe.isEmpty():
            return []

        # Removing that vertex from queue, whose neighbour will be
        # visited now
        next_state_tuple = fringe.pop()
        next_state = next_state_tuple[0]
        next_path = next_state_tuple[1]
        next_cost = next_state_tuple[2]

        # Check if the goal state is reached in the next node
        if problem.isGoalState(next_state):
            return next_path

        # If the node has not been visited, add all its successors to the
        # fringe stack
        if not next_state in visited_states:
            visited_states.add(next_state)

            # Get all possible successor states
            successor_state_tuples = problem.getSuccessors(next_state)

            # Insert the states in the fringes
            for idx in range(len(successor_state_tuples)):
                successor_state_tuple = successor_state_tuples[idx]
                successor_state = successor_state_tuple[0]
                successor_path = successor_state_tuple[1]
                successor_cost = successor_state_tuple[2]

                if not successor_state in visited_states:
                    new_path = next_path + [successor_path]
                    new_cost = problem.getCostOfActions(new_path)
                    new_tuple = (successor_state, new_path, new_cost)
                    # Determine the heuristic cost
                    heuristic_cost = new_cost + heuristic(successor_state, problem)
                    fringe.update(new_tuple, heuristic_cost)


# Abbreviations
bfs = breadthFirstSearch
dfs = depthFirstSearch
astar = aStarSearch
ucs = uniformCostSearch
