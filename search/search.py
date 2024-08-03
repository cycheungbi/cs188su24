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
    return  [s, s, w, s, w, w, s, w]

def depthFirstSearch(problem: SearchProblem):
    """Search the deepest nodes in the search tree first."""
    # Initialize the fringe (stack), visited set, and start state
    fringe = util.Stack()
    visited = set()
    start_state = problem.getStartState()
    
    # Push the initial state with an empty path
    fringe.push((start_state, []))
    
    while not fringe.isEmpty():
        current_state, path = fringe.pop()
        
        if problem.isGoalState(current_state):
            return path
        
        if current_state not in visited:
            visited.add(current_state)
            
            for successor, action, _ in problem.getSuccessors(current_state):
                if successor not in visited:
                    new_path = path + [action]
                    fringe.push((successor, new_path))
    
    # If no solution is found
    return []

def breadthFirstSearch(problem: SearchProblem):
    """Search the shallowest nodes in the search tree first."""
    # Initialize the fringe (queue), visited set, and start state
    fringe = util.Queue()
    visited = set()
    start_state = problem.getStartState()
    
    # Enqueue the initial state with an empty path
    fringe.push((start_state, []))
    
    while not fringe.isEmpty():
        current_state, path = fringe.pop()
        
        if problem.isGoalState(current_state):
            return path
        
        if current_state not in visited:
            visited.add(current_state)
            
            for successor, action, _ in problem.getSuccessors(current_state):
                if successor not in visited:
                    new_path = path + [action]
                    fringe.push((successor, new_path))
    
    # If no solution is found
    return []

def uniformCostSearch(problem: SearchProblem):
    """Search the node of least total cost first."""
    # Initialize the fringe (priority queue), visited set, and start state
    fringe = util.PriorityQueue()
    visited = set()
    start_state = problem.getStartState()
    
    # Push the initial state with an empty path and zero cost
    fringe.push((start_state, [], 0), 0)
    
    while not fringe.isEmpty():
        current_state, path, cost = fringe.pop()
        
        if problem.isGoalState(current_state):
            return path
        
        if current_state not in visited:
            visited.add(current_state)
            
            for successor, action, step_cost in problem.getSuccessors(current_state):
                if successor not in visited:
                    new_path = path + [action]
                    new_cost = cost + step_cost
                    fringe.push((successor, new_path, new_cost), new_cost)
    
    # If no solution is found
    return []

def nullHeuristic(state, problem=None):
    """
    A heuristic function estimates the cost from the current state to the nearest
    goal in the provided SearchProblem.  This heuristic is trivial.
    """
    return 0

def aStarSearch(problem: SearchProblem, heuristic=nullHeuristic):
    """Search the node that has the lowest combined cost and heuristic first."""
    # Initialize the fringe (priority queue), visited set, and start state
    fringe = util.PriorityQueue()
    visited = set()
    start_state = problem.getStartState()
    
    # Push the initial state with an empty path and zero cost
    fringe.push((start_state, [], 0), heuristic(start_state, problem))
    
    while not fringe.isEmpty():
        current_state, path, cost = fringe.pop()
        
        if problem.isGoalState(current_state):
            return path
        
        if current_state not in visited:
            visited.add(current_state)
            
            for successor, action, step_cost in problem.getSuccessors(current_state):
                if successor not in visited:
                    new_path = path + [action]
def aStarSearch(problem: SearchProblem, heuristic=nullHeuristic):
    """Search the node that has the lowest combined cost and heuristic first."""
    # Initialize the fringe (priority queue), visited set, and start state
    fringe = util.PriorityQueue()
    visited = set()
    start_state = problem.getStartState()
    
    # Push the initial state with an empty path and zero cost
    fringe.push((start_state, [], 0), heuristic(start_state, problem))
    
    while not fringe.isEmpty():
        current_state, path, cost = fringe.pop()
        
        if problem.isGoalState(current_state):
            return path
        
        if current_state not in visited:
            visited.add(current_state)
            
            for successor, action, step_cost in problem.getSuccessors(current_state):
                if successor not in visited:
                    new_path = path + [action]
                    new_cost = cost + step_cost
                    priority = new_cost + heuristic(successor, problem)
                    fringe.push((successor, new_path, new_cost), priority)
    
    # If no solution is found
    return []
    # If no solution is found
    return []

# Abbreviations
bfs = breadthFirstSearch
dfs = depthFirstSearch
astar = aStarSearch
ucs = uniformCostSearch