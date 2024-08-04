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
from pacman import GameState

class ReflexAgent(Agent):
    """
    A reflex agent chooses an action at each choice point by examining
    its alternatives via a state evaluation function.

    The code below is provided as a guide.  You are welcome to change
    it in any way you see fit, so long as you don't touch our method
    headers.
    """


    def getAction(self, gameState: GameState):
        """
        You do not need to change this method, but you're welcome to.

        getAction chooses among the best options according to the evaluation function.

        Just like in the previous project, getAction takes a GameState and returns
        some Directions.X for some X in the set {NORTH, SOUTH, WEST, EAST, STOP}
        """
        # Collect legal moves and successor states
        legalMoves = gameState.getLegalActions()

        # Choose one of the best actions
        scores = [self.evaluationFunction(gameState, action) for action in legalMoves]
        bestScore = max(scores)
        bestIndices = [index for index in range(len(scores)) if scores[index] == bestScore]
        chosenIndex = random.choice(bestIndices) # Pick randomly among the best

        "Add more of your code here if you want to"

        return legalMoves[chosenIndex]

    def evaluationFunction(self, currentGameState: GameState, action):
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

        # Initialize score with the successor state's score
        score = successorGameState.getScore()

        # Food evaluation
        foodList = newFood.asList()
        if foodList:
            # Encourage eating nearby food
            closestFoodDist = min(manhattanDistance(newPos, food) for food in foodList)
            score += 10.0 / (closestFoodDist + 1)  # Increased weight
            
            # Encourage clearing the board
            score += 100.0 / (len(foodList) + 1)  # New factor

        # Ghost evaluation
        for i, ghostState in enumerate(newGhostStates):
            ghostPos = ghostState.getPosition()
            ghostDist = manhattanDistance(newPos, ghostPos)
            
            if ghostDist > 0:
                if newScaredTimes[i] > ghostDist:
                    # Chase scared ghosts more aggressively
                    score += 200.0 / ghostDist
                else:
                    # Avoid non-scared ghosts more cautiously
                    if ghostDist < 3:
                        score -= 500.0 / ghostDist
                    else:
                        score -= 50.0 / ghostDist

        # Encourage exploration of new positions
        if action != Directions.STOP:
            score += 5

        # Prioritize power pellets
        capsules = currentGameState.getCapsules()
        if capsules:
            closestCapsuleDist = min(manhattanDistance(newPos, capsule) for capsule in capsules)
            score += 150.0 / (closestCapsuleDist + 1)  # High priority for power pellets

        # Avoid getting trapped
        legalActions = successorGameState.getLegalActions()
        if len(legalActions) == 1:  # Only one legal action might indicate a trap
            score -= 100

        return score

def scoreEvaluationFunction(currentGameState: GameState):
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

    def __init__(self, evalFn = 'scoreEvaluationFunction', depth = '2'):
        self.index = 0 # Pacman is always agent index 0
        self.evaluationFunction = util.lookup(evalFn, globals())
        self.depth = int(depth)

class MinimaxAgent(MultiAgentSearchAgent):
    """
    Your minimax agent (question 2)
    """

    def getAction(self, gameState: GameState):
        """
        Returns the minimax action from the current gameState using self.depth
        and self.evaluationFunction.
        """
        def minimax(state, agentIndex, depth):
            # If we've reached a terminal state or maximum depth, evaluate the state
            if state.isWin() or state.isLose() or depth == self.depth:
                return self.evaluationFunction(state)

            # If it's Pacman's turn (max player)
            if agentIndex == 0:
                value = float('-inf')
                for action in state.getLegalActions(agentIndex):
                    successor = state.generateSuccessor(agentIndex, action)
                    value = max(value, minimax(successor, 1, depth))
                return value
            # If it's a ghost's turn (min player)
            else:
                value = float('inf')
                nextAgent = (agentIndex + 1) % state.getNumAgents()
                # If all ghosts have moved, it's Pacman's turn and we increase the depth
                nextDepth = depth + 1 if nextAgent == 0 else depth
                for action in state.getLegalActions(agentIndex):
                    successor = state.generateSuccessor(agentIndex, action)
                    value = min(value, minimax(successor, nextAgent, nextDepth))
                return value

        # The actual getAction function starts here
        bestAction = None
        bestValue = float('-inf')
        for action in gameState.getLegalActions(0):  # 0 is Pacman's index
            successorState = gameState.generateSuccessor(0, action)
            value = minimax(successorState, 1, 0)  # Start with first ghost and depth 0
            if value > bestValue:
                bestValue = value
                bestAction = action

        return bestAction

class AlphaBetaAgent(MultiAgentSearchAgent):
    """
    Your minimax agent with alpha-beta pruning (question 3)
    """

    def getAction(self, gameState: GameState):
        """
        Returns the minimax action using self.depth and self.evaluationFunction
        """
        def alphaBeta(state, agentIndex, depth, alpha, beta):
            # If we've reached a terminal state or maximum depth, evaluate the state
            if state.isWin() or state.isLose() or depth == self.depth:
                return self.evaluationFunction(state)

            # If it's Pacman's turn (max player)
            if agentIndex == 0:
                value = float('-inf')
                for action in state.getLegalActions(agentIndex):
                    successor = state.generateSuccessor(agentIndex, action)
                    value = max(value, alphaBeta(successor, 1, depth, alpha, beta))
                    if value > beta:
                        return value
                    alpha = max(alpha, value)
                return value
            # If it's a ghost's turn (min player)
            else:
                value = float('inf')
                nextAgent = (agentIndex + 1) % state.getNumAgents()
                # If all ghosts have moved, it's Pacman's turn and we increase the depth
                nextDepth = depth + 1 if nextAgent == 0 else depth
                for action in state.getLegalActions(agentIndex):
                    successor = state.generateSuccessor(agentIndex, action)
                    value = min(value, alphaBeta(successor, nextAgent, nextDepth, alpha, beta))
                    if value < alpha:
                        return value
                    beta = min(beta, value)
                return value

        # The actual getAction function starts here
        bestAction = None
        alpha = float('-inf')
        beta = float('inf')
        bestValue = float('-inf')
        for action in gameState.getLegalActions(0):  # 0 is Pacman's index
            successorState = gameState.generateSuccessor(0, action)
            value = alphaBeta(successorState, 1, 0, alpha, beta)
            if value > bestValue:
                bestValue = value
                bestAction = action
            alpha = max(alpha, bestValue)

        return bestAction

class ExpectimaxAgent(MultiAgentSearchAgent):
    """
      Your expectimax agent (question 4)
    """

    def getAction(self, gameState: GameState):
        """
        Returns the expectimax action using self.depth and self.evaluationFunction

        All ghosts should be modeled as choosing uniformly at random from their
        legal moves.
        """
        def expectimax(state, agentIndex, depth):
            # If we've reached a terminal state or maximum depth, evaluate the state
            if state.isWin() or state.isLose() or depth == self.depth:
                return self.evaluationFunction(state)

            # If it's Pacman's turn (max player)
            if agentIndex == 0:
                value = float('-inf')
                for action in state.getLegalActions(agentIndex):
                    successor = state.generateSuccessor(agentIndex, action)
                    value = max(value, expectimax(successor, 1, depth))
                return value
            # If it's a ghost's turn (chance player)
            else:
                value = 0
                actions = state.getLegalActions(agentIndex)
                probability = 1.0 / len(actions)
                nextAgent = (agentIndex + 1) % state.getNumAgents()
                # If all ghosts have moved, it's Pacman's turn and we increase the depth
                nextDepth = depth + 1 if nextAgent == 0 else depth
                for action in actions:
                    successor = state.generateSuccessor(agentIndex, action)
                    value += probability * expectimax(successor, nextAgent, nextDepth)
                return value

        # The actual getAction function starts here
        bestAction = None
        bestValue = float('-inf')
        for action in gameState.getLegalActions(0):  # 0 is Pacman's index
            successorState = gameState.generateSuccessor(0, action)
            value = expectimax(successorState, 1, 0)  # Start with first ghost and depth 0
            if value > bestValue:
                bestValue = value
                bestAction = action

        return bestAction

def betterEvaluationFunction(currentGameState: GameState):
    """
    Your extreme ghost-hunting, pellet-nabbing, food-gobbling, unstoppable
    evaluation function (question 5).

    DESCRIPTION: This evaluation function considers several features:
    1. The current score
    2. The distance to the closest food
    3. The number of remaining food pellets
    4. The distance to the closest ghost
    5. The number of capsules (power pellets) left
    6. The game state (win/lose)
    """
    # Useful information you can extract from a GameState (pacman.py)
    newPos = currentGameState.getPacmanPosition()
    newFood = currentGameState.getFood()
    newGhostStates = currentGameState.getGhostStates()
    newScaredTimes = [ghostState.scaredTimer for ghostState in newGhostStates]

    # Initialize the score with the current game score
    score = currentGameState.getScore()

    # Calculate distance to the closest food
    foodList = newFood.asList()
    if foodList:
        closestFoodDist = min(manhattanDistance(newPos, food) for food in foodList)
        score += 1.0 / (closestFoodDist + 1)  # Closer food is better

    # Consider the number of remaining food pellets
    numFood = len(foodList)
    score -= 4 * numFood  # Fewer remaining food pellets is better

    # Evaluate ghost positions
    for ghostState in newGhostStates:
        ghostPos = ghostState.getPosition()
        ghostDist = manhattanDistance(newPos, ghostPos)
        if ghostState.scaredTimer > 0:
            # If the ghost is scared, we want to eat it
            score += 200.0 / (ghostDist + 1)
        else:
            # If the ghost is not scared, we want to avoid it
            if ghostDist < 2:
                score -= 500  # Heavy penalty for being very close to a ghost
            else:
                score -= 100.0 / ghostDist  # Less penalty for being further away

    # Consider capsules
    capsules = currentGameState.getCapsules()
    score -= 50 * len(capsules)  # Fewer remaining capsules is better

    # Heavily reward winning states and heavily penalize losing states
    if currentGameState.isWin():
        score += 1000
    elif currentGameState.isLose():
        score -= 1000

    return score

# Abbreviation
better = betterEvaluationFunction