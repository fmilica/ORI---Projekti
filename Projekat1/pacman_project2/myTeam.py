# myTeam.py
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


from captureAgents import CaptureAgent
import random, time, util
from game import Directions, Actions
import game
from util import nearestPoint, manhattanDistance
from featureExtractors import SimpleExtractor, CoordinateExtractor, IdentityExtractor

from baselineTeam import DefensiveReflexAgent
from capture import COLLISION_TOLERANCE

DEBUG = False
interestingValues = {}
SIGHT_RANGE = 5 # Manhattan distance
DEFENSE_TIMER_MAX = 100.0
USE_BELIEF_DISTANCE = True
arguments = {}

MINIMUM_PROBABILITY = .0001
PANIC_TIME = 80
CHEAT = False
beliefs = []
beliefsInitialized = []
FORWARD_LOOKING_LOOPS = 1

#################
# Team creation #
#################

def createTeam(firstIndex, secondIndex, isRed,
               first = 'ProbaAgent', second = 'ProbaDefensiveAgent', **args):
  """
  This function should return a list of two agents that will form the
  team, initialized using firstIndex and secondIndex as their agent
  index numbers.  isRed is True if the red team is being created, and
  will be False if the blue team is being created.

  As a potentially helpful development aid, this function can take
  additional string-valued keyword arguments ("first" and "second" are
  such arguments in the case of this function), which will come from
  the --redOpts and --blueOpts command-line arguments to capture.py.
  For the nightly contest, however, your team will be created without
  any extra arguments, so you should make sure that the default
  behavior is what you want for the nightly contest.
  """

  # The following line is an example only; feel free to change it.
  if 'numTraining' in args:
    interestingValues['numTraining'] = args['numTraining']
  return [eval(first)(firstIndex, args['numTraining']), eval(second)(secondIndex)]

##########
# Agents #
##########


class ProbaAgent(CaptureAgent):
    def __init__(self, index, numTraining=10, extractor=SimpleExtractor(), epsilon=0.05, alpha=1.0, gamma=0.8):
        CaptureAgent.__init__(self, index)
        self.weights = util.Counter()
        self.qValues = util.Counter()
        self.alpha = float(alpha)
        self.gamma = float(gamma)
        self.epsilon = float(epsilon)
        self.numTraining = numTraining
        self.episodes = 0
        self.extractor = extractor

    def registerInitialState(self, gameState):
        """
          Mozda nesto pametno treba da se radi
        """
        CaptureAgent.registerInitialState(self, gameState)

    def final(self, gameState):
        if self.episodes < self.numTraining:
            self.episodes += 1
            print('Training no. %d done' % self.episodes)
        elif self.episodes == self.numTraining:
            print('Training done (turning off alpha and epsilon)')
            self.alpha = 0.0
            self.epsilon = 0.0

    def getFeatures(self, gameState, action):
        self.extractor.getFeatures(gameState, action, self.isOnRedTeam(), self.index, self)

    def getQValue(self, state, action):
        """
          Should return Q(state,action) = w * featureVector
          where * is the dotProduct operator
        """
        total = 0.0
        weights = self.weights
        features = self.extractor.getFeatures(state, action, self.isOnRedTeam(), self.index, self)
        for feature in features:
            # Implements the Q calculation
            total += features[feature] * weights[feature]
        return total

    def computeValueFromQValues(self, state):
        """
          Returns max_action Q(state,action)
          where the max is over legal actions.  Note that if
          there are no legal actions, which is the case at the
          terminal state, you should return a value of 0.0.
        """
        possibleStateQValues = util.Counter()
        for action in state.getLegalActions(self.index):
            possibleStateQValues[action] = self.getQValue(state, action)

        if len(possibleStateQValues) > 0:
            return possibleStateQValues[possibleStateQValues.argMax()]
        return 0.0

    def getQValueState(self, state, action):
        """
          Returns Q(state,action)
          Should return 0.0 if we have never seen a state
          or the Q node value otherwise
        """
        return self.qValues[(state, action)]

    def computeActionFromQValues(self, state):
        """
          Compute the best action to take in a state.  Note that if there
          are no legal actions, which is the case at the terminal state,
          you should return None.
        """
        possibleStateQValues = util.Counter()
        possibleActions = state.getLegalActions(self.index)
        if len(possibleActions) == 0:
            return None

        for action in possibleActions:
            possibleStateQValues[action] = self.getQValue(state, action)

        best_actions = []
        best_value = possibleStateQValues[possibleStateQValues.argMax()]

        for action, value in possibleStateQValues.items():
            if value == best_value:
                best_actions.append(action)

        return random.choice(best_actions)

    def chooseAction(self, state):
        # Pick Action
        legalActions = state.getLegalActions(self.index)
        action = None

        if len(legalActions) > 0:
            if util.flipCoin(self.epsilon):
                action = random.choice(legalActions)
            else:
                action = self.computeActionFromQValues(state)

        foodCarry = state.getAgentState(self.index).numCarrying
        if foodCarry == 5:
            bestDist = 9999
            bestAction = None
            for action in legalActions:
                successor = self.getSuccessor(state, action)
                pos2 = successor.getAgentPosition(self.index)
                dist = self.getMazeDistance(self.start, pos2)
                if dist < bestDist:
                    bestAction = action
                    bestDist = dist
            return bestAction

        # Generate successor
        nextState = self.getSuccessor(state, action)
        reward = self.calculateReward(state, nextState)
        self.update(state, action, nextState, reward)
        return action

    def calculateReward(self, state, nextState):
        reward = -1

        if state.getAgentState(self.index).numCarrying:
            reward += 2

        if self.isOnRedTeam():
            firstHalf = True
            oppon = state.getBlueTeamIndices()
            opponents = [nextState.getAgentPosition(o) for o in oppon]
            if state.getAgentPosition(self.index)[0] > state.getWalls().width/2:
                reward += 3
            if len(state.getBlueCapsules()) != len(nextState.getBlueCapsules()):
                reward += 5
            if state.getAgentState(self.index).numCarrying and (state.getAgentPosition(self.index)[0] < state.getWalls().width/2 - 1):
                reward += 5
        else:
            firstHalf = False
            oppon = state.getRedTeamIndices()
            opponents = [nextState.getAgentPosition(o) for o in oppon]
            if state.getAgentPosition(self.index)[0] < state.getWalls().width/2:
                reward += 3
            if len(state.getRedCapsules()) != len(nextState.getRedCapsules()):
                reward += 5
            if state.getAgentState(self.index).numCarrying and (state.getAgentPosition(self.index)[0] > state.getWalls().width/2 + 1):
                reward += 5
        if state.data._foodEaten is not None:
            reward += 2

        ghosts = nextState.getGhostPositions()
        for g in ghosts:
            if g in opponents:
                #print('razdaljina %f' % manhattanDistance(g, nextState.getAgentPosition(self.index)))
                if manhattanDistance(g, nextState.getAgentPosition(self.index)) <= 1:
                    if self.scaredGhost(nextState, oppon, g):
                        reward += 5
                    else:
                        reward -= 7

        return reward

    def scaredGhost(self, state, opponents, pos):
        x, y = pos
        for o in opponents:
            agent = state.getAgentState(o)
            if state.getAgentPosition(o) == pos and agent.scaredTimer > 0:
                return True
        return False

    def update(self, state, action, nextState, reward):

        '''self.qValues[(state, action)] = self.getQValueState(state, action) + self.alpha * (
                reward + self.gamma * self.computeValueFromQValues(nextState) - self.getQValueState(state, action))'''

        features = self.extractor.getFeatures(state, action, self.isOnRedTeam(), self.index, self)
        diff = self.alpha * ((reward + self.gamma * self.computeValueFromQValues(nextState)) - self.getQValue(state, action))
        for feature in features.keys():
            self.weights[feature] = self.weights[feature] + diff * features[feature]
        #print(self.weights)

    def getSuccessor(self, gameState, action):
        """
        Finds the next successor which is a grid position (location tuple).
        """
        successor = gameState.generateSuccessor(self.index, action)
        pos = successor.getAgentState(self.index).getPosition()
        if pos != nearestPoint(pos):
            # Only half a grid position was covered
            return successor.generateSuccessor(self.index, action)
        else:
            return successor

class DummyAgent(CaptureAgent):
  """
  A Dummy agent to serve as an example of the necessary agent structure.
  You should look at baselineTeam.py for more details about how to
  create an agent as this is the bare minimum.
  """

  def registerInitialState(self, gameState):
    """
    This method handles the initial setup of the
    agent to populate useful fields (such as what team
    we're on).

    A distanceCalculator instance caches the maze distances
    between each pair of positions, so your agents can use:
    self.distancer.getDistance(p1, p2)

    IMPORTANT: This method may run for at most 15 seconds.
    """

    '''
    Make sure you do not delete the following line. If you would like to
    use Manhattan distances instead of maze distances in order to save
    on initialization time, please take a look at
    CaptureAgent.registerInitialState in captureAgents.py.
    '''
    CaptureAgent.registerInitialState(self, gameState)

    '''
    Your initialization code goes here, if you need any.
    '''


  def chooseAction(self, gameState):
    """
    Picks among actions randomly.
    """
    actions = gameState.getLegalActions(self.index)

    '''
    You should change this in your own agent.
    '''

    return random.choice(actions)

class ProbaDefensiveAgent(CaptureAgent):
    def __init__(self, index, numTraining=100, extractor=SimpleExtractor(), epsilon=0.05, alpha=1.0, gamma=0.8):
        CaptureAgent.__init__(self, index)
        self.weights = util.Counter()
        self.qValues = util.Counter()
        self.alpha = float(alpha)
        self.gamma = float(gamma)
        self.epsilon = float(epsilon)
        self.numTraining = numTraining
        self.episodes = 0
        self.extractor = extractor

    def registerInitialState(self, gameState):
        """
          Mozda nesto pametno treba da se radi
        """
        CaptureAgent.registerInitialState(self, gameState)

    def final(self, gameState):
        if self.episodes == self.numTraining:
            self.alpha = 0.0
            self.epsilon = 0.0

    def getFeatures(self, gameState, action):
        self.extractor.getDeffensiveFeatures(gameState, action, self.isOnRedTeam(), self.index)

    def getQValue(self, state, action):
        """
          Should return Q(state,action) = w * featureVector
          where * is the dotProduct operator
        """
        total = 0.0
        weights = self.weights
        features = self.extractor.getDeffensiveFeatures(self.getSuccessor(state, action), state, action, self.isOnRedTeam(), self.index,self)
        for feature in features:
            # Implements the Q calculation
            total += features[feature] * weights[feature]
        return total

    def computeValueFromQValues(self, state):
        """
          Returns max_action Q(state,action)
          where the max is over legal actions.  Note that if
          there are no legal actions, which is the case at the
          terminal state, you should return a value of 0.0.
        """
        possibleStateQValues = util.Counter()
        for action in state.getLegalActions(self.index):
            possibleStateQValues[action] = self.getQValue(state, action)

        if len(possibleStateQValues) > 0:
            return possibleStateQValues[possibleStateQValues.argMax()]
        return 0.0

    def getQValueState(self, state, action):
        """
          Returns Q(state,action)
          Should return 0.0 if we have never seen a state
          or the Q node value otherwise
        """
        return self.qValues[(state, action)]

    def computeActionFromQValues(self, state):
        """
          Compute the best action to take in a state.  Note that if there
          are no legal actions, which is the case at the terminal state,
          you should return None.
        """
        possibleStateQValues = util.Counter()
        possibleActions = state.getLegalActions(self.index)
        if len(possibleActions) == 0:
            return None

        for action in possibleActions:
            possibleStateQValues[action] = self.getQValue(state, action)

        best_actions = []
        best_value = possibleStateQValues[possibleStateQValues.argMax()]

        for action, value in possibleStateQValues.items():
            if value == best_value:
                best_actions.append(action)

        return random.choice(best_actions)

    def chooseAction(self, state):
        # Pick Action
        legalActions = state.getLegalActions(self.index)
        action = None

        if len(legalActions) > 0:
            if util.flipCoin(self.epsilon):
                action = random.choice(legalActions)
            else:
                action = self.computeActionFromQValues(state)

        # Generate successor
        nextState = self.getSuccessor(state, action)
        reward = self.calculateReward(state, nextState)
        self.update(state, action, nextState, reward)
        return action

    def calculateReward(self, state, nextState):
        reward = -1

        myGhost = state.getAgentState(self.index)

        if self.isOnRedTeam():
            oppon = state.getBlueTeamIndices()
            opponents = [nextState.getAgentPosition(o) for o in oppon]
        else:
            oppon = state.getRedTeamIndices()
            opponents = [nextState.getAgentPosition(o) for o in oppon]

        for o in oppon:
            if state.getAgentState(o).isPacman:
                '''if state.getAgentState(o).numCarrying > 0:
                    reward -= 3
                el'''
                if state.getAgentState(o).numReturned > 0:
                    reward -= state.getAgentState(o).numReturned * 2

        for g in opponents:
            if manhattanDistance(g, nextState.getAgentPosition(self.index)) <= 1:
                if myGhost.scaredTimer > 0:
                    reward -= 6
                else:
                    reward += 10

        #print(reward)
        return reward

    def scaredGhost(self, state, opponents, pos):
        x, y = pos
        for o in opponents:
            agent = state.getAgentState(o)
            if state.getAgentPosition(o) == pos and agent.scaredTimer > 0:
                return True
        return False

    def update(self, state, action, nextState, reward):

        '''self.qValues[(state, action)] = self.getQValueState(state, action) + self.alpha * (
                reward + self.gamma * self.computeValueFromQValues(nextState) - self.getQValueState(state, action))'''

        features = self.extractor.getDeffensiveFeatures(self.getSuccessor(state, action), state, action, self.isOnRedTeam(), self.index,self)
        diff = self.alpha * ((reward + self.gamma * self.computeValueFromQValues(nextState)) - self.getQValue(state, action))
        for feature in features.keys():
            self.weights[feature] = self.weights[feature] + diff * features[feature]
        #print(self.weights)

    def getSuccessor(self, gameState, action):
        """
        Finds the next successor which is a grid position (location tuple).
        """
        successor = gameState.generateSuccessor(self.index, action)
        pos = successor.getAgentState(self.index).getPosition()
        if pos != nearestPoint(pos):
            # Only half a grid position was covered
            return successor.generateSuccessor(self.index, action)
        else:
            return successor

def closestFood(pos, food, walls):
        """
        closestFood -- this is similar to the function that we have
        worked on in the search project; here its all in one place
        """
        fringe = [(pos[0], pos[1], 0)]
        expanded = set()
        while fringe:
            pos_x, pos_y, dist = fringe.pop(0)
            if (pos_x, pos_y) in expanded:
                continue
            expanded.add((pos_x, pos_y))
            # if we find a food at this location then exit
            if food[pos_x][pos_y]:
                return dist
            # otherwise spread out from the location to its neighbours
            nbrs = Actions.getLegalNeighbors((pos_x, pos_y), walls)
            for nbr_x, nbr_y in nbrs:
                fringe.append((nbr_x, nbr_y, dist + 1))
        # no food found
        return None
