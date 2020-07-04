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
               first = 'ProbaAgent', second = 'DefensiveReflexAgent', **args):
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
  return [eval(first)(firstIndex), eval(second)(secondIndex)]

##########
# Agents #
##########


class ProbaAgent(CaptureAgent):
    def __init__(self, index, extractor=SimpleExtractor(), numTraining=100, epsilon=0.05, alpha=0.5, gamma=0.8):
        CaptureAgent.__init__(self, index)
        self.weights = util.Counter()
        self.qValues = util.Counter()
        self.alpha = float(alpha)
        self.gamma = float(gamma)
        self.epsilon = float(epsilon)
        self.numTraining = numTraining
        self.extractor = extractor

    def registerInitialState(self, gameState):
        """
          Mozda nesto pametno treba da se radi
        """
        CaptureAgent.registerInitialState(self, gameState)

    def getFeatures(self, gameState, action):
        self.extractor.getFeatures(gameState, action, self.isOnRedTeam(), self.index)

    def getQValue(self, state, action):
        """
          Should return Q(state,action) = w * featureVector
          where * is the dotProduct operator
        """
        total = 0.0
        weights = self.weights
        features = self.extractor.getFeatures(state, action, self.isOnRedTeam(), self.index)
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
            possibleStateQValues[action] = self.qValues[(state, action)]

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
        # nagrada ako pokupi hranu
        if self.getFood(state) != self.getFood(nextState):
            reward = 1
        return reward

    def update(self, state, action, nextState, reward):
        '''
        self.weights[(state, action)] = self.getQValue(state, action) + self.alpha * (
        reward + self.gamma * self.computeValueFromQValues(nextState) - self.getQValue(state, action))
        '''

        self.qValues[(state, action)] = self.getQValue(state, action) + self.alpha * (
                reward + self.gamma * self.computeValueFromQValues(nextState) - self.getQValue(state, action))

        features = self.extractor.getFeatures(state, action, self.isOnRedTeam(), self.index)
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

'''
class ApproximateQAgent(CaptureAgent):

    def __init__( self, index ):
        CaptureAgent.__init__(self, index)
        self.weights = util.Counter()
        self.numTraining = 0
        if 'numTraining' in interestingValues:
            self.numTraining = interestingValues['numTraining']
        self.episodesSoFar = 0
        self.epsilon = 0.05
        self.discount = 0.8
        self.alpha = 0.2
        self.lastState = None
        self.lastAction = Directions.STOP
        self.episodeRewards = 0.0
        self.accumTrainRewards = 0.0
        self.accumTestRewards = 0.0

    def registerInitialState(self, gameState):
        self.start = gameState.getAgentPosition(self.index)
        self.lastAction = Directions.STOP
        CaptureAgent.registerInitialState(self, gameState)

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

    def chooseAction(self, state):
        # Append game state to observation history...
        self.observationHistory.append(state)
        # Pick Action
        legalActions = state.getLegalActions(self.index)
        action = None
        if (DEBUG):
            print("AGENT " + str(self.index) + " choosing action!")
        if len(legalActions):
            if util.flipCoin(self.epsilon) and self.isTraining():
                action = random.choice(legalActions)
                if (DEBUG):
                    print("ACTION CHOSE FROM RANDOM: " + action)
            else:
                action = self.computeActionFromQValues(state)
                if (DEBUG):
                    print("ACTION CHOSE FROM Q VALUES: " + action)

        self.lastAction = action
        """ 
        TODO
        ReflexCaptureAgent has some code that returns to your side if there are less than 2 pellets
        We added that here
        """
        foodLeft = len(self.getFood(state).asList())

        if foodLeft <= 2:
            bestDist = 9999
            for a in legalActions:
                successor = self.getSuccessor(state, a)
                pos2 = successor.getAgentPosition(self.index)
                dist = self.getMazeDistance(self.start,pos2)
                if dist < bestDist:
                    action = a
                    bestDist = dist

        if (DEBUG):
            print("AGENT " + str(self.index) + " chose action " + action + "!")

        self.lastState = state
        self.lastAction = action
        return action

    def getFeatures(self, gameState, action):
        """
        Returns a counter of features for the state
        """
        successor = self.getSuccessor(gameState, action)
        features = util.Counter()
        features['score'] = self.getScore(successor)
        if not self.red:
            features['score'] *= -1
        features['choices'] = len(successor.getLegalActions(self.index))

        """
        food = gameState.getFood()
        walls = gameState.getWalls()
        ghosts = gameState.getGhostPositions()
        
        features = util.Counter()
        
        features["bias"] = 1.0
        
        # compute the location of pacman after he takes the action
        x, y = gameState.getPacmanPosition()
        dx, dy = Actions.directionToVector(action)
        next_x, next_y = int(x + dx), int(y + dy)
        
        # count the number of ghosts 1-step away
        features["score"] = sum(
            (next_x, next_y) in Actions.getLegalNeighbors(g, walls) for g in ghosts)
        
        # if there is no danger of ghosts then add the food feature
        if not features["score"] and food[next_x][next_y]:
            features["eats-food"] = 1.0
        
        dist = closestFood((next_x, next_y), food, walls)
        if dist is not None:
            # make the distance a number less than one otherwise the update
            # will diverge wildly
            features["closest-food"] = float(dist) / (walls.width * walls.height)
        features.divideAll(10.0)
        return features
        """
        return features

    def computeActionFromQValues(self, state):
        """
          Compute the best action to take in a state.  Note that if there
          are no legal actions, which is the case at the terminal state,
          you should return None.
        """
        bestValue = -999999
        bestActions = None
        for action in state.getLegalActions(self.index):
            # For each action, if that action is the best then
            # update bestValue and update bestActions to be
            # a list containing only that action.
            # If the action is tied for best, then add it to
            # the list of actions with the best value.
            value = self.getQValue(state, action)
            if (DEBUG):
                print("ACTION: " + action + "           QVALUE: " + str(value))
            if value > bestValue:
                bestActions = [action]
                bestValue = value
            elif value == bestValue:
                bestActions.append(action)
        if bestActions == None:
            return Directions.STOP # If no legal actions return None
        return random.choice(bestActions) # Else choose one of the best actions randomly


    def getWeights(self):
        return self.weights

    def computeValueFromQValues(self, state):
        """
          Returns max_action Q(state,action)
          where the max is over legal actions.  Note that if
          there are no legal actions, which is the case at the
          terminal state, you should return a value of 0.0.
        """
        bestValue = -999999
        noLegalActions = True
        for action in state.getLegalActions(self.index):
            # For each action, if that action is the best then
            # update bestValue
            noLegalActions = False
            value = self.getQValue(state, action)
            if value > bestValue:
                bestValue = value
        if noLegalActions:
            return 0 # If there is no legal action return 0
        # Otherwise return the best value found
        return bestValue

    def getQValue(self, state, action):
        """
          Should return Q(state,action) = w * featureVector
          where * is the dotProduct operator
        """
        total = 0
        weights = self.getWeights()
        features = self.getFeatures(state, action)
        for feature in features:
            # Implements the Q calculation
            total += features[feature] * weights[feature]
        return total

    def getReward(self, gameState):
        foodList = self.getFood(gameState).asList()
        return -len(foodList)

    def observationFunction(self, gameState):
        if len(self.observationHistory) > 0 and self.isTraining():
            self.update(self.getCurrentObservation(), self.lastAction, gameState, self.getReward(gameState))
        return gameState.makeObservation(self.index)

    def isTraining(self):
        return self.episodesSoFar < self.numTraining

    def update(self, state, action, nextState, reward):
        """
           Should update your weights based on transition
        """
        if (DEBUG):
            print(self.newline())
            print("AGENT " + str(self.index) + " updating weights!")
            print("Q VALUE FOR NEXT STATE: " + str(self.computeValueFromQValues(nextState)))
            print("Q VALUE FOR CURRENT STATE: " + str(self.getQValue(state, action)))
        difference = (reward + self.discount * self.computeValueFromQValues(nextState)) * self.alpha
        difference -= self.getQValue(state, action)
        # Only calculate the difference once, not in the loop.
        newWeights = self.weights.copy()
        # Same with weights and features.
        features = self.getFeatures(state, action)
        for feature in features:
            # Implements the weight updating calculations
            newWeight = newWeights[feature] + difference * features[feature]
            if (DEBUG):
                print("AGENT " + str(self.index) + " weights for " + feature + ": " + str(newWeights[feature]) + " ---> " + str(newWeight))
            newWeights[feature]  = newWeight
        self.weights = newWeights.copy()
        #print "WEIGHTS AFTER UPDATE"
        #print self.weights

    def newline(self):
        return "-------------------------------------------------------------------------"

    def final(self, state):
        "Called at the end of each game."
        # call the super-class final method
        CaptureAgent.qfinal(self, state)
        if self.isTraining() and DEBUG:
            print("END WEIGHTS")
            print(self.weights)
        self.episodesSoFar += 1
        if self.episodesSoFar == self.numTraining:
            print("FINISHED TRAINING")

'''
class ApproximateAdversarialAgent(CaptureAgent):
    """
      Superclass for agents choosing actions via alpha-beta search, with
      positions of unseen enemies approximated by Bayesian inference
    """
    #####################
    # AI algorithm code #
    #####################

    SEARCH_DEPTH = 5

    def registerInitialState(self, gameState):
        CaptureAgent.registerInitialState(self, gameState)

        # Get all non-wall positions on the board
        self.legalPositions = gameState.data.layout.walls.asList(False)

        # Initialize position belief distributions for opponents
        self.positionBeliefs = {}
        for opponent in self.getOpponents(gameState):
          self.initializeBeliefs(opponent)

    def initializeBeliefs(self, agent):
        """
        Uniformly initialize belief distributions for opponent positions.
        """
        self.positionBeliefs[agent] = util.Counter()
        for p in self.legalPositions:
          self.positionBeliefs[agent][p] = 1.0

    def chooseAction(self, gameState):
        # Update belief distribution about opponent positions and place hidden
        # opponents in their most likely positions
        myPosition = gameState.getAgentState(self.index).getPosition()
        noisyDistances = gameState.getAgentDistances()
        probableState = gameState.deepCopy()

        for opponent in self.getOpponents(gameState):
          pos = gameState.getAgentPosition(opponent)
          if pos:
            self.fixPosition(opponent, pos)
          else:
            self.elapseTime(opponent, gameState)
            self.observe(opponent, noisyDistances[opponent], gameState)


        for opponent in self.getOpponents(gameState):
          probablePosition = self.guessPosition(opponent)
          conf = game.Configuration(probablePosition, Directions.STOP)
          probableState.data.agentStates[opponent] = game.AgentState(
            conf, probableState.isRed(probablePosition) != probableState.isOnRedTeam(opponent))

        # Run negamax alpha-beta search to pick an optimal move
        bestVal, bestAction = float("-inf"), None
        for opponent in self.getOpponents(gameState):
          value, action = self.expectinegamax(opponent,
                                              probableState,
                                              self.SEARCH_DEPTH,
                                              1,
                                              retAction=True)
          if value > bestVal:
            bestVal, bestAction = value, action

        return action

    def fixPosition(self, agent, position):
        """
        Fix the position of an opponent in an agent's belief distributions.
        """
        updatedBeliefs = util.Counter()
        updatedBeliefs[position] = 1.0
        self.positionBeliefs[agent] = updatedBeliefs

        def elapseTime(self, agent, gameState):
            """
            Elapse belief distributions for an agent's position by one time step.
            Assume opponents move randomly, but also check for any food lost from
            the previous turn.
            """
            updatedBeliefs = util.Counter()
            for (oldX, oldY), oldProbability in self.positionBeliefs[agent].items():
              newDist = util.Counter()
              for p in [(oldX - 1, oldY), (oldX + 1, oldY),
                        (oldX, oldY - 1), (oldX, oldY + 1)]:
                if p in self.legalPositions:
                  newDist[p] = 1.0
              newDist.normalize()
              for newPosition, newProbability in newDist.items():
                updatedBeliefs[newPosition] += newProbability * oldProbability

            lastObserved = self.getPreviousObservation()
            if lastObserved:
              lostFood = [food for food in self.getFoodYouAreDefending(lastObserved).asList()
                          if food not in self.getFoodYouAreDefending(gameState).asList()]
              for f in lostFood:
                updatedBeliefs[f] = 1.0/len(self.getOpponents(gameState))

            self.positionBeliefs[agent] = updatedBeliefs


    def observe(self, agent, noisyDistance, gameState):
        """
        Update belief distributions for an agent's position based upon
        a noisy distance measurement for that agent.
        """
        myPosition = self.getAgentPosition(self.index, gameState)
        teammatePositions = [self.getAgentPosition(teammate, gameState)
                             for teammate in self.getTeam(gameState)]
        updatedBeliefs = util.Counter()

        for p in self.legalPositions:
          if any([util.manhattanDistance(teammatePos, p) <= SIGHT_RANGE
                  for teammatePos in teammatePositions]):
            updatedBeliefs[p] = 0.0
          else:
            trueDistance = util.manhattanDistance(myPosition, p)
            positionProbability = gameState.getDistanceProb(trueDistance, noisyDistance)
            updatedBeliefs[p] = positionProbability * self.positionBeliefs[agent][p]

        if not updatedBeliefs.totalCount():
          self.initializeBeliefs(agent)
        else:
          updatedBeliefs.normalize()
          self.positionBeliefs[agent] = updatedBeliefs

    def guessPosition(self, agent):
        """
        Return the most likely position of the given agent in the game.
        """
        return self.positionBeliefs[agent].argMax()

    def expectinegamax(self, opponent, state, depth, sign, retAction=False):
        """
        Negamax variation of expectimax.
        """
        if sign == 1:
          agent = self.index
        else:
          agent = opponent

        bestAction = None
        if self.stateIsTerminal(agent, state) or depth == 0:
          bestVal = sign * self.evaluateState(state)
        else:
          actions = state.getLegalActions(agent)
          if Directions.STOP in actions:
              actions.remove(Directions.STOP)
          bestVal = float("-inf") if agent == self.index else 0
          for action in actions:
            successor = state.generateSuccessor(agent, action)
            value = -self.expectinegamax(opponent, successor, depth - 1, -sign)
            if agent == self.index and value > bestVal:
              bestVal, bestAction = value, action
            elif agent == opponent:
              bestVal += value/len(actions)

        if agent == self.index and retAction:
          return bestVal, bestAction
        else:
          return bestVal

    def stateIsTerminal(self, agent, gameState):
        """
        Check if the search tree should stop expanding at the given game state
        on the given agent's turn.
        """
        return len(gameState.getLegalActions(agent)) == 0

    def evaluateState(self, gameState):
        """
        Evaluate the utility of a game state.
        """
        util.raiseNotDefined()

    #####################
    # Utility functions #
    #####################

    def getAgentPosition(self, agent, gameState):
        """
        Return the position of the given agent.
        """
        pos = gameState.getAgentPosition(agent)
        if pos:
          return pos
        else:
          return self.guessPosition(agent)

    def agentIsPacman(self, agent, gameState):
        """
        Check if the given agent is operating as a Pacman in its current position.
        """
        agentPos = self.getAgentPosition(agent, gameState)
        return (gameState.isRed(agentPos) != gameState.isOnRedTeam(agent))

    def getOpponentDistances(self, gameState):
        """
        Return the IDs of and distances to opponents, relative to this agent.
        """
        return [(o, self.distancer.getDistance(
                 self.getAgentPosition(self.index, gameState),
                 self.getAgentPosition(o, gameState)))
                for o in self.getOpponents(gameState)]


class DefensiveAgent(ApproximateAdversarialAgent):
  """
  A defense-oriented agent that should never cross into the opponent's territory.
  """
  TERMINAL_STATE_VALUE = -1000000

  def stateIsTerminal(self, agent, gameState):
    return self.agentIsPacman(self.index, gameState) or \
      ApproximateAdversarialAgent.stateIsTerminal(self, agent, gameState)


class HunterDefenseAgent(DefensiveAgent):
  """
  A defense-oriented agent that actively seeks out an enemy agent in its territory
  and tries to hunt it down
  """
  def evaluateState(self, gameState):
    myPosition = self.getAgentPosition(self.index, gameState)
    if self.agentIsPacman(self.index, gameState):
        return DefensiveAgent.TERMINAL_STATE_VALUE

    score = 0
    pacmanState = [self.agentIsPacman(opponent, gameState)
                   for opponent in self.getOpponents(gameState)]
    opponentDistances = self.getOpponentDistances(gameState)

    for isPacman, (id, distance) in zip(pacmanState, opponentDistances):
      if isPacman:
        score -= 100000
        score -= 5 * distance
      elif not any(pacmanState):
        score -= distance

    return score


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
