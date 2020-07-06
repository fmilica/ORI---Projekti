# featureExtractors.py
# --------------------
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


"Feature extractors for Pacman game states"
from pyexpat import features

from game import Directions, Actions
import util
from util import manhattanDistance

from capture import COLLISION_TOLERANCE

class FeatureExtractor:
    def getFeatures(self, state, action):
        """
          Returns a dict from features to counts
          Usually, the count will just be 1.0 for
          indicator functions.
        """
        util.raiseNotDefined()

class IdentityExtractor(FeatureExtractor):
    def getFeatures(self, state, action):
        feats = util.Counter()
        feats[(state,action)] = 1.0
        return feats

class CoordinateExtractor(FeatureExtractor):
    def getFeatures(self, state, action):
        feats = util.Counter()
        feats[state] = 1.0
        feats['x=%d' % state[0]] = 1.0
        feats['y=%d' % state[0]] = 1.0
        feats['action=%s' % action] = 1.0
        return feats

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
            fringe.append((nbr_x, nbr_y, dist+1))
    # no food found
    return None

class SimpleExtractor(FeatureExtractor):
    """
    Returns simple features for a basic reflex Pacman:
    - whether food will be eaten
    - how far away the next food is
    - whether a ghost collision is imminent
    - whether a ghost is one step away
    """

    def inOtherHalf(self, pos, isRed, walls):
        x, y = pos
        if isRed:
            if x >= walls.width / 2:
                return True
            else:
                return False
        else:
            if x <= walls.width / 2:
                return True
            else:
                return False

    def closeToHome(self, isRed, pos, walls):
        x,y = pos
        distance = []
        if isRed:
            wh = walls.width/2 - 1
            for xh in range(walls.height):
                distance.append(manhattanDistance((xh, wh), (x, y)))
        else:
            wh = walls.width / 2 + 1
            for xh in range(walls.height):
                distance.append(manhattanDistance((xh, wh), (x, y)))

        return min(distance)

    def scaredGhost(self, state, opponents, pos):
        x, y = pos
        for o in opponents:
            agent = state.getAgentState(o)
            if state.getAgentPosition(o) == pos and agent.scaredTimer > 0:
                return True
        return False

    def getFeatures(self, state, action, isRed, agentIndex, agent):
        # extract the grid of food and wall locations and get the ghost locations
        food = 0
        capsules = []
        opponents = []
        greater = None
        agentPosition = state.getAgentPosition(agentIndex)

        walls = state.getWalls()
        ghosts = state.getGhostPositions()

        features = util.Counter()

        if isRed:
            food = state.getBlueFood()
            capsules = state.getBlueCapsules()
            oppon = state.getBlueTeamIndices()
            opponents = [state.getAgentPosition(o) for o in oppon]
            greater = True
            if state.getAgentPosition(agentIndex)[0] > walls.width / 2:
                features['attack'] = 1.0
            else:
                features['attack'] = -0.4
        else:
            food = state.getRedFood()
            capsules = state.getRedCapsules()
            oppon = state.getRedTeamIndices()
            opponents = [state.getAgentPosition(o) for o in oppon]
            greater = False

        # compute the location of pacman after he takes the action
        x, y = state.getAgentPosition(agentIndex)
        dx, dy = Actions.directionToVector(action)
        next_x, next_y = int(x + dx), int(y + dy)

        dists = []
        for g in ghosts:
            if g in opponents and self.inOtherHalf((next_x, next_y), isRed, walls) and not self.scaredGhost(state, oppon, g):
                dists.append(agent.getMazeDistance(g, (next_x, next_y)))

        dist = closestFood((next_x, next_y), food, walls)

        if dist is not None:
            # make the distance a number less than one otherwise the update
            # will diverge wildly
            features["closest-food"] = float(dist)*10 / (walls.width * walls.height)

        # count the number of ghosts 1-step away
        #features["#-of-ghosts-1-step-away"] = sum((next_x, next_y) in Actions.getLegalNeighbors(g, walls) for g in ghosts)

        # if there is no danger of ghosts then add the food feature
        '''if not features["#-of-ghosts-1-step-away"] and food[next_x][next_y]:
            features["eats-food"] = 1.0'''

        #print(dists)
        features['ghost-distance'] = -0.01
        '''for g in ghosts:
            if g in opponents and (next_x, next_y) in Actions.getLegalNeighbors(g, walls):
                features['ghost-distance'] = sum((next_x, next_y) in Actions.getLegalNeighbors(g, walls) for g in ghosts)'''

        if len(dists) > 0 and min(dists) < 5:
            features['ghost-distance'] = -float(min(dists))*10 / (walls.width * walls.height)
            if len(capsules) > 0:
                d = manhattanDistance(capsules[0], (next_x, next_y))
                if d < 5:
                    features['capsule-distance'] = min(dists)*10 / (walls.width * walls.height)
            dd = self.closeToHome(isRed, (next_x, next_y), walls)
            if dd < 8:
                features['go-back'] = -0.121
        else:
            features['ghost-distance'] = 1.0

        if state.getAgentState(agentIndex).numCarrying > 0:
            features['num-carring'] = 0.64
        else:
            features['num-carring'] = -0.001
            d = self.closeToHome(isRed, (next_x, next_y), walls)
            if d < 8:
                features['go-back'] = 0.061
            if len(dists) > 0 and min(dists) < 5:
                features['go-back'] = 0.0821

        if len(capsules) > 0:
            d = manhattanDistance(capsules[0], (next_x, next_y))
            if d > 5:
                features['capsule-distance'] = 0.645
            else:
                features['capsule-distance'] = -0.361

        features.divideAll(10.0)

        return features


    def getDeffensiveFeatures(self, successor ,state, action, isRed, agentIndex, agent):
        # extract the grid of food and wall locations and get the ghost locations
        walls = state.getWalls()
        myState = successor.getAgentState(agentIndex)
        myPos = myState.getPosition()
        features = util.Counter()

        if isRed:
            oppon = state.getBlueTeamIndices()
            opponents = [state.getAgentPosition(o) for o in oppon]
            if myPos[0] < (walls.width // 2)- 1:
                features['stay-on-your-side'] = 0.895
            else:
                features['stay-on-your-side'] = -0.92

            #ako nema nikoga na njegovoj strani, prebaci
        else:
            oppon = state.getRedTeamIndices()
            opponents = [state.getAgentPosition(o) for o in oppon]
            if myPos[0] > (walls.width // 2)- 1:
                features['stay-on-your-side'] = 0.895
            else:
                features['stay-on-your-side'] = -0.92
        # Computes whether we're on defense (1) or offense (0)
        features['onDefense'] = 1

        # Computes distance to invaders we can see
        enemies = [successor.getAgentState(i) for i in oppon]
        invaders = [a for a in enemies if a.isPacman and a.getPosition() != None]
        features['numInvaders'] = len(invaders)
        if len(invaders) > 0:
            dists = [agent.getMazeDistance(myPos, a.getPosition()) for a in invaders]
            features['invaderDistance'] = min(dists) * 10 / (walls.width * walls.height)

        if action == Directions.STOP: features['stop'] = 1
        rev = Directions.REVERSE[state.getAgentState(agentIndex).configuration.direction]
        if action == rev: features['reverse'] = 1

        features.divideAll(10.0)
        return features
