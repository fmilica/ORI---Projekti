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

    def getFeatures(self, state, action, isRed, agentIndex):
        # extract the grid of food and wall locations and get the ghost locations
        food = 0
        capsules = []
        opponents = []
        greater = None
        agentPosition = state.getAgentPosition(agentIndex)

        if isRed:
            food = state.getBlueFood()
            capsules = state.getBlueCapsules()
            oppon = state.getBlueTeamIndices()
            opponents = [state.getAgentPosition(o) for o in oppon]
            greater = True
        else:
            food = state.getRedFood()
            capsules = state.getRedCapsules()
            oppon = state.getRedTeamIndices()
            opponents = [state.getAgentPosition(o) for o in oppon]
            greater = False

        walls = state.getWalls()
        ghosts = state.getGhostPositions()

        features = util.Counter()

        #features["bias"] = 1.0

        # compute the location of pacman after he takes the action
        x, y = state.getPacmanPosition()
        dx, dy = Actions.directionToVector(action)
        next_x, next_y = int(x + dx), int(y + dy)
        '''
        # count the number of ghosts 1-step away
        features["#-of-ghosts-1-step-away"] = sum((next_x, next_y) in Actions.getLegalNeighbors(g, walls) for g in ghosts)

        # if there is no danger of ghosts then add the food feature
        if not features["#-of-ghosts-1-step-away"] and food[next_x][next_y]:
            features["eats-food"] = 10.0
        '''
        dist = closestFood((next_x, next_y), food, walls)
        if dist is not None:
            # make the distance a number less than one otherwise the update
            # will diverge wildly
            features["closest-food"] = float(dist) / (walls.width * walls.height)

        dists = []
        for g in ghosts:
            if g in opponents:
                dists.append(manhattanDistance(g, (x,y)))

                if manhattanDistance(g, agentPosition) <= COLLISION_TOLERANCE:
                    features['eaten'] = -50
                else:
                    features['eaten'] = 1
        if min(dists) < 11:
            features['ghost-distance'] = -15
        else:
            features['ghost-distance'] = 0

        #print("agentPosition[0]")
        #print(agentPosition[0])
        if state.getAgentState(agentIndex).numCarrying < 5:
            features['num-carring'] = -5
        elif agentPosition[0] >= walls.width/2:
            #features['go-back'] = 10
            pass
        else:
            features['num-carring'] = 5
            #features['go-back'] = 0

        features['capsules-eaten'] = -len(capsules)

        features.divideAll(10.0)
        return features