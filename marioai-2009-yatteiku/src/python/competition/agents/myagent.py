from enum import IntEnum

import numpy
from .marioagent import MarioAgent
from ga.individual import Individual

class ActionIndex(IntEnum):
    RIGHT = 1
    JUMP = 3
    SPEED = 4

class MyAgent(MarioAgent):
    def __init__(self, individual):
        self.usedActions = 3 # RIGHT, JUMP, SPEED
        self.individual = individual

    def reset(self):
        self.action = numpy.zeros(5, int)

    def getAction(self):
        decision = self.individual.action(self.levelScene, self.isMarioOnGround, self.mayMarioJump)
        self.action.fill(0)
        for i, action_idx in enumerate([ActionIndex.RIGHT, ActionIndex.JUMP, ActionIndex.SPEED]):
            if decision[i+1]:
                self.action[action_idx] = 1
        return self.action

    def integrateObservation(self, obs):
        if (len(obs) != 6):
            pass # Episode is over
        else:
            self.mayMarioJump, self.isMarioOnGround, self.marioFloats, self.enemiesFloats, self.levelScene, dummy = obs
