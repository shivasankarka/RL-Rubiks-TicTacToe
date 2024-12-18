import numpy as np
from copy import deepcopy
import random

from tictactoe import TicTacToe

ACTION_SPACE = 9

args = {
    'c': 1.0
}

class Node: 
    child: list
    visit_count: int
    reward: float
    env: TicTacToe
    terminate: bool
    parent: Node
    action: int
    
    def __init__(self, env: TicTacToe, terminate: bool, parent: Node, state: np.ndarray, action: int): 
        self.child = None
        self.reward = 0.0
        self.visit_count = 0
        self.env = env
        self.state = state 
        self.terminate = terminate
        self.parent = parent
        self.action = action
    
    def calcUCBscore(self):
        """
            Calculates the Upper confidence bound value         

        Returns:
            float: UCB score
        """
        if self.visit_count == 0:
            return np.inf

        top_node = self
        if top_node.parent:
            top_node = top_node.parent
        
        exploitation: float = self.reward / self.visit_count
        exploration: float = args['c'] * np.sqrt(np.log(top_node.visit_count) / self.visit_count)
        return exploitation + exploration
    
    def create_child_nodes(self):
        if self.terminate:
            return 
        
        actions = []
        games = []
        # I think I need to reduce the action space depending on the current state of the game. 
        for i in range(ACTION_SPACE): 
            actions.append(i)           
            games.append(deepcopy(self.game))

        child = {}
        for move, game in zip(actions, games):
            obs, ter, _ = game.step(move)
            child[move] = Node(game, ter, self, obs, move)
            
        self.child = child
        
    def explore(self):
        current_node = self

        while current_node.child: 
            child = current_node.child
            max_ucb = max(c.calcUCBscore() for c in child.values())
            actions = [a for a, c in child.items() if c.calcUCBscore() == max_ucb]
            move = np.random.choice(actions)
            current = child[move]

        if current.visit_count < 1:
            current.reward = current.reward + current.rollout()
        else: 
            current.create_child()
            if current.child:
                current = random.choice(current.child)
            current.reward = current.reward + current.rollout()

        current.visit_count += 1
        
        parent = current
        while parent:
            parent.visit_count += 1
            parent.reward += current.reward
            parent = parent.parent
    
    def rollout(self):
        if self.terminate:
            return 0

        val = 0
        terminate = False
        new_game = deepcopy(self.game)
        while not done: 
            action = new_game.action_space.sample()
            

            


            