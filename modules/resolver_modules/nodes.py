import numpy as np
import os
import sys

parent_parent_dir = os.path.dirname(os.path.dirname(os.path.dirname(os.path.abspath(__file__))))
sys.path.append(parent_parent_dir)

from  modules.state_modules.state_manager import State_Manager

import config

class Node: # Node class
    def __init__(self, state, parent, player_name=None):
        self.parent = parent
        self.children = []
        self.state = state
        self.leaf_node = False
        self.player_name = (
            player_name
        )



class Player_Node(Node): # Player_Node is a subclass of Node
    def __init__(self, player_name, action, parent, state=None):
        if state == None:
            state = State_Manager.process_action(
                parent.player_name, action, parent.state
            )
        super().__init__(
            state,
            parent,
            player_name,  # the name of the
        )
        self.action = action  # The action leading to this node
        self.strategy = self.create_strategy(player_name, state)
        self.regret = np.zeros((config.deck_size, config.deck_size, 5))
        self.R1 = np.full((config.deck_size,config.deck_size), 2 / config.deck_size**2)
        self.R2 = np.full((config.deck_size,config.deck_size), 2 / config.deck_size**2)

    def create_strategy(self, player_name, state):
        from modules.game_manager import Game_Manager
        legal_actions = Game_Manager.get_legal_actions(player_name, state)
        strategy = np.zeros((config.deck_size, config.deck_size, 5))
        if len(legal_actions) > 0:
            for action in legal_actions:
                strategy[:, :, self.get_action_index(action)] = 1 / len(
                    legal_actions
                )
        return strategy
    
    def get_action_index(self,action):
        return [
            "fold",
            "all in",
            "check",
            f"raise {config.blinds[1]}",
            f"raise {config.blinds[0]}",
        ].index(action)


class Chance_Node(Node): # Chance_Node is a subclass of Node
    def __init__(self, parent, action):
        super().__init__(
            State_Manager.process_action(parent.player_name, action, parent.state),
            parent,
        )
        self.action = action