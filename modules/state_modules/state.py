import os
import sys
import numpy as np
import copy

import config

class State:
    def __init__(
        self,
        current_players={},
        current_bet=0,
        all_in_players={},
        folded_players={},
        table=np.array([], dtype=int),
        pot=0,
        action_history=[[]],
    ):
        self.current_players = current_players
        self.current_bet = current_bet
        self.all_in_players = all_in_players
        self.folded_players = folded_players
        self.table = table
        self.pot = pot
        self.action_history = action_history

    def copy(self):

        # Create a new state with the same properties as the current one
        child = State(
            copy.deepcopy(self.current_players),
            self.current_bet,
            copy.deepcopy(self.all_in_players),
            copy.deepcopy(self.folded_players),
            self.table.copy(),
            self.pot,
            copy.deepcopy(self.action_history),
        )
        return child