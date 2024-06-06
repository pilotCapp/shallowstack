import numpy as np
import os
import sys
import copy

parent_parent_dir = os.path.dirname(
    os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
)
sys.path.append(parent_parent_dir)

from modules.oracle import Poker_Oracle
from modules.NN_modules.nn_manager import NN_manager
from modules.resolver_modules.nodes import Node, Player_Node, Chance_Node
from modules.state_modules.state_manager import State_Manager

import config


class Resolver:
    def __init__(self, oracle=Poker_Oracle()):
        self.oracle = oracle
        self.players = ["P1", "P2"]
        self.utility_matrixes = {}
        self.NN_manager = NN_manager(self)

    def init_node(self, R1, R2, player_name, state): #creates a new node based on input
        new_state = state.copy()
        remaining_players = {**new_state.all_in_players, **new_state.current_players}
        
        new_state.current_players = {
            "P1": remaining_players.pop(player_name),
            "P2": remaining_players.popitem()[1],
        }

        node = Player_Node("P1", None, None, new_state)
        node.R1 = R1
        node.R2 = R2
        return node


    def resolve(self, state, R1, R2, player_name): #resolves the game tree and returns the optimal strategy
        from modules.game_manager import Game_Manager 
        strategy = np.zeros((config.deck_size, config.deck_size, 5))
        root = self.init_node(R1, R2, player_name, state)
        for k in range(config.resolving_iterations):
            self.generate_subtree(root)
            self.range_update(root)
            self.evaluate_regret_strategize(root)
            strategy = strategy + root.strategy
            
        strategy = strategy / np.sum(strategy, axis=2, keepdims=True)
        return strategy, root.R1, R2

    def bayesian_update(self, strategy, range, action): #updated the range based on the strategy and bayesian theorem
        sum_action = 0
        sum_array = 0

        sum_action = np.sum(strategy, axis=(0, 1))[action]
        sum_array = np.sum(strategy)

        range_array = (
            copy.deepcopy(range)
            * np.transpose(strategy, (2, 0, 1))[action]
            * (sum_action + 0.001)
            / (sum_array + 0.001)
        )
        return range_array  # TODO: Normalize?

    def filter_range(self, range, table): #filters the range based on the table, removing impossible hands
        range_array=copy.deepcopy(range)
        for k in table:
            range_array[:, k] = 0
            range_array[k, :] = 0
        return range_array

    def range_update(self, node): #updates the range based on the parent node
        if node.parent != None:
            if type(node.parent) == Chance_Node:
                node.R1 = self.filter_range(node.parent.parent.R1, node.state.table)
                node.R2 = self.filter_range(node.parent.parent.R2, node.state.table)
            else:
                if node.parent.player_name == "P1":
                    node.R1 = self.bayesian_update(
                        node.parent.strategy,
                        node.parent.R1,
                        Resolver.get_action_index(node.action),
                    )
                    node.R2 = copy.deepcopy(node.parent.R2)
                else:
                    node.R1 = copy.deepcopy(node.parent.R1)
                    node.R2 = self.bayesian_update(
                        node.parent.strategy,
                        node.parent.R2,
                        Resolver.get_action_index(node.action),
                    )
            node.R1 = node.R1 / (np.sum(node.R1)+0.001)
            node.R2 = node.R2 / (np.sum(node.R2)+0.001)
            
        for child in node.children:
            self.range_update(child)

    def evaluate_regret_strategize(self, node): #evaluates the regret and strategizes based on the regret, main resolving method
        from modules.game_manager import Game_Manager

        if ( #if the node is a fold node
            node.state.action_history[-1]
            and node.state.action_history[-1][-1]
            and node.state.action_history[-1][-1][1] == "fold"
        ):
            node.v1, node.v2 = self.fold_vector(
                node, node.state.action_history[-1][-1], node.state
            )
            return (node.v1, node.v2)
        elif node.state.table.size == 5 and node.leaf_node and type(node.parent) != Chance_Node: #if the node is a showdown node
            utility_matrix = self.get_utility(node.state.table)
            node.v1 = np.tensordot(
                utility_matrix,
                node.parent.R2,
                axes=((0, 1), (0, 1)),
            ) * (
                {**node.state.current_players, **node.state.all_in_players}["P2"][1]
                / (
                    config.all_in_dampener * config.blinds[0] * node.state.table.size
                    + 1
                )
            )

            node.v2 = (
                np.tensordot(
                    -1 * utility_matrix,
                    node.parent.R1,
                    axes=((0, 1), (0, 1)),
                )
                * {**node.state.current_players, **node.state.all_in_players}["P1"][1]
                / (
                    config.all_in_dampener * config.blinds[0] * node.state.table.size
                    + 1
                )
            )

        elif ( #if the node is a leaf node and not showdown, using meural network to calculate the evaluations
            3
            <= node.state.table.size
            <= 5
            and node.leaf_node
        ):
            node.v1, node.v2 = self.NN_manager.predict(
                node.R1, node.R2, node.state.table, node.state.pot
            )
        else: #if the node is not a leaf node calculate based on children
            v1_children = np.zeros((config.deck_size, config.deck_size))
            v2_children = np.zeros((config.deck_size, config.deck_size))
            if type(node) == Player_Node: #if the node is a player node use weighted strategic average of children
                for child in node.children:
                    self.evaluate_regret_strategize(child)
                    v1_children = (
                        v1_children
                        + child.v1
                        * np.transpose(node.strategy, (2, 0, 1))[
                            Resolver.get_action_index(child.action)
                        ]
                    )
                    v2_children = (
                        v2_children
                        + child.v2
                        * np.transpose(node.strategy, (2, 0, 1))[
                            Resolver.get_action_index(child.action)
                        ]
                    )
                node.v1 = v1_children
                node.v2 = v2_children

            else: #if the node is a chance node use avarage of children
                for child in node.children:
                    self.evaluate_regret_strategize(child)
                    v1_children = v1_children + child.v1
                    v2_children = v2_children + child.v2
                node.v1 = v1_children / len(node.children)
                node.v2 = v2_children / len(node.children)


        if type(node) == Player_Node and len(node.children) > 0: # updates strategies based on regret if node is a player node
            regret = np.zeros((5, config.deck_size, config.deck_size))
            for child in node.children:
                if node.player_name == "P1":
                    regret[Resolver.get_action_index(child.action)] = child.v1 - node.v1
                if node.player_name == "P2":
                    regret[Resolver.get_action_index(child.action)] = child.v2 - node.v2
                node.regret += regret.transpose(1, 2, 0)
            node.strategy = self.softmax(node.regret)

            legal_actions = Game_Manager.get_legal_actions(node.player_name, node.state)
            remove_index = np.zeros(5)
            for action in legal_actions: #removes illegal actions after softmax
                remove_index[Resolver.get_action_index(action)] = 1

            node.strategy = (
                node.strategy
                * remove_index
            )
            node.strategy = node.strategy / (np.sum(node.strategy, axis=2, keepdims=True) + 1e-10) #normalises the strategy
        return node.v1, node.v2 #returns v1 and v2 for the parent node

    def softmax(self, x): #softmax used to remove negative values and normalize the strategy, 
        e_x = np.exp(x - np.max(x, axis=-1, keepdims=True))
        return e_x / np.sum(e_x, axis=-1, keepdims=True)

    def fold_vector(self, node, action, state): #returns fold vector for the parent node when a player folds
        fold_player = action[0]
        vector = np.tril(np.full((config.deck_size, config.deck_size), 1), k=-1) * (
            (node.state.folded_players[fold_player][1] + 1)
            / (config.all_in_dampener * config.blinds[0] * node.state.table.size + 1)
        )  # np.tri./triu
        for card in state.table:
            vector[:, card] = 0
            vector[card, :] = 0

        if fold_player == "P1":
            v1 = -vector
            v2 = vector
            return v1, v2
        else:
            v1 = vector
            v2 = -vector
            return v1, v2

    def get_utility(self, table): #returns the utility matrix for the table
        if table.size == 0:
            raise ValueError("Table is empty and utility cannot be calculated")
        else:
            return self.oracle.utility_matrix(table)

    def get_action_index(action): #returns the index of the action
        return [
            "fold",
            "all in",
            "check",
            f"raise {config.blinds[1]}",
            f"raise {config.blinds[0]}",
        ].index(action)

    def generate_subtree(self, node): #generates the subtree based on the node, iterating until next card is dealt or the game ends
        from modules.game_manager import Game_Manager

        game_state = node.state.table.size

        if type(node) == Player_Node:
            legal_actions = Game_Manager.get_legal_actions(node.player_name, node.state)
            next_player = self.get_next_player(node.player_name)
            for action in legal_actions:
                child = Player_Node(next_player, action, node)
                child_actions = Game_Manager.get_legal_actions(
                    child.player_name, child.state
                )
                if len(child_actions) == 0:
                    # print(child.player_name, child.state.action_history)
                    child = Chance_Node(node, action)
                    if child.state.table.size == 5:
                        child.leaf_node = True
                node.children.append(child)
                self.generate_subtree(child)
        elif type(node) == Chance_Node and game_state < 5:

            if game_state == 0:
                card_amount = 3
            else:
                card_amount = 1
            for i in range(config.chance_nodes):
                child = Player_Node(
                    self.get_next_player(node.player_name),
                    node.action,
                    node,
                    node.state,
                )
                child.state = State_Manager.turn_card(
                    child.state,
                    self.oracle.pick_cards(node.state.table, card_amount),
                )
                child.leaf_node = True
                node.children.append(child)
                # if child.state.table.size < 5: #If we want to create the whole tree (which we dont because it is too big)
                #     self.generate_subtree(child)

    def get_next_player(self, player_name): #returns the next player
        if player_name == "P1":
            return "P2"
        else:
            return "P1"

    def get_type(self, table): #returns the type of the table
        if len(table) == 3:
            type = "flop"
        elif len(table) == 4:
            type = "turn"
        elif len(table) == 5:
            type = "river"
        else:
            raise ValueError("wrong number of cards in table")

        return type


if __name__ == "__main__": # for testing
    from modules.players import AI
    from modules.state_modules.state import State

    resolver = Resolver()
    p1 = AI("P1", 1000)
    root_state = State()
    root_state.current_players = {
        "AI 1": [config.start_stack, 0],
        "Player 1": [config.start_stack, 0],
    }
    root_state.action_history[-1].append(('P2', 'raise 20'))
    root = resolver.init_node(p1.R1, p1.R2, "AI 1", root_state)
    root.state.table = np.array([8, 7, 17,13])
    resolver.generate_subtree(root)
    
    for i in range (1):
        resolver.range_update(root)
        resolver.evaluate_regret_strategize(root)

    print(root.strategy[5][1])
    print(root.strategy[22][18])
