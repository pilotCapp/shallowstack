import os
import sys
import numpy as np

parent_dir = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
sys.path.append(parent_dir)

import modules.state_modules.state_manager as state_manager
from modules.oracle import Poker_Oracle
from  modules.players import Player, User, AI

import config

class Game_Manager:

    def __init__(self, num_users, num_ai, blind=[20, 10], domain=config.resolving_iterations): #inits self creating players
        self.oracle = Poker_Oracle()
        self.state_manager = state_manager.State_Manager(self)
        self.domain = domain
        config.blinds = blind
        self.create_players(num_users, num_ai)

        self.state = state_manager.State_Manager.init_state()

    def create_players(self, num_users, num_ai): #creates players based on input

        self.players = [User(f"Player {k+1}") for k in range(num_users)]
        self.players += [AI(f"AI {i+1}", self.domain) for i in range(num_ai)]

        if len(self.players) < 2:
            raise ValueError("Not enough players")
        elif len(self.players) > 6:
            raise ValueError("Too many players")

    def rotate_blind(self): #rotates the blinds
        player = self.players.pop(0)
        self.players.append(player)

    def deal(self):  #deals the cards

        self.table = self.oracle.pick_cards(np.array([]), 5)
        self.hands = np.array([])
        for player in self.players:
            player.hand = self.oracle.pick_cards(np.append(self.table, self.hands), 2)
            self.hands = np.append(self.hands, player.hand)

    def start_game(self): #starts the game if there are more than 1 players and asks if the players want to play a new round

        while (
            len(self.players) > 1
            and input("Do you want to play a new round? (y/n)").lower() == "y"
        ):
            self.play_round()
        print("Game over")

    def play_round(self): #plays a round of poker
        print("New round")

        for player in self.players:
            if player.stack == 0:
                if (
                    input(
                        f"{player.name} is out of chips. Do you want to buy in? (y/n)"
                    ).lower()
                    == "y"
                ):
                    self.buy_in(player)
                else:
                    self.quit_game(player)

        self.rotate_blind()
        self.state.current_players = {
            player.name: [player.stack, 0] for player in self.players
        }  # TODO implement
        self.deal()

        if len(self.state.current_players) > 0:
            self.pre_flop()
        if len(self.state.current_players) > 0:
            self.flop()
        if len(self.state.current_players) > 0:
            self.turn()
        if len(self.state.current_players) > 0:
            self.river()
        self.showdown()

    def decision_round(self, player_names): #decision round where bets and all in requires new decision round
        print(f"the current players are {player_names}")
        # print(f"New decision round for {[player.name for player in players]}")
        for player_name in player_names:
            player = next(
                (player for player in self.players if player.name == player_name), None
            )

            legal_actions = Game_Manager.get_legal_actions(player_name, self.state)
            if len(legal_actions) == 0:
                break
            action = player.select_action(self.state, legal_actions)
            player_index = list(self.state.current_players.keys()).index(player_name)

            self.state = state_manager.State_Manager.process_action(player_name, action, self.state)
            print(f"{player_name} wants to {action}")

            if action == "all in":
                rotated_players = (
                    list(self.state.current_players.keys())[player_index:]
                    + list(self.state.current_players.keys())[:(player_index)]
                )
                print(f"{player_name} has gone all in")
                self.decision_round(rotated_players)

                break
            if player_name in self.state.current_players:
                print(
                    f"{player_name} has {self.state.current_players[player_name][1]} in the pot\n"
                )
            if action == f"raise {config.blinds[0]}" or action == f"raise {config.blinds[1]}":
                rotated_players = (
                    list(self.state.current_players.keys())[player_index + 1 :]
                    + list(self.state.current_players.keys())[:(player_index)]
                )
                print(rotated_players)

                new_current_players = {}
                new_current_players[player_name] = self.state.current_players[
                    player_name
                ]
                for player in rotated_players:
                    new_current_players[player] = self.state.current_players[player]
                self.state.current_players = new_current_players

                self.decision_round(rotated_players)

                break

    def pre_flop(self):
        self.state = state_manager.State_Manager.bet(
            list(self.state.current_players.keys())[0],
            config.blinds[0],
            self.state,
        )
        self.state = state_manager.State_Manager.bet(
            list(self.state.current_players.keys())[-1],
            config.blinds[1],
            self.state,
        )
        print(f"{list(self.state.current_players.keys())[0]} has the big blind")
        print(f"{list(self.state.current_players.keys())[-1]} has the small blind")

        self.decision_round(list(self.state.current_players.keys())[1:])

    def flop(self):
        self.state = state_manager.State_Manager.turn_card(self.state, self.table[:3])
        self.decision_round(list(self.state.current_players.keys()))

    def turn(self):
        self.state = state_manager.State_Manager.turn_card(
            self.state, np.array(self.table[3], dtype=int)
        )
        self.decision_round(list(self.state.current_players.keys()))

    def river(self):
        self.state = state_manager.State_Manager.turn_card(
            self.state, np.array(self.table[4], dtype=int)
        )
        self.decision_round(list(self.state.current_players.keys()))

    def showdown(self): #showdown selectinf the winners
        current_players = list(self.state.current_players.keys()) + list(
            self.state.all_in_players.keys()
        )
        print(f"the showdown players are {current_players}")
        strongest_hand = min(
            Poker_Oracle.hand_strength(self.table, player.hand)
            for player in self.players
            if player.name in current_players
        )
        winners = [
            player
            for player in self.players
            if Poker_Oracle.hand_strength(self.table, player.hand) == strongest_hand
            and player.name in current_players
        ]
        if len(winners) == 1:
            print(f"\n{winners[0].name} wins with a  strength of {strongest_hand}")
            print(f"\n {self.players[0].translate_cards(self.table)}")
            print(f" {self.players[0].translate_cards(winners[0].hand)}")

        else:
            print(
                f"\nthe winners are{[player.name for player in winners]} with a hand strength of {strongest_hand}"
            )
            print(f"\n {self.players[0].translate_cards(self.table)}")
            print(f" {[self.players[0].translate_cards(winner.hand) for winner in winners]}")
        self.end_round(winners)

    def end_round(self, winners): #ends the round and distributes the pot
        self.state.current_players = {
            **self.state.current_players,
            **self.state.all_in_players,
            **self.state.folded_players,
        }
        for winner in winners:
            self.state.current_players[winner.name][0] += self.state.pot / len(winners)

        for player in self.players:
            player.hand = np.array([])
            player.stack = self.state.current_players[player.name][0]

        self.state = state_manager.State_Manager.init_state()

    def get_legal_actions(player_name, state):  # returns the legal actions, if no legal actions new card must be dealt or showdown is in order
        actions = []
        if player_name not in list(state.current_players.keys()):
            # raise ValueError("Player is out of the action selection round")
            return actions
        if all(
            value[1] >= state.current_bet for value in state.current_players.values()
        ) and len(state.action_history[-1]) >= len(state.current_players):
            return actions
        if len(state.current_players) <= 1 and len(state.all_in_players) == 0:
            return actions

        actions.append("fold")

        if state.current_players[player_name][0] > 0:
            actions.append("all in")  # TODO implement player stack 0 handling
        if (
            state.current_players[player_name][0]
            > state.current_bet - state.current_players[player_name][1]
        ):
            actions.append("check")
        if state.current_players[player_name][0] > (
            state.current_bet - state.current_players[player_name][1] + config.blinds[1]
        ) and not any(
            f"raise {config.blinds[1]}" in action[1] for action in state.action_history[-1]
        ):
            #print(f"raise {config.blinds[1]}, {state.current_bet}, {state.current_players}")
            actions.append(f"raise {config.blinds[1]}")
        if state.current_players[player_name][0] > (
            state.current_bet - state.current_players[player_name][1] + config.blinds[0]
        ) and not any(
            f"raise {config.blinds[0]}" in action[1] for action in state.action_history[-1]
        ):
            #print(f"raise {config.blinds[0]}, {state.current_bet}, {state.current_players}")
            actions.append(f"raise {config.blinds[0]}")

        return actions

    def buy_in(self, player): #buys in the player if he is out of chips and user wants to
        player.stack = config.start_stack
        print(f"{player.name} has bought in")

    def quit_game(self, player): #quits the game if the player is out of chips and user does not want to buy in
        self.players.remove(player)