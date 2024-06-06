import numpy as np
import os
import sys
import time

parent_dir = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
sys.path.append(parent_dir)

from modules.oracle import Poker_Oracle
from modules.resolver_modules.resolver import Resolver

import config

class Player: #main Player class

    def __init__(self, name="Player X"):
        self.stack = config.start_stack
        self.stake = 0
        self.hand = np.array([])
        self.name = name

    def translate_cards(self,cards):
        suits = ["\u2660", "\u2665", "\u2666", "\u2663"]  # Spade Heart Diamond Club
        values = ["2", "3", "4", "5", "6", "7", "8", "9", "10", "J", "Q", "K", "A"][-config.type_size:]
        card_names = ""
        for card in cards:
            card_names += f"{suits[card%4]}{values[card//4]}"
        return card_names
    
class AI(Player): #AI class which can use both resolver and rollout based on selection

    def __init__(self, name="AI X", domain=1000):
        super().__init__(name)
        self.R1 = self.create_range()
        self.R2 = self.create_range()
        self.computer = Poker_Oracle()
        self.resolver = Resolver()
        self.domain = domain
        self.resolving = input(f"Do you want to use a resolver for {self.name}? (y/n)").lower() == "y"

    def create_range(self):
        range = np.full((config.deck_size,config.deck_size), 2 / config.deck_size**2)
        return range
    
    def select_action(self,state,legal_actions):
        if self.resolving:
            return self.select_action_resolve(state,legal_actions)
        else:
            return self.select_action_rollout(state,legal_actions)

    def select_action_resolve(self, state, legal_actions):
        print(f"{self.name} is tinking...")
        strategy, self.R1, self.R2 = self.resolver.resolve(
            state, self.R1, self.R2, self.name
        )
        
        probability = strategy[np.max(self.hand)][np.min(self.hand)]
        print(f"{self.name}´s current probability after resolving is {probability}")
        index = min(np.random.choice(np.arange(len(probability)), p=probability),len(legal_actions)-1)
        action = legal_actions[index]
        print(f"{self.name} wants to {action}")
        time.sleep(1)
        return action

    def select_action_rollout(self, state, legal_actions):
        print(f"{self.name} can select {legal_actions}")
        player_count = len(state.current_players.keys()) + len(state.all_in_players)
        hand_utility = self.computer.rollout(
            state.table, self.hand, player_count-1, self.domain
        )
        action="fold"
        if hand_utility > 1.5 * (1 / player_count) and "all in" in legal_actions:
            action= "all in"
        elif hand_utility > (1 / player_count) and "raise 20" in legal_actions:
            return "raise 20"
        elif hand_utility > 0.75 * (1 / player_count) and "raise 10" in legal_actions:
            action= "raise 10"
        elif (
            hand_utility > 0.3 * (1 / player_count)
            and state.current_bet - state.current_players[self.name][1] < 30
            and "check" in legal_actions
        ):
            action= "check"
        else:
            if state.current_bet - state.current_players[self.name][1] == 0:
                return "check"
            action = "fold"
        print(f"{self.name} wants to {action}")
        time.sleep(1)
        return action
        
class User(Player): #User class which can select actions based on input selection

    def __init__(self, name="User X"):
        super().__init__(name)

    def select_action(self, state, legal_actions):  # TODO
        print(f"{self.name}´s turn")
        print(state.table)
        print(
            f"\ntable {self.translate_cards(state.table)}\nhand  {self.translate_cards(self.hand)}\nthe pot is {state.pot}\nyour stack is {state.current_players[self.name][0]}\n and the current bet is {state.current_bet}\n "
        )
        if len(state.table)>0:
            print("current hand strength", Poker_Oracle.hand_strength(state.table, self.hand))
        if len(state.action_history[-1])>0:
            print(f"last action was {state.action_history[-1][-1][1]}")

        print("Please select an option:")
        for index, option in enumerate(legal_actions):
            print(f"{index + 1}. {option}")

        try:
            selection_index = int(
                input("Enter the number corresponding to your choice: ")
            )
        except Exception as e:
            print("Invalid input. Please enter a number.\n")
            self.select_action(state, legal_actions)

        try:
            if 0 < selection_index <= len(legal_actions):
                action = legal_actions[selection_index - 1]
                print(f"You selected: {action}")
            else:
                print(
                    "Invalid selection. Please enter a number within the provided range.\n"
                )
                return self.select_action(state, legal_actions)
        except ValueError:
            print("Invalid input. Please enter a number.\n")
            return self.select_action(state, legal_actions)

        return action