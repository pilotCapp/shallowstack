import os
import sys
import numpy as np

from modules.state_modules.state import State

import config

class State_Manager:
    
    def __init__(self, game_manager):
        self.game_manager = game_manager

    def init_state():
        return State()

    def turn_card(state, cards):
        new_state = state.copy()
        new_state.table = np.append(new_state.table, cards)
        new_state.action_history.append([])
        return new_state

    ### Actions that generates new states based on action
    
    def bet(player_name, amount, state):
        if amount < 0:
            raise ValueError("Negative bet")
        new_state = state.copy()
        if (
            new_state.current_players[player_name][0] < amount
        ):  # current_players={[stack,stake]}
            raise ValueError("Not enough chips")

        new_state.current_players[player_name][0] -= amount
        new_state.pot += amount
        new_state.current_players[player_name][1] += amount
        new_state.current_bet = max(
            new_state.current_players[player_name][1], new_state.current_bet
        )

        return new_state

    def fold(player_name, state):
        new_state = state.copy()
        new_state.folded_players[player_name] = new_state.current_players[player_name]
        new_state.current_players.pop(player_name)
        return new_state

    def all_in(player_name, state):
        new_state = state.copy()
        new_state = State_Manager.bet(
            player_name, new_state.current_players[player_name][0], new_state
        )
        new_state.current_bet = max(
            new_state.current_players[player_name][1], new_state.current_bet
        )

        new_state.all_in_players[player_name] = new_state.current_players[player_name]
        new_state.current_players.pop(player_name)
        return new_state

    def process_action(player_name, action, state): #process action method which takes the action selection and processes it
        # action_history[-1].append((player_name, action))
        if action == "fold":
            new_state = State_Manager.fold(player_name, state)
        elif action == "all in":
            new_state = State_Manager.all_in(player_name, state)
        elif action == "check":
            amount = max(0, state.current_bet - state.current_players[player_name][1])
            if amount < 0:
                raise ValueError("Negative bet")
            if (
                state.current_players[player_name][0] == amount
            ):  # check if player has chips left, else all in
                new_state = State_Manager.all_in(player_name, state)
            else:
                new_state = State_Manager.bet(player_name, amount, state)
        elif action.startswith("raise"):
            bet = int(action.split(" ")[1])
            state.current_bet = state.current_bet + bet
            amount = max(0, state.current_bet - state.current_players[player_name][1])
            if (
                state.current_players[player_name][0] == amount
            ):  # check if player has chips left, else all in
                new_state = State_Manager.all_in(player_name, state)
            elif state.current_players[player_name][0] < amount:
                amount = state.current_players[player_name][0]
                new_state = State_Manager.bet(player_name, amount, state)
                #raise ValueError("Not enough chips") #TODO band-aid fix
            else:
                new_state = State_Manager.bet(player_name, amount, state)
        new_state.action_history[-1].append((player_name, action))
        return new_state