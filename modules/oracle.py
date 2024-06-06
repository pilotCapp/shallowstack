import sys

# sys.path.append('/shallowstack/config.py')

import numpy as np
from phevaluator.evaluator import evaluate_cards

import sys
import os

# Get the parent directory of the current directory (i.e., the directory containing the main file)
parent_dir = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
# Add the parent directory to the Python path
sys.path.append(parent_dir)

import config


class Poker_Oracle:

    def create_deck(self):
        self.deck = np.arange(0, config.deck_size)

    def shuffle_deck(self):
        np.random.shuffle(self.deck)

    def pick_cards(self, table=np.array([]), amount=1): # picks the amount of cards designated based on the table
        self.create_deck()
        self.remove_cards(table)
        self.shuffle_deck()
        cards = np.array([], dtype=int)
        for i in range(amount):
            cards = np.append(cards, self.deck[-1])
            self.deck = np.delete(self.deck, -1)
        return cards

    def remove_cards(self, cards):
        for card in cards:
            self.deck = np.delete(self.deck, np.where(self.deck == card))

    def __init__(self):
        self.deck = np.array([])
        self.create_deck()
        self.shuffle_deck()

    @staticmethod
    def hand_strength_old(table, hand): #old hand strength function that I made myself, no longer compatible as new one inverts the valuse (lower is better)
        kicker = max((hand - 2) % 13 + 2) / 1000000
        cards = np.append(table, hand)
        cards.sort()
        if cards.size > config.deck_size:  ## TODO: Force config.deck_size cards
            raise ValueError("wrong number of cards")
        elif np.unique(cards.size) > cards.size:  # TODO: Force config.deck_size cards
            raise ValueError("Someone is cheating!")
        elif cards.max() > config.deck_size or cards.min() < 0:
            raise ValueError("Someone is joking!")

        types = (cards - 1) // 13
        types.sort()
        types = types[::-1]
        values = (cards - 2) % 13 + 2
        values.sort()
        values = values[::-1]

        unique_types = np.unique(types)
        unique_values = np.unique(values)

        for value in unique_values:
            if np.sum(values == value) > 4:
                return 0
                # raise ValueError("Someone is cheating!!")

        return max(
            Poker_Oracle.check_straight(values, types, kicker),
            Poker_Oracle.check_pairs(values, types, kicker),
            Poker_Oracle.check_flush(values, types, kicker),
        )

    @staticmethod
    def check_straight(values, types, kicker):
        old_card = 0
        old_type = 0
        card_streak = 1
        type_streak = 1
        start = values[0]

        for card, type in zip(values, types):
            if card == old_card - 1:
                card_streak += 1

                if type == old_type:
                    type_streak += 1
                else:
                    type_streak = 1
            elif card == old_card:
                pass
            else:
                card_streak = 1
                type_streak = 1
                start = card

            old_card = card

            if card_streak > 4:
                if type_streak > 4:
                    if start == 14:
                        return 10 + kicker
                    else:
                        return 9 + start / 100 + kicker
                else:
                    return 5 + start / 100 + kicker
        return 0

    @staticmethod
    def check_pairs(values, types, kicker):
        pair1 = 0, 0  # count, value
        pair2 = 0, 0  # count, value

        for value in np.unique(values):
            count = np.sum(values == value)
            if count > pair1[0]:
                if pair1[0] > pair2[0] or (
                    pair1[0] == pair2[0] and pair1[1] > pair2[1]
                ):
                    pair2 = pair1
                pair1 = count, value
            elif count > pair2[0]:
                pair2 = count, value
            elif count == pair2[0] and value > pair2[1]:
                pair2 = count, value

        if pair1[0] == 4:
            return 8 + pair1[1] / 100 + kicker
        elif pair1[0] == 3 and pair2[0] == 2:
            return 7 + pair1[1] / 100 + pair2[1] / 10000 + kicker
        elif pair1[0] == 2 and pair2[0] == 3:
            return 7 + pair2[1] / 100 + pair1[1] / 10000 + kicker
        elif pair1[0] == 3:
            return 4 + pair1[1] / 100 + kicker
        elif pair2[0] == 3:
            return 4 + pair2[1] / 100 + kicker
        elif pair1[0] == 2 & pair2[0] == 2:
            return (
                3
                + max(pair1[1], pair2[1]) / 100
                + min(pair1[1], pair2[1]) / 10000
                + kicker
            )
        elif pair1[0] == 2:
            return 2 + pair1[1] / 100 + kicker
        elif pair2[0] == 2:
            return 2 + pair2[1] / 100 + kicker
        else:
            return 1 + values[0] / 100 + kicker

    @staticmethod
    def check_flush(values, types, kicker):  # TODO fix?
        for type in types:
            if np.sum(types == type) > 4:
                return 6 + kicker
        return 0
    
    def rollout(self, table, hand, enemies, domain=config.rollout_instances): #rollout function which calculates win probability over the domain amount of instances
        wins = 0

        for k in range(domain):
            player_cards = np.zeros((enemies, 2))
            table = np.append(table, self.pick_cards(table, 5 - table.size))

            for j in range(enemies):
                player_cards[j] = self.pick_cards(table, 2)

            hand_strengths = np.zeros(enemies)
            for j in range(enemies):
                hand_strengths[j] = Poker_Oracle.hand_strength(table, player_cards[j])

            player_strength = Poker_Oracle.hand_strength(table, hand)

            if player_strength < hand_strengths.min():
                wins += 1
            elif player_strength == hand_strengths.min():
                wins += 1 / (1 + np.sum(hand_strengths == player_strength))

        return wins / domain

    def utility_matrix(self, table): #creates the utility matrix required by the resolver. It requires the high card first, therefore all arrays also need the hich card first
        utility_matrix = np.zeros(
            (config.deck_size, config.deck_size, config.deck_size, config.deck_size)
        )
        for i in range(config.deck_size):
            for k in range(0, i):
                hand1 = np.array([i, k])
                for j in range(config.deck_size):
                    for l in range(0, j):
                        hand2 = np.array([j, l])
                        if (
                            np.unique(np.concatenate((hand1, hand2, table))).size
                            == 4 + table.size
                        ):
                            player_strength1 = Poker_Oracle.hand_strength(table, hand1)
                            player_strength2 = Poker_Oracle.hand_strength(table, hand2)
                            utility_matrix[i, k, j, l] = (
                                1 if player_strength1 < player_strength2 else -1
                            )
        return utility_matrix

    @staticmethod
    def fn(k, i, j, l, table):
        return (
            1
            if Poker_Oracle.hand_strength(table, np.array([k + 2, i + 2]))
            < Poker_Oracle.hand_strength(table, np.array([j + 2, l + 2]))
            else -1
        )

    def utility_matrix_test(self, table):
        if table.size != 5:
            raise ValueError("wrong number of cards")
        utility = np.zeros(
            (config.deck_size, config.deck_size, config.deck_size, config.deck_size)
        )
        for k, i, j, l in np.ndindex(utility.shape):
            utility[k, i, j, l] = Poker_Oracle.fn(k, i, j, l, table)
        return utility

    def cheat_sheet( #creates a cheat sheet for the resolver to use based on the rollout and the possible hands
        self, table, enemies, domain
    ):  # TODO: Does the type here work? the type of the cards matter when table is set...
        sheet = np.zeros((config.type_size, config.type_size, 2))  # 0-12,0-12,0-1
        for k in range(0, config.type_size):
            for i in range(0, k + 1):
                for j in range(
                    0, 2
                ):  # 0 if they are same type, 1 if they are different
                    hand = np.array([k * config.type_size, i * config.type_size])
                    if j == 1:
                        type = min(
                            np.random.randint(1, 4), config.deck_size - 1
                        )  # randomly sets the type
                        hand[1] = hand[1] + type

                    sheet[k][i][j] = self.rollout(table, hand, enemies, domain)
        return sheet

    def hand_strength(table, hole): #new hand strength function that uses the phevaluator library, much more efficient
        np_cards = np.append(table, hole)
        cards = [int(card) for card in np_cards]
        hand_strength = evaluate_cards(*cards)
        return hand_strength


#test_oracle = Poker_Oracle()
#♠9♣10♥J♥Q♠A

#print(Poker_Oracle.hand_strength(np.array([0, 5, 10, 14, 20]), np.array([9, 1])))
#print(Poker_Oracle.hand_strength_old(np.array([0, 5, 10, 14, 20]), np.array([9, 1])))

# print(test_oracle.utility_matrix(np.array([2, 37, 48, 49, 15]))[0,1,2,3])
# sheet=test_oracle.cheat_sheet(np.array([2, 37, 48, 49, 15]), 1, 1000)

# print(sheet[1][0][0],sheet[1][0][1])

#print(test_oracle.pick_cards(np.array([29, 45]), 24))
