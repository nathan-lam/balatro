import eval7, pprint
import numpy as np
from scipy.special import comb, factorial
from scipy.stats import hypergeom
import matplotlib.pyplot as plt
import pandas as pd
import itertools
from copy import deepcopy
from collections import Counter

import time


class Balatro:

    NUM_RANKS = 13
    NUM_SUITS = 4
    RANKS = np.array(["K", "A", "2", "3", "4", "5", "6", "7", "8", "9", "T", "J", "Q"])

    SUITS = np.array(["C", "H", "S", "D"])

    def __init__(self, num_simulations, num_decks=1, seed=None, verbose=False):
        # init
        self.num_simulations = num_simulations
        self.num_decks = num_decks
        self.seed = seed
        self.verbose = verbose

        # Deck setup
        self.deck = np.arange(52*self.num_decks) % 52
        self.deck_arr = np.asarray(self.deck)
        self.rank_arr = self.deck_arr % self.NUM_RANKS
        self.suit_arr = self.deck_arr % self.NUM_SUITS

        # Default Balatro rules
        self.hand_size = 8
        self.num_hands = 4
        self.num_discards = 3
        self.chips = { # (chip, mult)
            "Flush Five": [160, 16],
            "Flush House": [140, 14],
            "Five of a Kind": [120, 12],
            "Straight Flush": [100, 8],
            "Four of a Kind": [60, 7],
            "Full House": [40, 4],
            "Flush": [35, 4],
            "Straight": [30, 4],
            "Three of a Kind": [30, 3],
            "Two Pair": [20, 2],
            "Pair": [10, 2],
            "High Card": [5, 1],
        }

        # Poker References
        self.straight_references = np.array(
            [np.arange(1+i,i+6) for i in range(8)] + 
            [[9, 10, 11, 12, 0],[10, 11, 12, 0, 1]])

    
    def get_shuffled_deck(self):
        """
        Makes a matrix of shuffled decks per row

        Returns:
            shuffled_deck (np.array): Array of shuffled cards. Shape: (num_simulations, shuffled cards in self.deck)
        """
        if self.seed and self.verbose:
            print(f"Using seed={self.seed}")
        rng = np.random.default_rng(seed=self.seed)
        shuffled_deck = np.array([rng.permutation(self.deck_arr) for _ in range(self.num_simulations)]) # Shape: (num_simulations, cards in deck)

        return shuffled_deck


    def simulate_one_round(self):
        
        starting_hand_idx = np.tile(np.arange(self.hand_size), (self.num_simulations,1)) # ith row says which card index to grab. Shape: (num_simulations, self.hand_size)
        shuffled_deck = self.get_shuffled_deck()
        for hand in range(self.num_hands):
            current_hand = shuffled_deck[np.arange(self.num_simulations)[:,None],starting_hand_idx]
            if self.verbose:
                print(f"Hand {hand}")
                print(self.print_hand(current_hand, sort=True))
            
            self.hand_eval(current_hand)
            
            pass

        score = self.get_score()

        return score
    
    def print_hand(self, hand_matrix, sort=False):
        """
        Integer encoding to human readable str

        Args:
            hand_matrix (np.array): Array of int. Size: (num_simulations, self.hand_size)
        Returns:
            hands_str (np.array): Array of str. Size: (num_simulations, self.hand_size)
        """
        idx_of_ranks_in_hand = hand_matrix % 13
        ranks_in_hand = self.RANKS[idx_of_ranks_in_hand]

        idx_of_suits_in_hand = hand_matrix % 4
        suits_in_hand = self.SUITS[idx_of_suits_in_hand]
        
        hands_str = np.char.add(ranks_in_hand, suits_in_hand)

        if sort:
            hands_str = np.sort(hands_str)

        return hands_str

    def hand_eval(self, current_hand, rank_count=None, suit_count=None):
        """
        Evaluate the hand

        Args:
            current_hand (np.array): Array of cards in hand. Shape: (num_simulations, self.hand_size)
        """
        # Slicing [:, :, None] compares each element with the arange
        rank_count = ((current_hand[:, :, None] % 13) == np.arange(self.NUM_RANKS)).sum(axis=1) # shape: (num_simulations, NUM_RANKS)
        suit_count = ((current_hand[:, :, None] % 4) == np.arange(self.NUM_SUITS)).sum(axis=1) # shape: (num_simulations, NUM_SUITS)
        
        return rank_count, suit_count


    def get_score(self, hand):
        pass


    def has_n_of_a_kind(self, rank_count, n):
        """
        Checks if array of hands each has a n-of-a-kind hand

        Args:
            rank_count (np.array): Array of rank counts. Shape: (num_simulations, NUM_RANKS)
            n (int): num of ranks to match
        Returns:
            is_n_of_a_kind (np.array): Bool array if each simulation is a n-of-a-kind. Shape: (num_simulations,)
        """
        if n < 1:
            raise ValueError(f"n must be a positive integer. Received {n}")
        is_n_of_a_kind = (rank_count >= n).any(axis=1)
        return is_n_of_a_kind

    def has_flush_five(self, rankd_count, suit_count):
        """
        Checks if array of hands each has a flush house

        Args:
            rank_count (np.array): Array of rank counts. Shape: (num_simulations, NUM_RANKS)
        Returns:
            is_flush_house (np.array): Bool array if each simulation is a flush house. Shape: (num_simulations,)
        """
        is_flush = self.has_flush(suit_count=suit_count)
        is_5_of_a_kind = self.has_5_of_a_kind(rank_count=rankd_count)
        is_flush_five = is_flush & is_5_of_a_kind
        return is_flush_five
        
    def has_flush_house(self, rank_count, suit_count):
        """
        Checks if array of hands each has a flush house

        Args:
            rank_count (np.array): Array of rank counts. Shape: (num_simulations, NUM_RANKS)
        Returns:
            is_flush_house (np.array): Bool array if each simulation is a flush house. Shape: (num_simulations,)
        """
        is_flush = self.has_flush(suit_count=suit_count)
        is_full_house = self.has_full_house(rank_count=rank_count)
        is_flush_house = is_flush & is_full_house
        return is_flush_house

    def has_5_of_a_kind(self, rank_count):
        """
        Checks if array of hands each has a 5-of-a-kind

        Args:
            rank_count (np.array): Array of rank counts. Shape: (num_simulations, NUM_RANKS)
        Returns:
            is_5_of_a_kind (np.array): Bool array if each simulation is a 5-of-a-kind. Shape: (num_simulations,)
        """
        is_5_of_a_kind = self.has_n_of_a_kind(rank_count=rank_count, n=5)
        return is_5_of_a_kind

    def has_straight_flush(self, rank_count, suit_count):
        """
        Checks if array of hands each has a straight flush

        Args:
            rank_count (np.array): Array of rank counts. Shape: (num_simulations, NUM_RANKS)
            suit_count (np.array): Array of suit counts. Shape: (num_simulations, NUM_SUITS)
        Returns:
            is_straight_flush (np.array): Bool array if each simulation is a high card. Shape: (num_simulations,)
        """
        is_flush = self.has_flush(suit_count=suit_count)
        is_straight = self.has_straight(rank_count=rank_count)
        is_straight_flush = is_flush & is_straight
        return is_straight_flush

    def has_4_of_a_kind(self, rank_count):
        """
        Checks if array of hands each has a 4-of-a-kind

        Args:
            rank_count (np.array): Array of rank counts. Shape: (num_simulations, NUM_RANKS)
        Returns:
            is_4_of_a_kind (np.array): Bool array if each simulation is a 4-of-a-kind. Shape: (num_simulations,)
        """
        is_4_of_a_kind = self.has_n_of_a_kind(rank_count=rank_count, n=4)
        return is_4_of_a_kind

    def has_full_house(self, rank_count):
        """
        Checks if array of hands each has a full house
        This relies on two pair returning true as a subset of a full house

        Args:
            rank_count (np.array): Array of rank counts. Shape: (num_simulations, NUM_RANKS)
        Returns:
            is_full_house (np.array): Bool array if each simulation is a full house. Shape: (num_simulations,)
        """
        is_3_of_a_kind = self.has_3_of_a_kind(rank_count=rank_count)
        is_two_pair = self.has_two_pair(rank_count=rank_count)
        is_full_house = is_3_of_a_kind & is_two_pair
        return is_full_house

    def has_flush(self, suit_count):
        """
        Checks if array of hands each has a flush

        Args:
            suit_count (np.array): Array of suit counts. Shape: (num_simulations, NUM_SUITS)
        Returns:
            is_flush (np.array): Bool array if each simulation is a flush. Shape: (num_simulations,)
        """
        is_flush = (suit_count >= 5).any(axis=1)
        return is_flush

    def has_straight(self, rank_count):
        """
        Checks if array of hands each has a straight
        Args:
            rank_count (np.array): Array of rank counts. Shape: (num_simulations, NUM_RANKS)
        Returns:
            is_straight (np.array): Bool array if each simulation is a straight. Shape: (num_simulations,)
        """
        rank_count_per_straight_ref = rank_count[:, self.straight_references] # Shape: (num_simulations, len(straight_references), poker_hand_size)
        is_straight_per_hand = (rank_count[:, self.straight_references] >= 1).all(axis=2) # Shape: (num_simulations, len(straight_references))
        is_straight = is_straight_per_hand.any(axis=1)
        return is_straight

    def has_3_of_a_kind(self, rank_count):
        """
        Checks if array of hands each has a 3-of-a-kind

        Args:
            rank_count (np.array): Array of rank counts. Shape: (num_simulations, NUM_RANKS)
        Returns:
            is_3_of_a_kind (np.array): Bool array if each simulation is a 3-of-a-kind. Shape: (num_simulations,)
        """
        is_3_of_a_kind = self.has_n_of_a_kind(rank_count=rank_count, n=3)
        return is_3_of_a_kind

    def has_two_pair(self, rank_count):
        """
        Checks if array of hands each has two pairs

        Args:
            rank_count (np.array): Array of rank counts. Shape: (num_simulations, NUM_RANKS)
        Returns:
            is_two_pair (np.array): Bool array if each simulation is a two_pair. Shape: (num_simulations,)
        """
        has_at_least_two_ranks = (rank_count >= 2).sum(axis=1) # Has at least 2 ranks. Shape: (num_simulations,)
        is_two_pair = has_at_least_two_ranks >= 2 # Has at least 2 ranks that have pairs
        return is_two_pair

    def has_2_of_a_kind(self, rank_count):
        """
        Checks if array of hands each has a 2-of-a-kind or a pair

        Args:
            rank_count (np.array): Array of rank counts. Shape: (num_simulations, NUM_RANKS)
        Returns:
            is_2_of_a_kind (np.array): Bool array if each simulation is a 2-of-a-kind. Shape: (num_simulations,)
        """
        is_2_of_a_kind = self.has_n_of_a_kind(rank_count=rank_count, n=2)
        return is_2_of_a_kind

    def has_high_card(self, rank_count=None, suit_count=None):
        """
        Checks if array of hands each has a high card
        #TODO: change to always return true? depends on the hierarchy of hands

        Args:
            rank_count (np.array): Array of rank counts. Shape: (num_simulations, NUM_RANKS)
        Returns:
            is_high_card (np.array): Bool array if each simulation is a high card. Shape: (num_simulations,)
        """    
        # at_most_1_count = (rank_count <= 1).all(axis=1)
        # is_not_flush = ~self.has_flush(suit_count=suit_count)
        # is_not_straight = ~self.has_straight(rank_count=rank_count)
        # is_high_card = at_most_1_count & is_not_straight
        is_high_card = np.tile(True, (self.num_simulations,)) # high card is a subset to all poker hands 
        return is_high_card

def main():

    B = Balatro(10, seed=123, verbose=True)
    B.simulate_one_round()

    print()


if __name__ == "__main__":
    main()
