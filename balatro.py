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

    def __init__(self, num_simulations, num_decks=1, strategy=None, seed=None, verbose=False):
        # init
        self.num_simulations = num_simulations
        self.num_decks = num_decks
        self.strategy = strategy # How cards are chosen
        self.seed = seed
        self.verbose = verbose

        if self.seed and self.verbose:
            print(f"Using seed={self.seed}")
        self.rng = np.random.default_rng(seed=self.seed)

        # Deck setup
        self.deck = np.arange(52*self.num_decks) % 52
        self.deck_arr = np.asarray(self.deck)
        self.seen_cards = np.zeros((self.num_simulations, 52))
        self.rank_arr = self.deck_arr % self.NUM_RANKS
        self.suit_arr = self.deck_arr % self.NUM_SUITS

        # Default Balatro rules
        self.max_hand_size = 8
        self.max_num_hands = 4
        self.max_num_discards = 3
        self.chips_mult = { # (chip, mult)
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
        self.poker_hands = np.array(list(self.chips_mult.keys()))
        self.straight_references = np.array(
            [np.arange(1+i,i+6) for i in range(8)] + 
            [[9, 10, 11, 12, 0],[10, 11, 12, 0, 1]])

    
    def get_shuffled_deck(self):
        """
        Makes a matrix of shuffled decks per row

        Returns:
            shuffled_deck (np.array): Array of shuffled cards. Shape: (num_simulations, shuffled cards in self.deck)
        """
        shuffled_deck = np.array([self.rng.permutation(self.deck_arr) for _ in range(self.num_simulations)]) # Shape: (num_simulations, cards in deck)
        return shuffled_deck

    def simulate_one_round(self):
        """
        Simulates one round of Balatro
        """

        shuffled_deck = self.get_shuffled_deck()
        starting_hand_idx = np.tile(np.arange(self.max_hand_size), (self.num_simulations,1)) # ith row says which card index to grab per simulation. Shape: (num_simulations, self.max_hand_size)
        
        total_score = np.zeros(self.num_simulations)
        current_hand = shuffled_deck[np.arange(self.num_simulations)[:,None],starting_hand_idx]
        is_done = np.repeat(False, self.num_simulations)
        ith_hand_played = 0
        while ~np.all(is_done):
            if self.verbose:
                print(f"Hand {ith_hand_played}")
                print(self.print_hand(current_hand, sort=True))
            # 1. evaluate hand whether to play hand or discard
            game_state = self.hand_eval(current_hand)

            # 2. score or draw cards to restore hand
            total_score += game_state["score"]

            # Update for next loop
            is_done |= game_state["play_hand"]
            ith_hand_played += 1

        return total_score, game_state["target_poker_hand"]
    
    def print_hand(self, hand_matrix, sort=False):
        """
        Integer encoding to human readable str

        Args:
            hand_matrix (np.array): Array of int. Size: (num_simulations, self.max_hand_size)
        Returns:
            hands_str (np.array): Array of str. Size: (num_simulations, self.max_hand_size)
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
            current_hand (np.array): Array of cards in hand. Shape: (num_simulations, self.max_hand_size)
        Returns:
            game_state (dict): Dict of current game state. See return of self.decision.

        """
        # Slicing [:, :, None] compares each element with the arange
        rank_count = ((current_hand[:, :, None] % 13) == np.arange(self.NUM_RANKS)).sum(axis=1) # shape: (num_simulations, NUM_RANKS)
        suit_count = ((current_hand[:, :, None] % 4) == np.arange(self.NUM_SUITS)).sum(axis=1) # shape: (num_simulations, NUM_SUITS)
        self.seen_cards[np.arange(self.num_simulations)[:,None], current_hand] += 1

        current_poker_hand_matches = np.vstack([
            self.has_flush_five(card_count=self.seen_cards),
            self.has_flush_house(card_count=self.seen_cards),
            self.has_5_of_a_kind(rank_count=rank_count),
            self.has_straight_flush(rank_count=rank_count, suit_count=suit_count),
            self.has_4_of_a_kind(rank_count=rank_count),
            self.has_full_house(rank_count=rank_count),
            self.has_flush(suit_count=suit_count),
            self.has_straight(rank_count=rank_count),
            self.has_3_of_a_kind(rank_count=rank_count),
            self.has_two_pair(rank_count=rank_count),
            self.has_2_of_a_kind(rank_count=rank_count),
            self.has_high_card(rank_count=rank_count, suit_count=suit_count)
            ]).T
        game_state = self.decision(current_hand, current_poker_hand_matches)
        return game_state


    def get_score(self, hand, target_poker_hand):
        """
        Scores chosen hand and target_poker_hand

        Args:
            hand (np.array): Array of cards in hand. Shape: (num_simulations, 5)
            target_poker_hand (np.array): Array of poker hand str. Shape: (num_simulations,)
        returns:
            score_arr (np.array): Array of int score. Shape: (num_simulations,)
        """
        # TODO: Add processing to only use the 5 cards of the poker hand
        # if len(hand) != 5:
        #     raise ValueError("Received invalid number of cards. Expected 5 cards to score a hand.")
        hand_ranks = hand % 13
        hand_ranks_face_corrected = np.where(np.logical_or(hand % 13 == 0, hand % 13 > 10), 10 ,hand % 13) # J,Q,K are valued as 10
        hand_ranks_ace_corrected = np.where(hand_ranks_face_corrected == 1, 11, hand_ranks_face_corrected) # A are valued as 11
        hand_rank_sum = hand_ranks.sum(axis=1)
        chips, mult = np.array([self.chips_mult[poker_hand] for poker_hand in target_poker_hand]).T
        score_arr = (hand_rank_sum + chips) * mult
        return score_arr

    def decision(self, current_hand, current_poker_hand_matches):
        """
        Function that chooses how to handle cards

        Args:
            current_hand (np.array): Array of cards in hand. Shape: (num_simulations, self.max_hand_size)
            current_poker_hand_matches (np.array): Array of bool indicating whether poker hand is available. Shape: (num_simulations, len(self.poker_hands))
        Returns:
            game_state (dict): Dict of current gamestate. 
                - play_hand (np.array): Bool array if True, hand will play best poker hand available to be score. Otherwise, do not score. Shape: (num_simulations,)
                - score (np.array): Array of int score. Shape: (num_simulations,)
                - cards_kept (list): list of cards kept in hand per simulation. len = num_simulations
                - target_poker_hand (np.array): Array of poker hand str deemed as the best target to aim. Shape: (num_simulations,)
                - current_poker_hand_matches (np.array): Array of bool of current poker hand available. Shape: (num_simulations, len(self.poker_hands))
                - poker_hand_prob (np.array): Array of probabilities the next hand will be a given poker hand. Shape: (num_simulations, len(self.poker_hands))
        """
        strategy = self.strategy if self.strategy else "one_hand"
        valid_strategies = ["one_hand", "random", "greedy", "likelihood", "breadth", "depth", "neural_network"]
        game_state = {}
        
        if strategy == "one_hand":
            # Play only one hand to be scored.
            play_hand = np.repeat(True, self.num_simulations)
            cards_kept = current_hand
            target_poker_hand = self.poker_hands[current_poker_hand_matches.argmax(axis=1)]
            poker_hand_prob = current_poker_hand_matches.astype(float)

        elif strategy == "random":
            # No evaluation, randomly pick cards to discard
            play_hand = np.repeat(False, num_simulations)
            cards_kept = self.rng.choice(current_hand, size=(self.num_simulations, 5), replace=False, axis=1)
            target_poker_hand = self.poker_hands[current_poker_hand_matches.argmax(axis=1)]
            poker_hand_prob = np.zeros((self.num_simulations, len(self.poker_hands)))

        elif strategy == "greedy":
            # Evaluate which cards to keep based on greedy metrics
            # Keep cards for most complete poker hands
            pass

        elif strategy == "likelihood":
            # Evaluate which cards to keep based on the best expected score
            # This is gonna be hard to implement
            pass
        
        elif strategy == "breadth":
            # Evaluate by beadth first search
            #TODO: Think more about how this can be done
            pass

        elif strategy == "depth":
            # Evaluate by depth first search
            #TODO: Think more about how this can be done
            pass

        elif strategy == "neural_network":
            # Evaluation based on decisions from a neural network
            #TODO: Make an RL neural network
            pass

        else:
            raise ValueError(f"Unrecognized strategy. Expected {valid_strategies} but received strategy")

        score = self.get_score(current_hand, target_poker_hand) * play_hand.astype(int) # zero out score of unplayed hands

        game_state = {
            "play_hand": play_hand,
            "score": score,
            "cards_kept": cards_kept,
            "target_poker_hand": target_poker_hand,
            "current_poker_hand_matches": current_poker_hand_matches,
            "poker_hand_prob": poker_hand_prob,
        }
        return game_state

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

    def has_flush_five(self, card_count):
        """
        Checks if array of hands each has a flush five

        Args:
            card_count (np.array): Array of card counts. Shape: (num_simulations, 52)
        Returns:
            is_flush_five (np.array): Bool array if each simulation is a flush five. Shape: (num_simulations,)
        """

        is_flush_five = (card_count == 5).any(axis=1)
        return is_flush_five
        
    def has_flush_house(self, card_count):
        """
        Checks if array of hands each has a flush house

        Args:
            card_count (np.array): Array of card counts. Shape: (num_simulations, 52)
        Returns:
            is_flush_house (np.array): Bool array if each simulation is a flush house. Shape: (num_simulations,)
        """
        is_triple_flush = (card_count >= 3).any(axis=1)
        is_two_rank_duplicates = (card_count >= 2).sum(axis=1) >= 2 # has two ranks each with duplicates
        is_flush_house = is_triple_flush & is_two_rank_duplicates
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

    B = Balatro(1_000_000, seed=1234, verbose=True)
    scores, poker_hands = B.simulate_one_round()

    df = pd.DataFrame({"Poker_hand": poker_hands, "Score": scores})
    df.boxplot(column="Score",by="Poker_hand")
    plt.show()


if __name__ == "__main__":
    main()
