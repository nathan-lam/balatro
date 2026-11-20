from itertools import product, combinations
import numpy as np



class Generate_hands:
    NUM_RANKS = 13
    NUM_SUITS = 4
    ranks_13_choose_5 = list(combinations(range(NUM_RANKS),5))
    ranks_13_choose_2 = list(combinations(range(NUM_RANKS),2))
    ranks_13_choose_1 = list(combinations(range(NUM_RANKS),1))
    suits_4_choose_3 = list(combinations(range(NUM_SUITS),3))
    suits_4_choose_2 = list(combinations(range(NUM_SUITS),2))
    suits_4_choose_1 = list(combinations(range(NUM_SUITS),1))

    def __init__(self):
        pass

    def generate_straight_flush(self):
        """
        (10 choose 1) * (4 choose 1) = 40
        """
        deck = np.arange(52)

        # Suppose suit = card % 4 (0..3) and rank = card // 13 (0..12)
        # Build a (4, 13) grid where grid[s, r] is the card index for suit s and rank r
        grid = np.empty((4, 13), dtype=deck.dtype)
        suits = deck % 4
        ranks = deck // 13
        grid[suits, ranks] = deck

        # 1. Standard 5-card windows along the rank axis (shape: 4 suits x 9 straights x 5 cards)
        windows = np.lib.stride_tricks.sliding_window_view(grid, window_shape=5, axis=1)

        # 2. Optional: include Ace-low wrap-around (rank 12 followed by ranks 0, 1, 2, 3)
        wheel = np.column_stack([grid[:, 12], grid[:, :4]])[:, None, :]
        all_sf = np.concatenate([windows, wheel], axis=1).reshape(-1, 5)

    def generate_4_of_a_kind(self):
        """
        13 * (4 choose 4) = 13
        """
        output = np.array([[ith_rank + jth_suit * 13 for jth_suit in range(4)] for ith_rank in range(13)])
        return output
    

    def generate_full_houses(self):
        """
        13 * (4 choose 3) * 12 * (4 choose 2) = 3744 combinations
        """
        pass

    def generate_flushes(self):
        """
        (13 choose 5) * (4 choose 1) = 5148 combinations
        """
        pass

    def generate_straights(self):
        """
        (13 choose 5) * (4 choose 1)^5 = 329472 combinations
        """
        pass

    def generate_3_of_a_kind(self):
        """
        13 * (4 choose 3) = 52 combinations
        """
        pass

    def generate_two_pairs(self):
        """
        (13 choose 2) * (4 choose 2) * (4 choose 2) = 2808 combinations
        """ 
        output = np.empty((2808, 4))
        rank_combinations = self.ranks_13_choose_2
        suit_combinations = self.suits_4_choose_2
        for i, ((rank1, rank2), (first_suit1, first_suit2), (second_suits1, second_suits2)) in enumerate(product(rank_combinations, suit_combinations, suit_combinations)):
            output[i] = np.array([
                rank1 + 4*first_suit1, rank1 + 4*first_suit2,
                rank2 + 4*second_suits1, rank2 + 4*second_suits2,
            ], dtype=int)
        return output

    def generate_2_of_a_kind(self):
        """
        13 * (4 choose 2) = 78 combinations
        """
        output = np.empty((78, 2))
        rank_combinations = self.ranks_13_choose_1
        suit_combinations = self.suits_4_choose_2
        for i, ((rank,), (suit1, suit2)) in enumerate(product(rank_combinations, suit_combinations)):
            output[i] = np.array([rank + 4*suit1, rank + 4*suit2], dtype=int)
        return output

def main():
    GH = Generate_hands()
    print(GH.generate_two_pairs().tolist())
    print()

if __name__ == "__main__":
    main()