from itertools import combinations
import numpy as np

class Generate_hands:
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
        pass

    def generate_2_of_a_kind(self):
        """
        13 * (4 choose 2) = 78 combinations
        """
        output = np.empty((78, 2))
        for i, (value, suits) in enumerate(product(range(13), np.array(list(combinations(range(4),2))))):
            output[i] = value + 4*suits
        return output

