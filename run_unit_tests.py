
import numpy as np

from balatro import Balatro
import unittest


class TestBalatro(unittest.TestCase):
    SEED = 123

    def test_get_shuffled_deck(self):
        self.assertListEqual(observed_output.tolist(), expected_output)

    def test_simulate_one_round(self):
        self.assertListEqual(observed_output.tolist(), expected_output)

    def test_hand_eval(self):
        self.assertListEqual(observed_output.tolist(), expected_output)

    def test_get_score(self):
        self.assertListEqual(observed_output.tolist(), expected_output)

    def test_has_n_of_a_kind(self):

        expected_output = [False]

        rank_count = np.array([
            [0,0,0,0,0,0,0,0,0,0,0,0,0], # False
            [1,0,0,0,0,0,0,0,0,0,0,0,0], # False
            [1,1,0,0,0,0,0,0,0,0,0,0,0], # False
            [1,1,1,0,0,0,0,0,0,0,0,0,0], # False
            [1,1,1,1,0,0,0,0,0,0,0,0,0], # False
            [1,1,1,1,1,0,0,0,0,0,0,0,0], # False
            [1,1,1,1,1,1,0,0,0,0,0,0,0], # False
            [1,1,1,1,1,1,1,0,0,0,0,0,0], # False
            [1,1,1,1,1,1,1,1,0,0,0,0,0], # False
            [1,1,1,1,1,1,1,1,1,0,0,0,0], # False
            [1,1,1,1,1,1,1,1,1,1,0,0,0], # False
            [1,1,1,1,1,1,1,1,1,1,1,0,0], # False
            [1,1,1,1,1,1,1,1,1,1,1,1,0], # False
            [1,1,1,1,1,1,1,1,1,1,1,1,1], # False
            [2,0,0,0,0,0,0,0,0,0,0,0,0], 
            [3,0,0,0,0,0,0,0,0,0,0,0,0],
            [4,0,0,0,0,0,0,0,0,0,0,0,0],
            [5,0,0,0,0,0,0,0,0,0,0,0,0],
            [6,0,0,0,0,0,0,0,0,0,0,0,0],
            [7,0,0,0,0,0,0,0,0,0,0,0,0],
            [8,0,0,0,0,0,0,0,0,0,0,0,0],
            [9,0,0,0,0,0,0,0,0,0,0,0,0],
            [10,0,0,0,0,0,0,0,0,0,0,0,0],
            [11,0,0,0,0,0,0,0,0,0,0,0,0],
            [12,0,0,0,0,0,0,0,0,0,0,0,0],
            [13,0,0,0,0,0,0,0,0,0,0,0,0],
            [14,0,0,0,0,0,0,0,0,0,0,0,0],
        ])

        self.assertListEqual(observed_output.tolist(), expected_output)

    def test_has_n_of_a_kind_error1(self):
        with self.assertRaises(ValueError):
            Balatro(num_simulations=1).has_n_of_a_kind(rank_count=None, n=-1)


    def test_has_flush_five(self):
        self.assertListEqual(observed_output.tolist(), expected_output)

    def test_has_flush_house(self):
        self.assertListEqual(observed_output.tolist(), expected_output)

    def test_has_5_of_a_kind(self):

        expected_output = [
            False, False, False, False, False, False, False, False, False, False, False, False, False, False, False, False, False
            True, True, True, True, True, True, True, True, True, True, 
            ]

        rank_count = np.array([
            [0,0,0,0,0,0,0,0,0,0,0,0,0],  # False
            [1,0,0,0,0,0,0,0,0,0,0,0,0],  # False
            [1,1,0,0,0,0,0,0,0,0,0,0,0],  # False
            [1,1,1,0,0,0,0,0,0,0,0,0,0],  # False
            [1,1,1,1,0,0,0,0,0,0,0,0,0],  # False
            [1,1,1,1,1,0,0,0,0,0,0,0,0],  # False
            [1,1,1,1,1,1,0,0,0,0,0,0,0],  # False
            [1,1,1,1,1,1,1,0,0,0,0,0,0],  # False
            [1,1,1,1,1,1,1,1,0,0,0,0,0],  # False
            [1,1,1,1,1,1,1,1,1,0,0,0,0],  # False
            [1,1,1,1,1,1,1,1,1,1,0,0,0],  # False
            [1,1,1,1,1,1,1,1,1,1,1,0,0],  # False
            [1,1,1,1,1,1,1,1,1,1,1,1,0],  # False
            [1,1,1,1,1,1,1,1,1,1,1,1,1],  # False
            [2,0,0,0,0,0,0,0,0,0,0,0,0],  # False
            [3,0,0,0,0,0,0,0,0,0,0,0,0],  # False
            [4,0,0,0,0,0,0,0,0,0,0,0,0],  # False
            [5,0,0,0,0,0,0,0,0,0,0,0,0],  # True
            [6,0,0,0,0,0,0,0,0,0,0,0,0],  # True
            [7,0,0,0,0,0,0,0,0,0,0,0,0],  # True
            [8,0,0,0,0,0,0,0,0,0,0,0,0],  # True
            [9,0,0,0,0,0,0,0,0,0,0,0,0],  # True
            [10,0,0,0,0,0,0,0,0,0,0,0,0], # True
            [11,0,0,0,0,0,0,0,0,0,0,0,0], # True
            [12,0,0,0,0,0,0,0,0,0,0,0,0], # True
            [13,0,0,0,0,0,0,0,0,0,0,0,0], # True
            [14,0,0,0,0,0,0,0,0,0,0,0,0], # True
        ])

        observed_output = Balatro(num_simulations=1).has_5_of_a_kind(rank_count=rank_count)
        self.assertListEqual(observed_output.tolist(), expected_output)

    def test_has_straight_flush(self):

        expected_output = [
            False, False, False, False, False, False, False, False, False, False, False, False, False, False, False, False, 
            True, True, True, True, True, True, True, True, True, True, True, 
            ]

        rank_count = np.array([
            [0,0,0,0,0,0,0,0,0,0,0,0,0],  # False
            [1,0,0,0,0,0,0,0,0,0,0,0,0],  # False
            [1,1,0,0,0,0,0,0,0,0,0,0,0],  # False
            [1,1,1,0,0,0,0,0,0,0,0,0,0],  # False
            [1,1,1,1,0,0,0,0,0,0,0,0,0],  # False
            [1,1,1,1,1,0,0,0,0,0,0,0,0],  # False
            [1,1,1,1,1,1,0,0,0,0,0,0,0],  # False
            [1,1,1,1,1,1,1,0,0,0,0,0,0],  # False
            [1,1,1,1,1,1,1,1,0,0,0,0,0],  # False
            [1,1,1,1,1,1,1,1,1,0,0,0,0],  # False
            [1,1,1,1,1,1,1,1,1,1,0,0,0],  # False
            [1,1,1,1,1,1,1,1,1,1,1,0,0],  # False
            [1,1,1,1,1,1,1,1,1,1,1,1,0],  # False
            [1,1,1,1,1,1,1,1,1,1,1,1,1],  # False
            [2,0,0,0,0,0,0,0,0,0,0,0,0],  # False
            [3,0,0,0,0,0,0,0,0,0,0,0,0],  # False
            [4,0,0,0,0,0,0,0,0,0,0,0,0],  # True
            [5,0,0,0,0,0,0,0,0,0,0,0,0],  # True
            [6,0,0,0,0,0,0,0,0,0,0,0,0],  # True
            [7,0,0,0,0,0,0,0,0,0,0,0,0],  # True
            [8,0,0,0,0,0,0,0,0,0,0,0,0],  # True
            [9,0,0,0,0,0,0,0,0,0,0,0,0],  # True
            [10,0,0,0,0,0,0,0,0,0,0,0,0], # True
            [11,0,0,0,0,0,0,0,0,0,0,0,0], # True
            [12,0,0,0,0,0,0,0,0,0,0,0,0], # True
            [13,0,0,0,0,0,0,0,0,0,0,0,0], # True
            [14,0,0,0,0,0,0,0,0,0,0,0,0], # True
        ])
        self.assertListEqual(observed_output.tolist(), expected_output)

    def test_has_4_of_a_kind(self):

        expected_output = [
            False, False, False, False, False, False, False, False, False, False, False, False, False, False, False, False, 
            True, True, True, True, True, True, True, True, True, True, True, 
            ]

        rank_count = np.array([
            [0,0,0,0,0,0,0,0,0,0,0,0,0],  # False
            [1,0,0,0,0,0,0,0,0,0,0,0,0],  # False
            [1,1,0,0,0,0,0,0,0,0,0,0,0],  # False
            [1,1,1,0,0,0,0,0,0,0,0,0,0],  # False
            [1,1,1,1,0,0,0,0,0,0,0,0,0],  # False
            [1,1,1,1,1,0,0,0,0,0,0,0,0],  # False
            [1,1,1,1,1,1,0,0,0,0,0,0,0],  # False
            [1,1,1,1,1,1,1,0,0,0,0,0,0],  # False
            [1,1,1,1,1,1,1,1,0,0,0,0,0],  # False
            [1,1,1,1,1,1,1,1,1,0,0,0,0],  # False
            [1,1,1,1,1,1,1,1,1,1,0,0,0],  # False
            [1,1,1,1,1,1,1,1,1,1,1,0,0],  # False
            [1,1,1,1,1,1,1,1,1,1,1,1,0],  # False
            [1,1,1,1,1,1,1,1,1,1,1,1,1],  # False
            [2,0,0,0,0,0,0,0,0,0,0,0,0],  # False
            [3,0,0,0,0,0,0,0,0,0,0,0,0],  # False
            [4,0,0,0,0,0,0,0,0,0,0,0,0],  # True
            [5,0,0,0,0,0,0,0,0,0,0,0,0],  # True
            [6,0,0,0,0,0,0,0,0,0,0,0,0],  # True
            [7,0,0,0,0,0,0,0,0,0,0,0,0],  # True
            [8,0,0,0,0,0,0,0,0,0,0,0,0],  # True
            [9,0,0,0,0,0,0,0,0,0,0,0,0],  # True
            [10,0,0,0,0,0,0,0,0,0,0,0,0], # True
            [11,0,0,0,0,0,0,0,0,0,0,0,0], # True
            [12,0,0,0,0,0,0,0,0,0,0,0,0], # True
            [13,0,0,0,0,0,0,0,0,0,0,0,0], # True
            [14,0,0,0,0,0,0,0,0,0,0,0,0], # True
        ])

        observed_output = Balatro(num_simulations=1).has_4_of_a_kind(rank_count=rank_count)
        self.assertListEqual(observed_output.tolist(), expected_output)

    def test_has_full_house(self):
        self.assertListEqual(observed_output.tolist(), expected_output)

    def test_has_flush(self):
        self.assertListEqual(observed_output.tolist(), expected_output)

    def test_has_straight(self):

        expected_output = [True, True, True, True, True, True, True, True, True, True, False, False, False]

        rank_count = np.array([
            # Standard set of straights
            [0,1,1,1,1,1,0,0,0,0,0,0,0], # True: A,2,3,4,5 Straight
            [0,0,1,1,1,1,1,0,0,0,0,0,0], # True: 2,3,4,5,6 Straight
            [0,0,0,1,1,1,1,1,0,0,0,0,0], # True: 3,4,5,6,7 Straight
            [0,0,0,0,1,1,1,1,1,0,0,0,0], # True: 4,5,6,7,8 Straight
            [0,0,0,0,0,1,1,1,1,1,0,0,0], # True: 5,6,7,8,9 Straight
            [0,0,0,0,0,0,1,1,1,1,1,0,0], # True: 6,7,8,9,T Straight
            [0,0,0,0,0,0,0,1,1,1,1,1,0], # True: 7,8,9,T,J Straight
            [0,0,0,0,0,0,0,0,1,1,1,1,1], # True: 8,9,T,J,Q Straight
            [1,0,0,0,0,0,0,0,0,1,1,1,1], # True: 9,T,J,Q,K Straight
            [1,1,0,0,0,0,0,0,0,0,1,1,1], # True: T,J,Q,K,A Straight
            # Bad straights
            [1,1,1,1,1,0,0,0,0,0,0,0,0], # False: K,A,2,3,4 Wrap around Straight
            [0,1,1,1,1,0,0,0,0,0,0,0,0], # False: Not enough cards
            [1,0,1,1,1,0,1,0,0,0,0,0,0], # False: Holes in the straight
        ])

        observed_output = Balatro(num_simulations=1).has_straight(rank_count=rank_count)
        self.assertListEqual(observed_output.tolist(), expected_output)

    def test_has_3_of_a_kind(self):

        expected_output = [
            False, False, False, False, False, False, False, False, False, False, False, False, False, False, False, 
            True, True, True, True, True, True, True, True, True, True, True, True, 
            ]

        rank_count = np.array([
            [0,0,0,0,0,0,0,0,0,0,0,0,0],  # False
            [1,0,0,0,0,0,0,0,0,0,0,0,0],  # False
            [1,1,0,0,0,0,0,0,0,0,0,0,0],  # False
            [1,1,1,0,0,0,0,0,0,0,0,0,0],  # False
            [1,1,1,1,0,0,0,0,0,0,0,0,0],  # False
            [1,1,1,1,1,0,0,0,0,0,0,0,0],  # False
            [1,1,1,1,1,1,0,0,0,0,0,0,0],  # False
            [1,1,1,1,1,1,1,0,0,0,0,0,0],  # False
            [1,1,1,1,1,1,1,1,0,0,0,0,0],  # False
            [1,1,1,1,1,1,1,1,1,0,0,0,0],  # False
            [1,1,1,1,1,1,1,1,1,1,0,0,0],  # False
            [1,1,1,1,1,1,1,1,1,1,1,0,0],  # False
            [1,1,1,1,1,1,1,1,1,1,1,1,0],  # False
            [1,1,1,1,1,1,1,1,1,1,1,1,1],  # False
            [2,0,0,0,0,0,0,0,0,0,0,0,0],  # False
            [3,0,0,0,0,0,0,0,0,0,0,0,0],  # True
            [4,0,0,0,0,0,0,0,0,0,0,0,0],  # True
            [5,0,0,0,0,0,0,0,0,0,0,0,0],  # True
            [6,0,0,0,0,0,0,0,0,0,0,0,0],  # True
            [7,0,0,0,0,0,0,0,0,0,0,0,0],  # True
            [8,0,0,0,0,0,0,0,0,0,0,0,0],  # True
            [9,0,0,0,0,0,0,0,0,0,0,0,0],  # True
            [10,0,0,0,0,0,0,0,0,0,0,0,0], # True
            [11,0,0,0,0,0,0,0,0,0,0,0,0], # True
            [12,0,0,0,0,0,0,0,0,0,0,0,0], # True
            [13,0,0,0,0,0,0,0,0,0,0,0,0], # True
            [14,0,0,0,0,0,0,0,0,0,0,0,0], # True
        ])

        observed_output = Balatro(num_simulations=1).has_3_of_a_kind(rank_count=rank_count)
        self.assertListEqual(observed_output.tolist(), expected_output)

    def test_has_two_pair(self):
        self.assertListEqual(observed_output.tolist(), expected_output)

    def test_has_2_of_a_kind(self):

        expected_output = [
            False, False, False, False, False, False, False, False, False, False, False, False, False, False,
            True, True, True, True, True, True, True, True, True, True, True, True, True,
            ]

        rank_count = np.array([
            [0,0,0,0,0,0,0,0,0,0,0,0,0],  # False
            [1,0,0,0,0,0,0,0,0,0,0,0,0],  # False
            [1,1,0,0,0,0,0,0,0,0,0,0,0],  # False
            [1,1,1,0,0,0,0,0,0,0,0,0,0],  # False
            [1,1,1,1,0,0,0,0,0,0,0,0,0],  # False
            [1,1,1,1,1,0,0,0,0,0,0,0,0],  # False
            [1,1,1,1,1,1,0,0,0,0,0,0,0],  # False
            [1,1,1,1,1,1,1,0,0,0,0,0,0],  # False
            [1,1,1,1,1,1,1,1,0,0,0,0,0],  # False
            [1,1,1,1,1,1,1,1,1,0,0,0,0],  # False
            [1,1,1,1,1,1,1,1,1,1,0,0,0],  # False
            [1,1,1,1,1,1,1,1,1,1,1,0,0],  # False
            [1,1,1,1,1,1,1,1,1,1,1,1,0],  # False
            [1,1,1,1,1,1,1,1,1,1,1,1,1],  # False
            [2,0,0,0,0,0,0,0,0,0,0,0,0],  # True
            [3,0,0,0,0,0,0,0,0,0,0,0,0],  # True
            [4,0,0,0,0,0,0,0,0,0,0,0,0],  # True
            [5,0,0,0,0,0,0,0,0,0,0,0,0],  # True
            [6,0,0,0,0,0,0,0,0,0,0,0,0],  # True
            [7,0,0,0,0,0,0,0,0,0,0,0,0],  # True
            [8,0,0,0,0,0,0,0,0,0,0,0,0],  # True
            [9,0,0,0,0,0,0,0,0,0,0,0,0],  # True
            [10,0,0,0,0,0,0,0,0,0,0,0,0], # True
            [11,0,0,0,0,0,0,0,0,0,0,0,0], # True
            [12,0,0,0,0,0,0,0,0,0,0,0,0], # True
            [13,0,0,0,0,0,0,0,0,0,0,0,0], # True
            [14,0,0,0,0,0,0,0,0,0,0,0,0], # True
        ])

        observed_output = Balatro(num_simulations=1).has_2_of_a_kind(rank_count=rank_count)
        self.assertListEqual(observed_output.tolist(), expected_output)

    def test_has_high_card(self):

        expected_output = [True]

        observed_output = Balatro(num_simulations=1).has_high_card()
        self.assertListEqual(observed_output.tolist(), expected_output)

if __name__ == "__main__":
    unittest.main()