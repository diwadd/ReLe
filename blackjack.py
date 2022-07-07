import random
import itertools


class Card:
    def __init__(self, rank, suit):
        self.rank = rank
        self.suit = suit

    def __repr__(self) -> str:
        return f"{self.rank}-{self.suit}"
    
    def __str__(self) -> str:
        return self.__repr__()


class Deck:
    def __init__(self):
        self.cards = self._generate_cards()

    def _generate_cards(self):
        suits = ["c", "d", "h", "s"]
        ranks = [str(n) for n in range(2, 11)] + list("AJQK")

        return [Card(r, s) for r in ranks for s in suits]

    def __call__(self):
        return random.choice(self.cards)

class PolicyStickAt20Or21:
    def __init__(self):
        pass

    def decision(self, score):
        if score >= 20:
            return "stick"
        else:
            return "hit"

def generateStates():

    current_sum = [i for i in range(12, 22)]
    dealers_card = [i for i in range(2, 11)] + ["a"]
    usable_ace = [False, True]

    return list(itertools.product(*[current_sum, dealers_card, usable_ace]))


d = Deck()
print(d())

states = generateStates()
n_states = len(states)
print(f"Number of states: {n_states}")
print(states[0])

value_function = [0 for i in range(n_states)]