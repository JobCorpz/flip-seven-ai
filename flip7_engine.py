from __future__ import annotations
from dataclasses import dataclass
from typing import List, Optional, Any, Dict
import random
import copy


@dataclass(frozen=True)
class Card:
    kind: str  # 'number', 'modifier', 'action', 'mult'
    value: Optional[int] = None
    name: Optional[str] = None

    def __repr__(self):
        if self.kind == 'number':
            return f"N{self.value}"
        if self.kind == 'modifier':
            return f"+{self.value}"
        if self.kind == 'mult':
            return "x2"
        return f"{self.name}"


class Flip7Deck:
    """
    Deck for Flip 7 with custom distribution:
      - Number cards: 0 x1, 1 x1, 2 x2, 3 x3, ..., 12 x12
      - Modifier cards: +2 ... +10 (one each)
      - Action cards: Freeze x2, FlipThree x2, SecondChance x1, x2 multiplier x1
    Total cards: 94
    """

    def __init__(self, seed: Optional[int] = None):
        self.rng = random.Random(seed)
        self.cards: List[Card] = []
        # number cards
        # 0:1, 1:1, 2:2, ..., 12:12
        self.cards.append(Card('number', 0))
        self.cards.append(Card('number', 1))
        for n in range(2, 13):
            for _ in range(n):
                self.cards.append(Card('number', n))

        # modifiers +2..+10 (one each)
        for m in range(2, 11):
            self.cards.append(Card('modifier', m))

        # action and special cards to fill remaining counts (total action cards = 15)
        # We'll use: Freeze x2, FlipThree x2, SecondChance x1, x2 x1 -> 6
        # plus the 9 modifiers above -> 15 total action cards
        self.cards.extend([Card('action', name='Freeze') for _ in range(2)])
        self.cards.extend([Card('action', name='FlipThree') for _ in range(2)])
        self.cards.append(Card('action', name='SecondChance'))
        self.cards.append(Card('mult'))

        assert len(self.cards) == 94, f"Deck size mismatch: {len(self.cards)}"

    def shuffle(self, seed: Optional[int] = None):
        if seed is not None:
            self.rng.seed(seed)
        self.rng.shuffle(self.cards)

    def draw(self) -> Card:
        if not self.cards:
            raise IndexError("Drawing from empty deck")
        return self.cards.pop(0)

    def remaining(self) -> int:
        return len(self.cards)

    def copy(self) -> 'Flip7Deck':
        d = Flip7Deck()
        d.cards = self.cards.copy()
        d.rng = random.Random()
        return d

    def set_cards(self, cards: List[Card]):
        self.cards = list(cards)


class GameState:
    """
    Minimal Flip 7 game state to support MCTS determinization and simulations.

    - Tracks per-player total score.
    - Tracks the current player's ongoing 'line' (numbers collected, modifiers, x2, second chance).
    - Supports `clone()` and `set_deck()` for determinization.
    - Provides `legal_actions()` and `apply_action(action)` where action is 'hit' or 'stay'.
    """

    def __init__(self, num_players: int = 2, deck: Optional[Flip7Deck] = None, seed: Optional[int] = None):
        self.num_players = num_players
        self.player_totals: List[int] = [0 for _ in range(num_players)]
        self.current_player: int = 0
        self.deck = deck if deck is not None else Flip7Deck(seed)
        self.deck.shuffle(seed)

        # Current turn info (only for the active player)
        self.current_numbers: List[int] = []
        self.flat_modifiers: int = 0
        self.x2: bool = False
        self.second_chance_count: int = 0
        self.round_over: bool = False
        self.winner: Optional[int] = None

    def clone(self) -> 'GameState':
        return copy.deepcopy(self)

    def legal_actions(self) -> List[str]:
        if self.round_over or self.winner is not None:
            return []
        return ['hit', 'stay']

    def set_deck(self, cards: List[Card]):
        self.deck.set_cards(cards)

    def _sum_current_score(self) -> int:
        s = sum(self.current_numbers)
        if self.x2:
            s *= 2
        s += self.flat_modifiers
        if len(set(self.current_numbers)) >= 7:
            s += 15
        return s

    def _bank_current(self):
        score = 0
        if self.current_numbers:
            score = self._sum_current_score()
        self.player_totals[self.current_player] += score
        # reset current line
        self.current_numbers = []
        self.flat_modifiers = 0
        self.x2 = False
        self.second_chance_count = 0
        self.round_over = True
        if self.player_totals[self.current_player] >= 200:
            self.winner = self.current_player

    def apply_action(self, action: str) -> Dict[str, Any]:
        """
        Apply an action: 'hit' or 'stay'. Returns a result dict describing outcome.
        """
        if action not in self.legal_actions():
            raise ValueError(f"Illegal action: {action}")

        self.round_over = False

        if action == 'stay':
            self._bank_current()
            # advance player
            self.current_player = (self.current_player + 1) % self.num_players
            self.round_over = False
            return {'result': 'stayed', 'banked': True}

        # action == 'hit'
        card = self.deck.draw()
        outcome = self._process_draw(card)
        # if round ended due to bank or bust, advance player
        if outcome.get('round_end'):
            self.current_player = (self.current_player + 1) % self.num_players
            self.round_over = False
        return outcome

    def _process_draw(self, card: Card) -> Dict[str, Any]:
        # Number card
        if card.kind == 'number':
            v = card.value
            if v in self.current_numbers:
                # duplicate -> potential bust
                if self.second_chance_count > 0:
                    # discard the duplicate and consume second chance
                    self.second_chance_count -= 1
                    return {'result': 'duplicate_discarded', 'card': card}
                else:
                    # bust: score 0 for round, end turn
                    # reset current line (no points)
                    self.current_numbers = []
                    self.flat_modifiers = 0
                    self.x2 = False
                    self.second_chance_count = 0
                    self.round_over = True
                    return {'result': 'bust', 'card': card, 'round_end': True}
            else:
                self.current_numbers.append(v)
                # Flip7 bonus check
                if len(set(self.current_numbers)) >= 7:
                    # bank with +15 and end
                    score = self._sum_current_score()
                    self.player_totals[self.current_player] += score
                    self.current_numbers = []
                    self.flat_modifiers = 0
                    self.x2 = False
                    self.second_chance_count = 0
                    self.round_over = True
                    if self.player_totals[self.current_player] >= 200:
                        self.winner = self.current_player
                    return {'result': 'flip7', 'card': card, 'banked': score, 'round_end': True}
                return {'result': 'number_added', 'card': card}

        # Modifier card: add flat modifier
        if card.kind == 'modifier':
            self.flat_modifiers += card.value
            return {'result': 'modifier_added', 'card': card}

        # Multiplier x2
        if card.kind == 'mult':
            self.x2 = True
            return {'result': 'x2_added', 'card': card}

        # Other action cards
        if card.kind == 'action':
            name = card.name
            if name == 'Freeze':
                # forces a bank
                self._bank_current()
                self.round_over = True
                return {'result': 'freeze', 'card': card, 'round_end': True}
            if name == 'FlipThree':
                # force next three cards (or until bust)
                draws = []
                for _ in range(3):
                    if self.deck.remaining() == 0:
                        break
                    c = self.deck.draw()
                    draws.append(c)
                    res = self._process_draw(c)
                    # if a nested FlipThree triggers recursively, that behavior is allowed
                    if res.get('round_end'):
                        return {'result': 'flipthree_resolved', 'card': card, 'draws': draws, 'round_end': True}
                return {'result': 'flipthree_done', 'card': card, 'draws': draws}
            if name == 'SecondChance':
                self.second_chance_count += 1
                return {'result': 'secondchance_added', 'card': card}

        return {'result': 'unknown_card', 'card': card}

    def game_over(self) -> bool:
        return self.winner is not None

    def __repr__(self):
        return (
            f"GameState(player_totals={self.player_totals}, current_player={self.current_player}, "
            f"current_numbers={self.current_numbers}, flat_modifiers={self.flat_modifiers}, x2={self.x2}, "
            f"second_chance={self.second_chance_count}, deck_remaining={self.deck.remaining()})"
        )


if __name__ == '__main__':
    # quick smoke test
    gs = GameState(num_players=2, seed=42)
    print(gs)
    print('Top 5 deck:', gs.deck.cards[:5])