from __future__ import annotations
import random
import copy
from math import sqrt, log
from typing import Optional, Dict, Any, Tuple

from flip7_engine import GameState, Flip7Deck, Card
import sys
import csv


class Node:
    def __init__(self, state: GameState, parent: Optional['Node'] = None, action: Optional[str] = None):
        self.state = state
        self.parent = parent
        self.action = action
        self.children: Dict[str, Node] = {}
        self.visits = 0
        self.value = 0.0

    def ucb1(self, c_param: float = 1.4) -> float:
        if self.visits == 0:
            return float('inf')
        return self.value / self.visits + c_param * sqrt(log(self.parent.visits) / self.visits)


class MCTSAgent:
    def __init__(self, sims: int = 1000, flip7_weight: float = 50.0):
        self.sims = sims
        self.flip7_weight = flip7_weight

    def determinize(self, state: GameState) -> GameState:
        st = state.clone()
        # shuffle the unknown deck (determinization)
        random.shuffle(st.deck.cards)
        return st

    def default_policy(self, state: GameState) -> Tuple[float, Dict[str, Any]]:
        # random playout until round end or deck empty
        st = state.clone()
        player = st.current_player
        start_score = st.player_totals[player]
        flip7_hit = False
        while True:
            if st.game_over() or st.deck.remaining() == 0:
                break
            actions = st.legal_actions()
            if not actions:
                break
            act = random.choice(actions)
            res = st.apply_action(act)
            if res.get('result') == 'flip7':
                flip7_hit = True
            if res.get('round_end'):
                break
        end_score = st.player_totals[player]
        reward = end_score - start_score
        if flip7_hit:
            reward += self.flip7_weight
        return reward, {'flip7': flip7_hit}

    def expand(self, node: Node):
        actions = node.state.legal_actions()
        for a in actions:
            if a not in node.children:
                new_state = node.state.clone()
                # determinize per-child expansion
                random.shuffle(new_state.deck.cards)
                node.children[a] = Node(new_state, parent=node, action=a)

    def best_child(self, node: Node, c_param: float = 1.4) -> Node:
        choices = list(node.children.values())
        scores = [child.ucb1(c_param) for child in choices]
        return choices[scores.index(max(scores))]

    def backup(self, node: Node, reward: float):
        cur = node
        while cur is not None:
            cur.visits += 1
            cur.value += reward
            cur = cur.parent

    def tree_policy(self, root: Node) -> Node:
        cur = root
        while True:
            if not cur.children:
                return cur
            # all children visited? choose best by UCB1
            cur = self.best_child(cur)

    def run(self, root_state: GameState, sims: Optional[int] = None) -> str:
        if sims is None:
            sims = self.sims
        root = Node(self.determinize(root_state))
        # ensure children exist for hit/stay
        self.expand(root)

        for i in range(sims):
            # selection
            node = root
            # selection using UCB1 until a leaf
            while node.children:
                node = self.best_child(node)

            # expand
            if node.visits > 0:
                self.expand(node)
                if node.children:
                    node = random.choice(list(node.children.values()))

            # simulate from a determinized clone
            det_state = node.state.clone()
            random.shuffle(det_state.deck.cards)
            reward, info = self.default_policy(det_state)

            # backpropagate
            self.backup(node, reward)

        # choose best action (highest average value)
        best_act = None
        best_score = float('-inf')
        for a, child in root.children.items():
            if child.visits == 0:
                score = float('-inf')
            else:
                score = child.value / child.visits
            if score > best_score:
                best_score = score
                best_act = a
        return best_act


def simulate_vs_stay(state: GameState, sims_list=(10, 100, 1000), flip7_weight: float = 50.0):
    results = {}
    for sims in sims_list:
        hit_busts = 0
        hit_points = 0.0
        stay_busts = 0
        stay_points = 0.0
        for _ in range(sims):
            # Hit scenario
            st_hit = state.clone()
            random.shuffle(st_hit.deck.cards)
            try:
                res = st_hit.apply_action('hit')
            except IndexError:
                # empty deck
                continue
            reward, info = MCTSAgent(sims=0, flip7_weight=flip7_weight).default_policy(st_hit)
            if res.get('result') == 'bust':
                hit_busts += 1
            hit_points += reward

            # Stay scenario
            st_stay = state.clone()
            random.shuffle(st_stay.deck.cards)
            try:
                res2 = st_stay.apply_action('stay')
            except IndexError:
                continue
            reward2, info2 = MCTSAgent(sims=0, flip7_weight=flip7_weight).default_policy(st_stay)
            if res2.get('result') == 'bust':
                stay_busts += 1
            stay_points += reward2

        results[sims] = {
            'hit_bust_rate': hit_busts / max(1, sims),
            'stay_bust_rate': stay_busts / max(1, sims),
            'hit_avg_points': hit_points / max(1, sims),
            'stay_avg_points': stay_points / max(1, sims),
        }
    return results


if __name__ == '__main__':
    # quick smoke test and tuning harness
    gs = GameState(num_players=2, seed=123)
    print('Initial:', gs)

    # If '--tune' passed, run tuning over multiple flip7_weight values
    if '--tune' in sys.argv:
        weights = [0, 10, 25, 50, 100]
        sims_list = (10, 100, 1000)
        out_file = 'flip7_weight_tuning.csv'
        with open(out_file, 'w', newline='') as f:
            writer = csv.writer(f)
            writer.writerow(['weight', 'sims', 'hit_bust_rate', 'stay_bust_rate', 'hit_avg_points', 'stay_avg_points'])
            for w in weights:
                print(f'Running tuning for weight={w}')
                res = simulate_vs_stay(gs, sims_list=sims_list, flip7_weight=w)
                for s, metrics in res.items():
                    writer.writerow([w, s, metrics['hit_bust_rate'], metrics['stay_bust_rate'], metrics['hit_avg_points'], metrics['stay_avg_points']])
        print(f'Tuning results saved to {out_file}')
        sys.exit(0)

    agent = MCTSAgent(sims=100)
    action = agent.run(gs, sims=50)
    print('MCTS chose:', action)

    sim_res = simulate_vs_stay(gs, sims_list=(10, 100, 1000))
    for k, v in sim_res.items():
        print(f"Sims={k}: {v}")
