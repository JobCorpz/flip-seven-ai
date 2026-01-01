from __future__ import annotations
import random
import csv
from typing import List

from flip7_engine import GameState
from mcts_agent import MCTSAgent


class RandomAgent:
    def choose_action(self, state: GameState) -> str:
        actions = state.legal_actions()
        if not actions:
            return 'stay'
        return random.choice(actions)


class HeuristicAgent:
    """Stays when current line score >= 15, else hits."""
    def choose_action(self, state: GameState) -> str:
        actions = state.legal_actions()
        if 'stay' not in actions and 'hit' not in actions:
            return 'stay'
        cur_sum = sum(state.current_numbers)
        if state.x2:
            cur_sum *= 2
        cur_sum += state.flat_modifiers
        if cur_sum >= 15:
            return 'stay'
        return 'hit'


def play_game(agents: List, seed: int = None) -> int:
    st = GameState(num_players=len(agents), seed=seed)
    while not st.game_over():
        cur = st.current_player
        agent = agents[cur]
        if isinstance(agent, MCTSAgent):
            action = agent.run(st, sims=agent.sims)
        else:
            action = agent.choose_action(st)
        st.apply_action(action)
    return st.winner


def run_experiment(sims_list=(10, 100, 1000), weights=(0, 10, 25, 50, 100), opponents=('random', 'heuristic'), games_per_setting=2, out_file='experiment_results.csv'):
    random.seed(0)
    with open(out_file, 'w', newline='') as f:
        writer = csv.writer(f)
        writer.writerow(['mcts_sims', 'flip7_weight', 'opponent', 'games', 'mcts_wins', 'opponent_wins'])
        for sims in sims_list:
            for w in weights:
                for opp in opponents:
                    mcts_wins = 0
                    opp_wins = 0
                    for g in range(games_per_setting):
                        # instantiate agents fresh each game
                        mcts = MCTSAgent(sims=sims, flip7_weight=w)
                        if opp == 'random':
                            opponent = RandomAgent()
                        else:
                            opponent = HeuristicAgent()
                        agents = [mcts, opponent]
                        winner = play_game(agents, seed=random.randrange(1_000_000))
                        if winner == 0:
                            mcts_wins += 1
                        elif winner == 1:
                            opp_wins += 1
                    writer.writerow([sims, w, opp, games_per_setting, mcts_wins, opp_wins])
                    print(f'sims={sims} weight={w} vs={opp} -> mcts_wins={mcts_wins} / {games_per_setting}')


if __name__ == '__main__':
    # quick smoke run (small games_per_setting)
    run_experiment(games_per_setting=2, out_file='experiment_results_smoke.csv')
