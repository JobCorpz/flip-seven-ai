# Flip7 AI: Monte Carlo Tree Search in a Stochastic Card Game

## Abstract

This repository presents an experimental implementation of **Monte Carlo Tree Search (MCTS)** applied to *Flip 7*, a stochastic, imperfect‑information card game. The project investigates how classical search-based decision algorithms behave under hidden information, high outcome variance, and reward shaping.

The primary objective is not optimal play, but to **analyze decision quality, risk sensitivity, and reward design** in an environment that departs from deterministic board games.

---

## Research Motivation

Flip 7 exhibits several properties that make it well suited for AI search experiments:

* Imperfect information due to an unknown deck order
* High variance outcomes caused by bust conditions and bonuses
* Non-linear reward structure (modifiers, multipliers, bonuses)
* Risk–reward trade-offs at every decision point (hit vs stay)

These properties expose limitations of naive MCTS designs and highlight the importance of reward formulation and rollout policy selection.

---

## System Overview

The system consists of three primary components:

1. A lightweight **game engine** modeling Flip 7 mechanics
2. A **determinized MCTS agent** for action selection
3. An **experiment harness** for controlled evaluation and parameter sweeps

```text
.
├── flip7_engine.py        # Game rules and state transitions
├── mcts_agent.py          # Monte Carlo Tree Search implementation
├── experiment_harness.py  # Benchmarking and evaluation
├── experiment_results.csv # Generated experimental data
└── README.md
```

---

## Game Engine

The Flip 7 engine implements a minimal but complete rule set required for Monte Carlo simulation.

### Deck Specification

* Number cards: 0×1, 1×1, 2×2, …, 12×12
* Modifier cards: +2 through +10 (one each)
* Action cards: Freeze ×2, FlipThree ×2, SecondChance ×1
* Multiplier cards: ×2 ×1

Total deck size: **94 cards**

### Scoring Model

* Sum of unique number cards collected in the current turn
* Optional ×2 multiplier applied to the sum
* Flat modifiers added after multiplication
* Flip7 bonus: +15 points for collecting seven unique numbers
* Bust occurs when a duplicate number is drawn without an available SecondChance

A game terminates when a player's total score reaches or exceeds **200 points**.

---

## Monte Carlo Tree Search Agent

The agent applies **Monte Carlo Tree Search with determinization** to address hidden information.

At each decision point, unknown deck orderings are sampled to produce fully observable game states, over which standard MCTS is executed.

### Algorithmic Pipeline

1. Determinization by shuffling unknown deck cards
2. Tree traversal using UCB1 for action selection
3. Expansion on previously unvisited actions
4. Simulation via stochastic rollouts to the end of the current round
5. Backpropagation of observed rewards

Only two actions are available at each node: `hit` and `stay`.

---

## Reward Design

The reward signal is defined as the **marginal score gain for the acting player over the course of a single turn**:

```python
reward = end_score - start_score
```

To study risk sensitivity, an auxiliary incentive is introduced for achieving a Flip7 event:

```python
if flip7_hit:
    reward += flip7_weight
```

The Flip7 weight parameter allows controlled adjustment of aggressive versus conservative behavior, enabling systematic evaluation of reward shaping effects.

---

## Baseline Policies

Two baseline opponents are implemented for comparative evaluation.

### Random Policy

Selects uniformly among legal actions. This policy exhibits high variance and serves as a lower-bound benchmark.

### Heuristic Policy

Estimates current turn value and elects to stay once a predefined threshold is reached. This policy prioritizes score stability and low bust probability.

---

## Experimental Framework

The experiment harness automates match execution and aggregates results across varying configurations.

### Tunable Parameters

* Number of MCTS simulations per action
* Flip7 reward weight
* Opponent policy
* Number of games per configuration

### Metrics Collected

* Total wins per agent
* Win rate by simulation budget
* Win rate by reward weight

Results are exported to CSV format to facilitate external statistical analysis.

---

## Empirical Observations

Preliminary experiments reveal several consistent trends:

* MCTS significantly outperforms purely random play
* Conservative heuristic policies often outperform naive MCTS
* Increasing simulation count does not monotonically improve performance
* Overweighting rare bonus events increases bust frequency

These findings underscore the sensitivity of MCTS to rollout strategy and reward formulation in stochastic domains.

---

## Limitations

This implementation intentionally omits several advanced components:

* Opponent-aware or adversarial reward modeling
* Minimax-style value propagation
* Information Set MCTS (ISMCTS)
* Learned or heuristic-guided rollout policies

As a result, the agent optimizes short-term expected value rather than long-horizon competitive advantage.

---

## Future Work

Potential extensions include:

* Relative reward functions incorporating opponent score
* Explicit bust penalties and risk regularization
* ISMCTS with information set abstraction
* Policy-guided rollouts via supervised or reinforcement learning
* Formal statistical evaluation of convergence behavior

---

## Reproducibility

To run the experimental benchmark:

```bash
python experiment_harness.py
```

To perform Flip7 reward tuning:

```bash
python mcts_agent.py --tune
```

---

## Conclusion

This repository serves as a focused case study in applying MCTS to stochastic, imperfect-information environments. The results highlight both the strengths and limitations of classical search methods when deployed outside deterministic, fully observable domains.

The project is intended as a foundation for further research rather than a final competitive agent.
