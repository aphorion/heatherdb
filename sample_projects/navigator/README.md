# Navigator — Robot Learning Through SDM

A robot learns obstacle avoidance by storing sensor→action experiences in HeatherDB's SDM. No neural network. No training loop. No reward function. Every experience is immediately available through associative reconstruction. **SDM as instant reinforcement learning.**

## The idea

Traditional RL: thousands of episodes, gradient descent, replay buffers, reward shaping, hyperparameter tuning. Hours to days of training.

SDM: every experience is one write. The robot gets better every single step. The basins of attraction ARE the learned policy. Drop it in a maze it's never seen — it navigates using "muscle memory" reconstructed from all past experiences.

## How it works

```
8 distance sensors (relative to heading)
    ↓
Encode as 32-dim vector [val, 1-val, val², (1-val)²] per sensor
    ↓
Place in action's partition of 128-dim space (4 actions × 32 dims)
    ↓
Write to HeatherDB (one-shot learning)

At recall time:
    ↓
For each candidate action:
  Encode sensors into that action's partition → query HeatherDB
  Measure reconstruction fidelity (cosine similarity)
    ↓
Pick action with HIGHEST fidelity → deepest basin of attraction
    ↓
Robot moves
```

The action is literally **reconstructed** from the sensor pattern. EAM's basins of attraction make similar sensor readings converge to the same action — the robot generalizes to novel situations automatically.

## Why SDM beats a vector DB here

- **Blending**: "wall ahead + wall right" (never seen together) reconstructs a blend of "wall ahead → turn left" and "wall right → turn left" responses
- **Noise tolerance**: real sensors are noisy — SDM reconstructs the clean action from degraded input
- **One-shot**: each experience is immediately available. No batch training needed.
- **Reinforcement**: successful maneuvers that happen often create stronger attractors — the robot develops "instincts"

## Quick start

```bash
# Start HeatherDB with dimension 128 (4 action partitions × 32 sensor dims)
cargo run --release -p heather_server -- \
  --data-dir /tmp/nav_db \
  --dimension 128 \
  --port 6380

# Setup
cd sample_projects/navigator
python -m venv .venv && source .venv/bin/activate
pip install -r requirements.txt

# Run
python run.py
```

### Commands

| Command | Description |
|---------|-------------|
| `train [steps]` | Train in maze A using heuristic (default: 500 steps) |
| `test [steps]` | Test in maze B (never seen) using only SDM (default: 200 steps) |
| `compare` | Compare SDM navigation vs random actions |
| `stats` | Show experience count and HeatherDB stats |
| `quit` | Exit |

### Example session

```
nav> train 500
  Training in maze A...
  500 experiences stored in SDM
  Steps: 500, Collisions: 23 (4.6% collision rate)

nav> test 200
  Testing in maze B (never seen)...
  Robot navigates using SDM muscle memory only.
  Steps: 200, Collisions: 31 (15.5% collision rate)

nav> compare
  Comparison over 200 steps in unseen maze:
    SDM navigation:  31 collisions (15.5%)
    Random actions:   89 collisions (44.5%)
    Improvement: 65% fewer collisions
```

## The mazes

**Maze A (training):** Complex layout with corridors and dead ends. Robot explores using a heuristic (avoid walls, occasional random turns) and stores every sensor→action pair.

**Maze B (testing):** Different layout, never seen during training. Robot navigates purely from EAM reconstruction. It generalizes because the sensor patterns (wall ahead, corridor to the right, etc.) activate the same attractors regardless of which maze they're in.

## Project structure

```
navigator/
  .env                  # HEATHER_URL
  requirements.txt      # httpx, python-dotenv, numpy (minimal!)
  navigator/
    __init__.py
    client.py           # HeatherDB HTTP client
    world.py            # 2D grid, robot, sensors, rendering
    brain.py            # SDM sensorimotor memory (encode/decode/learn/decide)
    cli.py              # Interactive terminal with visualization
  run.py                # Entry point
```

## SDM as instant RL

| | Traditional RL | SDM (Navigator) |
|---|---|---|
| **Training** | Thousands of episodes | One-shot per experience |
| **Learning loop** | Forward pass → loss → backward pass → update weights | Write vector. Done. |
| **Generalization** | Learned function approximation | Basins of attraction |
| **Novel situations** | Depends on training distribution | Blends similar experiences via superposition |
| **Infrastructure** | GPU, framework, hyperparameters | One HTTP call to HeatherDB |
