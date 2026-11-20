# Pacman Agent

A template for coding a pacman agent.

## Setting up the environment
1. Copy the code from this framework to create your Pacman Agent.
2. Clone your own pacman-agent repository (e.g., `git clone git@github.com:Username/pacman-agent.git`)
3. Go into pacman-agent folder, `cd pacman-agent/`
4. Run `git submodule update --init --remote` to pull the last pacman-contest
5. Create a virtual environment (e.g., `python3.8 -m venv venv`)
6. Activate the virtual environment with `source venv/bin/activate`
7. Go to the pacman-contest folder and install the requirements:
    - `cd pacman-contest/`
    - `pip install -r requirements.txt`
    - `pip install -e .`

## Coding a new agent
In the root folder do the following:
1. Create in `my_team.py` a class with the name of your agent that inherits from `CaptureAgent`, e.g. `class ReflexCaptureAgent(CaptureAgent):`
2. In the new class, override the `def choose_action(self, game_state):` function to return the best next action (check the given source code example).
3. (Optional) Add any other functions to the class for reasoning / learning and improving your agents decision which could also use other code sources in the same folder.

To debug the agent you can run `capture.py` between the `baseline_team` and your current agent:
1. `cd pacman-contest/src/contest/`
2. `python capture.py -r baseline_team -b ../../../my_team.py`



## Our agent (`my_team.py`)
All the code is in `my_team.py` (team of 2 agents, created by `create_team`):
- **OffensiveAgent**: raids the enemy half, banks food, escapes ghosts / uses capsules.
- **DefensiveAgent**: stays home, chases visible invaders, investigates eaten food, patrols the border.

Both use the same pipeline every turn:
1. **Strategy** – choose goal cells (food, home, capsule, invader...).
2. **A\* search** – shortest safe path to the closest goal (heuristic: maze distance, admissible).
3. **Linear evaluation** – `Q(s,a) = Σ wᵢ·fᵢ(s,a)` picks the final action (follow the plan, don't die, don't stop...).

Run against the baseline:
```
cd pacman-contest/src/contest/
python capture.py -r ../../../my_team.py -b baseline_team
```
