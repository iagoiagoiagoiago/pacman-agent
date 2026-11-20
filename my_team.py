# my_team.py
# ---------------------------------------------------------------------------
# Pacman Capture-the-Flag agent - AI course (UPF, 2026-27)
#
# HOW THIS AGENT THINKS (one sentence per layer)
# ---------------------------------------------------------------------------
#   1. STRATEGY  : each turn the agent decides *where* it wants to go
#                  (food, home, a capsule, an invader...)  -> a set of goal cells.
#   2. PLANNING  : A* search (heuristic search, ASTheory4) finds the shortest
#                  SAFE path to the closest goal. Cells too close to an enemy
#                  ghost are treated as blocked, so paths go around it.
#   3. DECISION  : every legal action is scored with a linear evaluation
#                  Q(s, a) = sum_i  w_i * f_i(s, a)
#                  (the same form used in approximate Q-learning / function
#                  approximation, ASTheory8-9, but with hand-tuned weights).
#                  The main feature is "does this action follow the A* plan?",
#                  the others are safety nets (don't die, don't stop...).
#
# TEAM: one OffensiveAgent (raids enemy food) + one DefensiveAgent (guards home).
#
# Why A*?  The heuristic we use is the true maze distance in the EMPTY maze
# (no enemies). It comes from a *relaxed problem* (we ignore the enemies), so
# it never overestimates the real cost -> it is ADMISSIBLE and CONSISTENT, and
# A* returns an optimal (shortest) safe path.
# ---------------------------------------------------------------------------

import random

import contest.util as util
from contest.capture_agents import CaptureAgent
from contest.game import Directions, Actions
from contest.util import nearest_point


#################
# Team creation #
#################

def create_team(first_index, second_index, is_red,
                first='OffensiveAgent', second='DefensiveAgent', num_training=0):
    """
    Called by the framework to build our team.
    Returns a list with two agents: one attacker and one defender.
    """
    agent_classes = {'OffensiveAgent': OffensiveAgent, 'DefensiveAgent': DefensiveAgent}
    return [agent_classes[first](first_index), agent_classes[second](second_index)]


#############################
# Tunable constants (knobs) #
#############################

DANGER_RADIUS = 5       # an enemy ghost this close (maze distance) means "danger"
SAFETY_RADIUS = 3       # A* paths never pass this close to a dangerous enemy
CARRY_LIMIT = 5         # go home to bank points after carrying this much food
TIME_MARGIN = 6         # extra moves kept as margin when racing home at the end
SCARED_SAFE_TIME = 5    # a ghost scared for more than this many moves is harmless
HUNT_RADIUS = 4         # attacker (while on home side) hunts invaders this close
GHOST_MEMORY = 10       # turns the attacker remembers a ghost it no longer sees
SIGHT_RANGE = 5         # enemies within 5 squares (Manhattan) are always visible
INFINITY = float('inf')
DEBUG = False           # True = let exceptions crash (useful while developing)


##############################################
# Base class: A* planner + linear evaluation #
##############################################

class PlannerAgent(CaptureAgent):
    """
    Common machinery used by both agents:
      - map pre-processing (home side, border cells, neighbour lists)
      - A* search with enemy-avoidance
      - linear feature evaluation to pick the final action

    Subclasses only have to implement:
      - decide(game_state)   -> (goals, threats, plan)
      - get_weights()        -> dict of feature weights
      - is_shelter(cell)     -> True if enemies cannot hurt us in that cell
    """

    def __init__(self, index, time_for_computing=.1):
        super().__init__(index, time_for_computing)
        self.start = None
        self.plan = []          # last A* plan (list of actions), useful for debugging
        self.mode = ''          # short text describing the current strategy

    # ------------------------------------------------------------------
    # Initialisation (we have 15 s here, so we pre-compute everything)
    # ------------------------------------------------------------------
    def register_initial_state(self, game_state):
        # The parent call also pre-computes all maze distances (self.distancer)
        super().register_initial_state(game_state)

        self.start = game_state.get_agent_position(self.index)
        self.walls = game_state.get_walls()
        self.width = self.walls.width
        self.height = self.walls.height

        # Red team owns the left half (x < width/2), blue the right half.
        middle = self.width // 2
        home_column = middle - 1 if self.red else middle

        # All free cells of the map, and those that belong to our half
        self.free_cells = [(x, y) for x in range(self.width) for y in range(self.height)
                           if not self.walls[x][y]]
        self.home_cells = [c for c in self.free_cells if self.is_home_side(c)]

        # Border cells: last column of our half. Reaching one = "being home".
        self.border = [(home_column, y) for y in range(self.height)
                       if not self.walls[home_column][y]]

        # Pre-compute the neighbours of every cell (successor function for A*)
        self.neighbours = {}
        for (x, y) in self.free_cells:
            self.neighbours[(x, y)] = []
            for action in [Directions.NORTH, Directions.SOUTH, Directions.EAST, Directions.WEST]:
                dx, dy = Actions.direction_to_vector(action)
                nx, ny = int(x + dx), int(y + dy)
                if not self.walls[nx][ny]:
                    self.neighbours[(x, y)].append((action, (nx, ny)))

    # ------------------------------------------------------------------
    # Small helpers
    # ------------------------------------------------------------------
    def is_home_side(self, cell):
        """True if the cell is on our half of the board."""
        return cell[0] < self.width // 2 if self.red else cell[0] >= self.width // 2

    def my_position(self, game_state):
        return nearest_point(game_state.get_agent_position(self.index))

    def distance(self, a, b):
        """Exact maze distance (pre-computed by the framework)."""
        return self.get_maze_distance(a, b)

    def visible_enemies(self, game_state):
        """
        Enemies we can currently see (within 5 squares of us or our teammate).
        Returns a list of (agent_state, position).
        """
        enemies = []
        for i in self.get_opponents(game_state):
            state = game_state.get_agent_state(i)
            pos = state.get_position()
            if pos is not None:
                enemies.append((state, nearest_point(pos)))
        return enemies

    def get_successor(self, game_state, action):
        """State after we take `action` (handles half-steps like the template)."""
        successor = game_state.generate_successor(self.index, action)
        pos = successor.get_agent_state(self.index).get_position()
        if pos != nearest_point(pos):
            return successor.generate_successor(self.index, action)
        return successor

    # ------------------------------------------------------------------
    # A* SEARCH  (ASTheory4 - heuristic search)
    # ------------------------------------------------------------------
    def a_star(self, start, goals, threats=(), allowed=None):
        """
        Finds the shortest SAFE path from `start` to the closest cell in `goals`.

        Search problem:
            state       = a cell (x, y)
            actions     = North / South / East / West (no walls)
            step cost   = 1
            goal test   = cell in goals
        Safety rule (threats = positions of dangerous enemies):
            a cell reached after g moves is UNSAFE if an enemy is within
            min(g, SAFETY_RADIUS) moves of it (see is_unsafe). Unsafe cells
            are simply not expanded, so the path goes around the enemy.
        allowed (optional) = function cell -> bool restricting where we may go
            (e.g. the defender is not allowed to leave its half).

        Heuristic h(cell) = min over goals of maze_distance(cell, goal).
            It is the exact cost of the RELAXED problem without enemies, so it
            is admissible and consistent -> A* graph search is optimal.

        Returns: list of actions ([] if already at a goal), or None if no safe
                 path exists.
        """
        goals = set(goals)
        if not goals:
            return None
        if start in goals:
            return []

        def heuristic(cell):
            return min(self.distance(cell, goal) for goal in goals)

        frontier = util.PriorityQueue()        # ordered by f = g + h
        frontier.push(start, heuristic(start))
        best_g = {start: 0}                    # cheapest known cost to each cell
        parent = {start: None}                 # cell -> (previous cell, action)
        closed = set()                         # already expanded cells

        while not frontier.is_empty():
            cell = frontier.pop()
            if cell in closed:
                continue                       # stale duplicate entry in the queue
            closed.add(cell)

            if cell in goals:                  # goal test when EXPANDING (optimality)
                return self._reconstruct_path(parent, cell)

            for action, next_cell in self.neighbours[cell]:
                if allowed is not None and not allowed(next_cell):
                    continue
                new_g = best_g[cell] + 1
                if self.is_unsafe(next_cell, new_g, threats):
                    continue
                if new_g < best_g.get(next_cell, INFINITY):
                    best_g[next_cell] = new_g
                    parent[next_cell] = (cell, action)
                    frontier.push(next_cell, new_g + heuristic(next_cell))

        return None                            # no safe path

    @staticmethod
    def _reconstruct_path(parent, cell):
        """Walks the parent pointers back from the goal to build the action list."""
        actions = []
        while parent[cell] is not None:
            previous, action = parent[cell]
            actions.append(action)
            cell = previous
        actions.reverse()
        return actions

    def is_unsafe(self, cell, steps, threats):
        """
        True if a threat could be on top of us (or right next to us) when we
        reach `cell`:  maze_distance(threat, cell) <= min(steps, SAFETY_RADIUS)

        - `steps` part : near cells are judged precisely (can it get there first?)
        - SAFETY_RADIUS: far cells only need to stay a few squares away from the
          threat. Requiring "the ghost can never get there first" for far cells
          would block almost the whole enemy half against a defender standing
          in the middle. We re-plan every turn, so if the ghost actually chases
          us the near cells become unsafe and the plan changes.
        """
        if self.is_shelter(cell):
            return False
        limit = min(steps, SAFETY_RADIUS)
        return any(self.distance(t, cell) <= limit for t in threats)

    def is_shelter(self, cell):
        """Overridden by subclasses: cells where threats cannot hurt us."""
        return False

    # ------------------------------------------------------------------
    # ACTION SELECTION: linear evaluation  Q(s,a) = w . f(s,a)
    # ------------------------------------------------------------------
    def choose_action(self, game_state):
        """
        Called every turn by the framework (must answer in < 1 s; in tests a
        turn takes ~1-15 ms). An exception would make us forfeit the game, so
        as a safety net any unexpected error falls back to a random legal move.
        Set DEBUG = True while developing to see errors instead.
        """
        if DEBUG:
            return self.choose_best_action(game_state)
        try:
            return self.choose_best_action(game_state)
        except Exception:
            return random.choice(game_state.get_legal_actions(self.index))

    def choose_best_action(self, game_state):
        """
        1) decide() picks goals and runs A*.
        2) Every legal action is scored with the linear evaluation.
        3) The best-scoring action is returned (random tie-break).
        """
        goals, threats, self.plan = self.decide(game_state)

        actions = game_state.get_legal_actions(self.index)
        weights = self.get_weights()
        best_value, best_actions = -INFINITY, []
        for action in actions:
            features = self.get_features(game_state, action, goals, threats)
            value = sum(weights.get(name, 0) * f for name, f in features.items())
            if value > best_value:
                best_value, best_actions = value, [action]
            elif value == best_value:
                best_actions.append(action)
        return random.choice(best_actions)

    def get_features(self, game_state, action, goals, threats):
        """
        Features f_i(s, a) describing what happens if we take `action`.
        All agents share the same features; each subclass weights them
        differently in get_weights().
        """
        features = util.Counter()
        my_pos = self.my_position(game_state)
        successor = self.get_successor(game_state, action)
        new_pos = self.my_position(successor)

        # 1. Does this action follow the A* plan?  (the main signal)
        features['on_plan'] = 1 if self.plan and action == self.plan[0] else 0

        # 2. How far is the closest goal after the move? (useful when there is no plan)
        if goals:
            features['goal_distance'] = min(self.distance(new_pos, g) for g in goals)

        # 3. Did we die? (when eaten we are sent back to our start position)
        if new_pos == self.start and self.distance(my_pos, self.start) > 1:
            features['died'] = 1

        # 4. Danger from threats (ignored if the new cell is a shelter)
        if threats and not self.is_shelter(new_pos):
            dists = [self.distance(new_pos, t) for t in threats]
            features['threat_adjacent'] = 1 if min(dists) <= 1 else 0
            features['threat_proximity'] = sum(max(0, DANGER_RADIUS - d) for d in dists)

        # 5. Leaving our half (only matters for the defender)
        features['leaves_home'] = 0 if self.is_home_side(new_pos) else 1

        # 6. Movement habits: standing still / turning back
        features['stop'] = 1 if action == Directions.STOP else 0
        current_direction = game_state.get_agent_state(self.index).configuration.direction
        features['reverse'] = 1 if action == Directions.REVERSE[current_direction] else 0

        return features

    # To be implemented by subclasses
    def decide(self, game_state):
        raise NotImplementedError

    def get_weights(self):
        raise NotImplementedError


##################
# Offensive agent #
##################

class OffensiveAgent(PlannerAgent):
    """
    Raids the enemy half, eats food and brings it back home.

    Strategy (ordered list of candidate goals; the first one with a safe A*
    path wins):
        - must go home  : few food left, or time running out while carrying
        - hunt          : we are a ghost at home and an invader is very close
        - capsule       : an enemy ghost is near and a power capsule exists
        - bank food     : carrying a lot, or carrying anything while in danger
        - eat food      : default behaviour
        - capsule/home  : fall-backs when food is unreachable
        - regroup       : move (inside our half) away from the ghosts, so the
                          next A* search can try another corridor
    Ghosts seen recently are remembered for a few turns (ghost_memory).
    """

    def __init__(self, index, time_for_computing=.1):
        super().__init__(index, time_for_computing)
        # enemy index -> (last seen position, turns since seen)
        self.ghost_memory = {}

    def is_shelter(self, cell):
        # Enemy ghosts cannot cross into our half as ghosts -> home is safe.
        return self.is_home_side(cell)

    def dangerous_ghosts(self, game_state):
        """
        Positions of enemy ghosts that can still eat us: the ones we see now,
        plus the ones we saw during the last GHOST_MEMORY turns.

        Why the memory? A defender camping at the border disappears from view
        as soon as we step back 6 squares. Without memory we would walk straight
        back into it; with memory A* keeps avoiding it and looks for another
        corridor into the enemy half.
        """
        my_pos = self.my_position(game_state)
        for i in self.get_opponents(game_state):
            state = game_state.get_agent_state(i)
            pos = state.get_position()
            harmless = state.is_pacman or state.scared_timer > SCARED_SAFE_TIME
            if pos is not None:                        # we see it right now
                if harmless:
                    self.ghost_memory.pop(i, None)
                else:
                    self.ghost_memory[i] = (nearest_point(pos), 0)
            elif i in self.ghost_memory:               # we remember it
                old_pos, age = self.ghost_memory[i]
                # Forget it if the memory is too old, it became a Pacman, or
                # its old cell is within sight (5 squares) and it is not there.
                if (age + 1 > GHOST_MEMORY or harmless or
                        util.manhattan_distance(my_pos, old_pos) <= SIGHT_RANGE):
                    del self.ghost_memory[i]
                else:
                    self.ghost_memory[i] = (old_pos, age + 1)
        return [pos for pos, age in self.ghost_memory.values()]

    def decide(self, game_state):
        my_state = game_state.get_agent_state(self.index)
        my_pos = self.my_position(game_state)
        carrying = my_state.num_carrying
        food = self.get_food(game_state).as_list()
        capsules = self.get_capsules(game_state)
        ghosts = self.dangerous_ghosts(game_state)

        # timeleft counts the moves of ALL 4 agents -> our remaining moves = /4
        moves_left = game_state.data.timeleft // 4
        dist_home = min(self.distance(my_pos, b) for b in self.border)
        nearest_ghost = min([self.distance(my_pos, g) for g in ghosts], default=INFINITY)
        in_danger = nearest_ghost <= DANGER_RADIUS

        # Invaders close to us (only relevant while we are a ghost at home)
        invaders = [pos for state, pos in self.visible_enemies(game_state)
                    if state.is_pacman and self.distance(my_pos, pos) <= HUNT_RADIUS]

        # Build the ordered list of (mode name, goal cells, allowed-cells function)
        candidates = []
        must_go_home = len(food) <= 2 or (carrying > 0 and moves_left <= dist_home + TIME_MARGIN)

        if must_go_home:
            candidates.append(('go home (end)', self.border, None))
        else:
            if not my_state.is_pacman and invaders and my_state.scared_timer == 0:
                candidates.append(('hunt invader', invaders, self.is_home_side))
            if in_danger and capsules:
                candidates.append(('capsule (escape)', capsules, None))
            if carrying >= CARRY_LIMIT or (in_danger and carrying > 0):
                candidates.append(('bank food', self.border, None))
            candidates.append(('eat food', food, None))
            if capsules:
                candidates.append(('capsule', capsules, None))
            if carrying > 0:
                candidates.append(('go home', self.border, None))

        # Food is blocked by a ghost: move (inside our half) to a cell far from
        # the ghosts (never where we already stand, so we do not just wait).
        # Next turn A* will try to enter the enemy half through another corridor.
        far_cells = [c for c in self.home_cells if c != my_pos and
                     all(self.distance(c, g) > DANGER_RADIUS for g in ghosts)]
        candidates.append(('regroup', far_cells or self.border, self.is_home_side))
        candidates.append(('go home', self.border, None))

        # Pick the first candidate for which A* finds a safe path
        for mode, goals, allowed in candidates:
            plan = self.a_star(my_pos, goals, ghosts, allowed)
            if plan is not None:
                self.mode = mode
                return goals, ghosts, plan

        # No safe path at all: let the evaluation function escape the ghosts
        self.mode = 'escape'
        return self.border, ghosts, None

    def get_weights(self):
        return {
            'on_plan': 100,           # follow A*
            'goal_distance': -2,      # otherwise get closer to the goal
            'died': -10000,           # never walk into a ghost
            'threat_adjacent': -1000, # never end next to a ghost
            'threat_proximity': -10,  # prefer to keep some distance
            'leaves_home': 0,         # the attacker is supposed to leave home
            'stop': -5,
            'reverse': -1,
        }


###################
# Defensive agent #
###################

class DefensiveAgent(PlannerAgent):
    """
    Stays on our half and protects our food.

    Strategy:
        - visible invader      : chase it with A* (if we are scared, keep
                                 distance 2 instead, so it cannot eat us)
        - food just disappeared: an invisible invader is there -> go check
        - otherwise            : patrol the border cell closest to the food
                                 that is most exposed (nearest to the border)
    """

    def __init__(self, index, time_for_computing=.1):
        super().__init__(index, time_for_computing)
        self.suspect_cells = []   # where we believe an unseen invader is

    def is_shelter(self, cell):
        return False              # invaders can hurt us anywhere when we are scared

    def eaten_food(self, game_state):
        """Food cells we were defending that disappeared since the last turn."""
        previous = self.get_previous_observation()
        if previous is None:
            return []
        before = set(self.get_food_you_are_defending(previous).as_list())
        now = set(self.get_food_you_are_defending(game_state).as_list())
        return list(before - now)

    def patrol_point(self, game_state):
        """Border cell closest to our most exposed food (the one nearest the border)."""
        food = self.get_food_you_are_defending(game_state).as_list()
        if not food:
            return random.choice(self.border)
        exposed = min(food, key=lambda f: min(self.distance(f, b) for b in self.border))
        return min(self.border, key=lambda b: self.distance(b, exposed))

    def decide(self, game_state):
        my_state = game_state.get_agent_state(self.index)
        my_pos = self.my_position(game_state)
        scared = my_state.scared_timer > 0
        invaders = [pos for state, pos in self.visible_enemies(game_state) if state.is_pacman]

        threats = []
        if invaders:
            self.suspect_cells = []
            if scared:
                # Scared ghosts die when touching a Pacman: shadow the closest
                # invader at distance 2, ready to strike when the timer ends.
                target = min(invaders, key=lambda p: self.distance(my_pos, p))
                goals = [c for c in self.home_cells if self.distance(c, target) == 2]
                threats = invaders
                self.mode = 'shadow invader (scared)'
            else:
                goals = invaders
                self.mode = 'chase invader'
        else:
            eaten = self.eaten_food(game_state)
            if eaten:
                self.suspect_cells = eaten
            if my_pos in self.suspect_cells:
                self.suspect_cells = []        # we checked, nobody here anymore
            if self.suspect_cells:
                goals = self.suspect_cells
                self.mode = 'investigate eaten food'
            else:
                goals = [self.patrol_point(game_state)]
                self.mode = 'patrol'

        plan = self.a_star(my_pos, goals, threats, allowed=self.is_home_side)
        return goals, threats, plan

    def get_weights(self):
        return {
            'on_plan': 100,
            'goal_distance': -2,
            'died': -10000,
            'threat_adjacent': -1000,
            'threat_proximity': -10,
            'leaves_home': -1000,     # the defender must stay a ghost on our half
            'stop': -1,               # standing still at the patrol point is fine
            'reverse': -1,
        }
