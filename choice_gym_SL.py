
import copy
import numpy as np
import pandas as pd
import sys

# If gympn is not installed in your environment:
# 1) git clone https://github.com/bpogroup/gympn.git
# 2) add your local path below, e.g. sys.path.append("C:/path/to/gympn")
# sys.path.append("C:/path/to/gympn")
sys.path.append("C:/Users/20183272/OneDrive - TU Eindhoven/Documents/GitHub/gympn")

import torch
from gympn.networks import HeteroActor
from torch_geometric.nn.conv.han_conv import HANConv
from torch_geometric.nn.aggr.basic import SumAggregation

# Add all necessary classes to safe globals
torch.serialization.add_safe_globals([HeteroActor, HANConv, SumAggregation])

from simpn.simulator import SimToken
from simpn.visualisation import Visualisation
from gympn.simulator import GymProblem
from simpn.reporters import SimpleReporter
from gympn.solvers import HeuristicSolver, GymSolver, RandomSolver
import random

#from policy_conformance import PolicyConformance
from policy_grammar import (
    IfThenElse, And, Compare, Count, Sum, Div, Number,
    Fire, Postpone, make_heuristic,
)

# ── Scenario parameters ───────────────────────────────────────────────────────
# Tune these to control how tight the system is and how often deliveries are late.

# Mean inter-arrival time for components (exponential distribution).
# Higher → fewer components available → more production waiting.
CHIP_ARRIVAL_SCALE       = 2.0   # chips abundant — not a bottleneck for either product
PHONE_CASE_ARRIVAL_SCALE = 3.5   # ρ_case = 3.5/5 = 0.7 → cases build in stock → rarely the bottleneck

# Time the production resource is held busy after an action fires.
# Phone: ρ_resource = 2/5 = 0.4 → M/D/1 analysis predicts ~5% queue-induced misses → SLA ≈ 0.95
# Game:  ρ_resource = 1/3 = 0.33 → resource usually free when focused on game
GAME_PRODUCTION_DELAY  = 1
PHONE_PRODUCTION_DELAY = 2

# Time between consecutive demand arrivals.
GAME_DEMAND_INTERVAL  = 3       # ~33 game demands per 100-unit episode
PHONE_DEMAND_INTERVAL = 5       # ~20 phone demands per 100-unit episode

# Deadline window: new demand is due current_clock + Uniform[LO, HI] time units.
#
# Phone: tight (2-5) — a production that must wait behind a queued demand will often
#        miss the lower end of the window, giving the target ~5% miss rate (SLA ≈ 0.95).
# Game:  wider (3-6) — fresh demands survive a brief phone-rescue focus switch,
#        targeting game SLA in the 0.5-0.8 range via the alternating-focus effect.
GAME_DEADLINE_LO,  GAME_DEADLINE_HI  = 3, 6
PHONE_DEADLINE_LO, PHONE_DEADLINE_HI = 2, 5

# ── Model setup ───────────────────────────────────────────────────────────────

# Set up the assembly system model
assembly_system = GymProblem(allow_postpone=True, causal_rl=False)

# Create a separate random number generator for this instance, seeded deterministically
# This ensures the same sequence across different GymProblem instances
assembly_system.rng = np.random.default_rng(42)  # Fixed seed for reproducibility

# Define the variables that make up the state-space of the system
arrival_chip = assembly_system.add_var("chip supply", var_attributes=["chip_id"])
arrival_phone_case = assembly_system.add_var("phone case supply", var_attributes=["phone_case_id"])

game_demand_arrival = assembly_system.add_var("game_demand_arrival", var_attributes=["request_date"])
phone_demand_arrival = assembly_system.add_var("phone_demand_arrival", var_attributes=["request_date"])
phone_demand = assembly_system.add_var("phone_demand", var_attributes=["request_date"])
game_demand = assembly_system.add_var("game_demand", var_attributes=["request_date"])

stock_chip = assembly_system.add_var("stock_chip", var_attributes=["chip_id"])
stock_phone_case = assembly_system.add_var("stock_phone_case", var_attributes=["phone_case_id"])

phone_resource = assembly_system.add_var("phone_resource", var_attributes=["phone_id"])
game_resource = assembly_system.add_var("game_resource", var_attributes=["game_id"])

game_delivered = assembly_system.add_var("game_delivered", var_attributes=["request_date", "on_time"])
phone_delivered = assembly_system.add_var("phone_delivered", var_attributes=["request_date", "on_time"])

assembly_system.set_unobservable(token_attrs={'stock_chip': ['chip_id'], 
                                              'stock_phone_case': ['phone_case_id'], 
                                              'phone_resource': ['phone_id'],
                                              'game_resource': ['game_id']}) 

# Define the events that cause arrivals
def chip_arrival(arrival):
    chip_id  = arrival["chip_id"] + 1
    delay    = np.random.default_rng(42 + chip_id).exponential(scale=CHIP_ARRIVAL_SCALE)
    new_arrival = {"chip_id": chip_id}
    return [SimToken(new_arrival, delay=delay), SimToken(new_arrival, delay=delay)]


def phone_case_arrival(arrival):
    phone_case_id = arrival["phone_case_id"] + 1
    delay         = np.random.default_rng(42 + phone_case_id).exponential(scale=PHONE_CASE_ARRIVAL_SCALE)
    new_arrival   = {"phone_case_id": phone_case_id}
    return [SimToken(new_arrival, delay=delay), SimToken(new_arrival, delay=delay)]


assembly_system.add_event([arrival_chip], [arrival_chip, stock_chip],
                          behavior=chip_arrival,
                          name="chip_arrival")

assembly_system.add_event([arrival_phone_case], [arrival_phone_case, stock_phone_case],
                          behavior=phone_case_arrival,
                          name="phone_case_arrival")

def game_demand_arrive(demand):
    new_demand = {"request_date": assembly_system.clock + random.randint(GAME_DEADLINE_LO, GAME_DEADLINE_HI)}
    return [SimToken(demand, delay=GAME_DEMAND_INTERVAL), SimToken(new_demand, delay=0)]

def phone_demand_arrive(demand):
    new_demand = {"request_date": assembly_system.clock + random.randint(PHONE_DEADLINE_LO, PHONE_DEADLINE_HI)}
    return [SimToken(demand, delay=PHONE_DEMAND_INTERVAL), SimToken(new_demand, delay=0)]

assembly_system.add_event([game_demand_arrival], [game_demand_arrival, game_demand],
                          behavior=game_demand_arrive,
                          name="game_demand_arrive")

assembly_system.add_event([phone_demand_arrival], [phone_demand_arrival, phone_demand],
                          behavior=phone_demand_arrive,
                          name="phone_demand_arrive")

def game_production_behavior(stock_chip, game_resource, game_demand):
    on_time = assembly_system.clock <= game_demand["request_date"]
    delivered = {"request_date": game_demand["request_date"], "on_time": on_time}
    return [SimToken(game_resource, delay=GAME_PRODUCTION_DELAY), SimToken(delivered, delay=0)]

def phone_production_behavior(stock_chip, stock_phone_case, phone_resource, phone_demand):
    on_time = assembly_system.clock <= phone_demand["request_date"]
    delivered = {"request_date": phone_demand["request_date"], "on_time": on_time}
    return [SimToken(phone_resource, delay=PHONE_PRODUCTION_DELAY), SimToken(delivered, delay=0)]

assembly_system.add_action([stock_chip, game_resource, game_demand], [game_resource, game_delivered],
                           behavior=game_production_behavior,
                           reward_function=lambda stock_chip, game_resource, game_demand: 1 if assembly_system.clock <= game_demand["request_date"] else 0,
                           name="game_production")

assembly_system.add_action([stock_chip, stock_phone_case, phone_resource, phone_demand], [phone_resource, phone_delivered],
                           behavior=phone_production_behavior,
                           reward_function=lambda stock_chip, stock_phone_case, phone_resource, phone_demand: 3 if assembly_system.clock <= phone_demand["request_date"] else 0,
                           name="phone_production")

# Describe the initial state of the system
arrival_chip.put({"chip_id": 0})
arrival_phone_case.put({"phone_case_id": 0})
game_demand_arrival.put({"request_date": 0})
phone_demand_arrival.put({"request_date": 0})
game_resource.put({"game_id": 1})
phone_resource.put({"phone_id": 1})

# ── Grammar-based policy trees ────────────────────────────────────────────────

def _sla_below(delivered_place: str, threshold: float) -> And:
    """Express  SLA(place) < threshold  using division.

    SLA = SUM(place, 'on_time', ALL) / COUNT(place, ALL)

    The outer And short-circuits on COUNT(place) == 0, preventing division
    by zero when no deliveries have occurred yet (treated as no violation).
    Python sums booleans as 0/1, so Sum(..., 'on_time') counts on-time
    deliveries correctly.
    """
    return And(
        Compare('>', Count(delivered_place), Number(0)),
        Compare('<',
            Div(Sum(delivered_place, 'on_time'), Count(delivered_place)),
            Number(threshold),
        ),
    )


# π_sla — SLA-driven policy
#   IF   SLA(phone_delivered) < 0.95       → FIRE phone_production  (rescue phone SLA)
#   ELIF SLA(game_delivered)  < 0.95       → FIRE game_production   (rescue game SLA)
#   ELIF game_demand > 0                   → FIRE game_production   (phone SLA OK → work on game)
#   ELSE                                   → FIRE phone_production  (fallback)
#
# Once phone SLA is ≥ 0.95, the policy switches attention to game production.
# make_heuristic falls back to 'postpone' if the chosen transition is not enabled.
_policy_tree_sla = IfThenElse(
    condition=_sla_below('phone_delivered', 0.95),
    then=Fire('phone_production'),
    else_=IfThenElse(
        condition=_sla_below('game_delivered', 0.95),
        then=Fire('game_production'),
        else_=IfThenElse(
            condition=Compare('>', Count('game_demand'), Number(0)),
            then=Fire('game_production'),
            else_=Fire('phone_production'),
        ),
    ),
)

sla_pattern = make_heuristic(_policy_tree_sla)

# After simulation, print some statistics
train = False
test = True
visualize = False
conformance_analysis = False

w_p = "data/train/best_policy.pth"

if train:
    print("Training the model...")
    assembly_system.training_run(length=100, args_dict={"open_tensorboard": True, "verbose": True})
    print("Model trained successfully")

def _episode_sla(pn, place_id: str) -> float:
    """Compute end-of-episode SLA for *place_id*: on_time / total, or 1.0 if empty."""
    for p in pn.places:
        if p._id == place_id:
            total = len(p.marking)
            if total == 0:
                return 1.0
            on_time = sum(1 for t in p.marking if t.value["on_time"])
            return on_time / total
    return 1.0

if test:
    frozen_pn = copy.deepcopy(assembly_system)

    episodes = 20

    print("Testing the SLA policy (pi_sla)...")
    rewards_sla      = []
    sla_phone_sla    = []
    sla_game_sla     = []
    for episode in range(episodes):
        assembly_system = copy.deepcopy(frozen_pn)
        sla_model = HeuristicSolver(heuristic_function=sla_pattern)
        rewards = assembly_system.testing_run(solver=sla_model, length=100, visualize=visualize)
        rewards_sla.append(rewards)
        sla_phone_sla.append(_episode_sla(assembly_system, 'phone_delivered'))
        sla_game_sla.append(_episode_sla(assembly_system, 'game_delivered'))

    print("SLA policy (pi_sla) rewards:        ", rewards_sla)
    print("SLA policy (pi_sla) phone SLA:       ", sla_phone_sla)
    print("SLA policy (pi_sla) game SLA:        ", sla_game_sla)
