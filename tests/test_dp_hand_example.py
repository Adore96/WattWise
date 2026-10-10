"""
Hand example from the DP walkthrough: 1-unit battery, 3 slots
(off-peak 33, day 47, peak 106), 1 kWh load in the peak slot only,
no solar, eta = 1, no reserve, start empty.

Do nothing: buy 1 kWh at peak = 106.  Optimal: charge off-peak, hold,
discharge at peak = 33.
"""
import numpy as np
import pytest

from dp_solver import solve_dp
import model

HAND_DAY = {
    "solar": np.array([0.0, 0.0, 0.0]),
    "demand": np.array([0.0, 0.0, 1.0]),
    "price": np.array([33.0, 47.0, 106.0]),
}
HAND_PARAMS = dict(eta_c=1.0, eta_d=1.0, max_charge=1.0, max_discharge=1.0, reserve_frac=0.0)


def test_do_nothing_costs_106():
    assert model.baseline_bill(HAND_DAY) == pytest.approx(106.0)


def test_value_and_policy_tables_delta_1():
    res = solve_dp(HAND_DAY, capacity=1, delta=1.0, **HAND_PARAMS)
    # V[t] = [V_t(s=0), V_t(s=1)], computed by hand (t = 0 is slot 1)
    assert np.allclose(res["V"], [[33, 0], [47, 0], [106, 0], [0, 0]])
    # policy in steps: -1 charge, 0 hold, +1 discharge
    assert res["K"][0, 0] == -1   # slot 1, empty: charge at 33
    assert res["K"][1, 0] == -1   # slot 2, empty: charge at 47 beats buying at 106
    assert res["K"][1, 1] == 0    # slot 2, full: hold
    assert res["K"][2, 0] == 0    # slot 3, empty: nothing to discharge, buy
    assert res["K"][2, 1] == 1    # slot 3, full: discharge
    assert np.allclose(res["schedule"], [-1, 0, 1])
    assert res["bill"] == pytest.approx(33.0)


def test_finer_grid_delta_half_same_optimum():
    res = solve_dp(HAND_DAY, capacity=1, delta=0.5, **HAND_PARAMS)
    assert res["bill"] == pytest.approx(33.0)
    assert res["optimal_value"] == pytest.approx(33.0)
    assert np.allclose(res["schedule"], [-1, 0, 1])


def test_replay_matches_and_is_feasible():
    res = solve_dp(HAND_DAY, capacity=1, delta=0.5, **HAND_PARAMS)
    sim = model.simulate(res["schedule"], HAND_DAY, 1, **HAND_PARAMS)
    assert sim["feasible"]
    assert sim["bill"] == pytest.approx(res["bill"])
