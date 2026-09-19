# Q-Signal: Hybrid Quantum–Classical Traffic Signal Optimization

> *A defensible hybrid architecture — not a quantum speedup claim.*

Q-Signal demonstrates how QUBO/QAOA can be integrated into a safety-constrained traffic-signal controller and evaluated fairly against classical baselines in a reproducible simulated network.

---

## Quick Start

```bash
# 1. Create and activate virtual environment
python -m venv .venv
.venv\Scripts\activate          # Windows
# source .venv/bin/activate     # Linux/macOS

# 2. Install dependencies
pip install -r requirements.txt

# 3. Launch the dashboard
streamlit run app.py

# 4. Run the test suite
pytest tests/ -v

# 5. Run the benchmark (3 seeds, 600s per run)
python experiments/run_benchmark.py --duration 600 --seeds 3 --verbose

# 6. Analyse results
python experiments/analyze_results.py
```

---

## Architecture

```
Traffic Demand Generator
        │
        ▼
  State Estimator / Event Manager
        │
   ┌────┴─────┐
   │          │
   ▼          ▼
Classical   QUBO Builder
Baseline         │
(Fixed /    ▼
Pressure)  QAOA / Quantum Simulator (Qiskit Aer)
                 │
         Candidate action + energy
                 │
     Safety & Feasibility Validator  ◄── always deterministic
                 │
    Emergency Override (NetworkX route)
                 │
     Simulator update → Metrics → Streamlit Dashboard
```

### Layers

| Layer | Module(s) | Technology |
|---|---|---|
| Network | `qsignal/network.py`, `qsignal/models.py` | NetworkX, Python dataclasses |
| Simulation | `qsignal/simulator.py`, `qsignal/demand.py` | Custom discrete-time |
| Control | `qsignal/controllers/` | Python modules |
| Quantum | `quantum/` | Qiskit Aer, (PennyLane optional) |
| Safety | `qsignal/safety.py` | Deterministic validator |
| Emergency | `qsignal/emergency.py` | NetworkX shortest path |
| Presentation | `app.py`, `qsignal/visualization.py` | Streamlit, Folium, Matplotlib |

---

## Project Structure

```
q-signal/
├── app.py                        # Streamlit dashboard
├── requirements.txt
├── config.yaml                   # All tunable parameters
├── README.md
├── qsignal/
│   ├── models.py                 # Dataclasses (no Streamlit)
│   ├── network.py                # Grid builder, route finding
│   ├── demand.py                 # Demand profiles, fractional accumulation
│   ├── simulator.py              # Discrete-time simulator
│   ├── events.py                 # Incident injection
│   ├── metrics.py                # Aggregation, improvement %
│   ├── safety.py                 # Deterministic validator
│   ├── emergency.py              # Green corridor manager
│   ├── visualization.py          # Folium + Matplotlib (no st.*)
│   └── controllers/
│       ├── fixed_time.py         # Fixed-cycle controller
│       ├── pressure.py           # Actuated pressure controller
│       └── hybrid_qaoa.py        # Unified solve_action() interface
├── quantum/
│   ├── qubo.py                   # QUBO matrix builder
│   ├── exact_solver.py           # Brute-force oracle (n ≤ 20)
│   ├── qiskit_solver.py          # Qiskit Aer QAOA
│   └── pennylane_solver.py       # PennyLane (optional)
├── experiments/
│   ├── run_benchmark.py
│   └── analyze_results.py
├── data/
│   ├── networks/
│   └── experiments/
└── tests/
    ├── test_simulator.py
    ├── test_qubo.py
    ├── test_safety.py
    └── test_emergency.py
```

---

## QUBO Formulation

Each intersection `i` has binary variable `x_i`:
- `x_i = 0` → North–South (NS) phase is green
- `x_i = 1` → East–West (EW) phase is green

Energy function minimized by QAOA:

```
E(x) =
  − Σ_i [benefit(i,NS)×(1−x_i) + benefit(i,EW)×x_i]   ← local demand
  + λ_coord × Σ_(i,j) (x_i + x_j − 2·x_i·x_j)         ← coordination
  + λ_switch × Σ_i switch_penalty(i) × x_i              ← switch cost
  + λ_fair   × Σ_i starvation_score(i)                  ← fairness
  + λ_spill  × Σ_i spillback_score(i)                   ← spillback
```

All `λ` weights are configurable in `config.yaml`. Safety constraints are **not** encoded in the QUBO; they are enforced deterministically by `qsignal/safety.py` **after** every solver call.

---

## Controllers

| Controller | Mode string | Description |
|---|---|---|
| Fixed-Time | `fixed` | Alternates phases on a fixed cycle |
| Pressure-Based | `pressure` | Serves the higher-pressure movement with hysteresis |
| Hybrid QAOA | `qaoa` | QUBO/QAOA → safety validation → deploy |

QAOA fallback order:
1. Qiskit Aer QAOA (primary)
2. Brute-force exact solver (n ≤ 20 variables)
3. Greedy diagonal assignment

---

## Safety Priority Order

```
Emergency reservation  >  Clearance interval  >  Min-green  >  Optimizer proposal  >  Fixed fallback
```

The safety validator runs on **every** controller output. It is never bypassed.

---

## Emergency Green Corridor

1. NetworkX `shortest_path()` finds the fastest route from origin to destination.
2. The first `advance_reservations` intersections ahead are phase-reserved.
3. Reserved phases cannot be overridden by the optimizer.
4. As the vehicle advances, reservations roll forward.
5. After the vehicle passes, reservations expire and normal optimization resumes.
6. Clearance buffer (default 15 s) prevents immediate restoration.

---

## Metrics

All metrics are collected every simulation second and exportable as CSV.

| Metric | Definition |
|---|---|
| Mean queue | Time-average of approach queues |
| Max queue | Maximum observed queue on any approach |
| Throughput | Vehicles reaching destination |
| Mean wait time | Vehicle-seconds waiting / completed vehicles |
| Phase switches | Signal changes (excl. yellow/all-red) |
| Est. fuel (L) | Idle vehicles × 0.00083 L/s *(model estimate)* |
| Est. CO₂ (kg) | Fuel × 2.31 kg/L *(model estimate)* |

> **Note:** Fuel and CO₂ are model-based estimates using documented factors.
> They are used only for **relative comparison** between controllers, not as
> absolute environmental measurements.

---

## Assumptions and Limitations

- The custom simulator uses a simplified departure model (one movement per intersection, no car-following or lane changing).
- Vehicle routing is simplified: vehicles are assigned a random destination and "complete" after 120 simulation seconds.
- QAOA runs on a **local classical simulator** (Qiskit Aer). This is not quantum hardware. The QUBO encoding and circuit execution are classical computations with quantum-circuit overhead.
- No quantum speedup is claimed. The contribution is the hybrid architecture, QUBO formulation, emergency corridor, and controlled experimental comparison.
- Emissions estimates use the IPCC/EPA default: 0.83 mL/s idle fuel consumption, 2.31 kg CO₂/L petrol.

---

## Honest Reporting

```
QAOA gap (%) = 100 × (QAOA energy − exact energy) / |exact energy|
```

The dashboard reports:
- QAOA energy per decision interval
- QAOA gap vs brute-force optimum
- Which solver was used (QAOA or fallback)
- All controller results under identical demand, seed, and incidents

---

## Configuration

All parameters are in `config.yaml`. Key values:

```yaml
signal:
  min_green_s: 10          # minimum green before phase switch
  yellow_s: 3              # yellow clearance
  all_red_s: 2             # all-red clearance

qubo:
  lambda_coord: 0.5        # coordination weight
  lambda_switch: 0.3       # switch penalty
  lambda_fair: 0.4         # fairness weight
  lambda_spill: 0.6        # spillback weight

qaoa:
  shots: 512
  depth: 2
  time_budget_s: 5.0
```

---

## References

- Qiskit documentation: https://docs.quantum.ibm.com/
- Qiskit Aer: https://qiskit.github.io/qiskit-aer/
- NetworkX: https://networkx.org/
- Streamlit: https://docs.streamlit.io/
- Folium: https://python-visualization.github.io/folium/
- OpenStreetMap: https://www.openstreetmap.org/copyright

---

*Q-Signal v1.0 — Built with Qiskit 2.5.2 · Streamlit 1.64.0 · NetworkX 3.4.2*
