# 🚦 Q-Signal: Quick Access Guide

Welcome to **Q-Signal: Hybrid Quantum–Classical Traffic Signal Optimization**.

---

## ⚡ 1-Click Easy Launch (Windows)

Anyone can run the system immediately by double-clicking the following files:

| File | What it does |
| :--- | :--- |
| **`run_dashboard.bat`** | 🚀 **Launches the Interactive Streamlit Web Dashboard** in your default browser at `http://localhost:8501` |
| **`run_tests.bat`** | ✅ **Runs all 28 unit tests** to verify network simulation, QUBO, and safety layers |
| **`run_benchmark.bat`** | 📊 **Executes reproducible offline benchmarks** across multiple random seeds |

---

## 🧭 Dashboard Controls & Button Guide

The dashboard includes friendly, color-coded buttons with icons to make navigation seamless:

### ⚡ Quick Scenario Presets (Sidebar)
- **`🏙️ Balanced`**: Standard balanced traffic flow with Hybrid QAOA optimizer.
- **`🚗 Heavy Grid`**: Heavy gridlock stress test.
- **`🌅 Rush Hour`**: Heavy inbound commuter traffic rush.
- **`🚨 Emergency Vehicle`**: Deploys an emergency vehicle to demonstrate the green corridor wave.

### 🎮 Playback Controls
- **`▶️ Start`**: Starts continuous real-time traffic simulation.
- **`⏸️ Pause`**: Freezes current simulation state for inspection.
- **`⏭️ Step (1s)`**: Advances the simulation clock by 1 step.
- **`🔄 Reset`**: Clears queues and resets the network to t=0s.

### 🚨 Real-Time Incident Injection
- **`📈 Demand Surge`**: Injects a +40% traffic spike on the selected approach.
- **`🚧 Lane Closure`**: Halts movement on a lane to simulate road work.
- **`💥 Accident Hazard`**: Simulates a bottleneck collision.
- **`🚨 Dispatch Emergency Vehicle`**: Clears a green corridor from origin to destination.

### 💾 Data & Reports
- **`⬇️ Download metrics.csv`**: Exports detailed simulation history for offline analysis.

---

## 📑 Dashboard Tabs Navigation

1. **`🚦 Live Simulation & Map`**: Interactive Leaflet map with real curved street routes, live intersection signal timers, and rolling queue chart.
2. **`📊 Performance Analytics`**: Bar chart comparisons of Queue, Delay, Throughput, and CO₂ between controllers.
3. **`🌱 Carbon & Monthly Traffic AI`**: Environmental CO₂ offset analytics, fuel savings, and monthly traffic projection AI.
4. **`🚨 Emergency & Incidents`**: Real-time tracking of active emergency vehicles and roadway obstructions.
5. **`📘 How It Works & Guide`**: Architecture summary and safety validation rules.

---

## 💻 Manual Terminal Command
If you prefer running from the terminal:
```powershell
# Run dashboard
& ".\.venv\Scripts\streamlit.exe" run app.py

# Run unit tests
& ".\.venv\Scripts\pytest.exe" tests/ -v
```
