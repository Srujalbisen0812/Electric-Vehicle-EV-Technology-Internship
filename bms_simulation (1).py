"""
Battery Management System (BMS) Simulation
=========================================
Models a 48V / 100Ah Li-Ion pack (8-cell series, NMC chemistry).

Features
--------
* SOC estimation  – Coulomb counting + OCV lookup model
* SOH estimation  – Cycle ageing + temperature-stress degradation
* Thermal model   – RC-network: I²R heat, convective dissipation, active cooling
* Cell-level mon  – Per-cell voltage, temperature, resistance & passive balancing
* Fault detection – Over/under voltage, over-temperature, cell imbalance

Usage
-----
    python bms_simulation.py                 # interactive menu (Unix)
    python bms_simulation.py --auto          # 60-second automated discharge demo
    python bms_simulation.py --auto --soc 40 --ambient 35
"""

import math
import time
import random
import argparse
import sys
from dataclasses import dataclass, field
from typing import List, Optional
from enum import Enum, auto


# ─────────────────────────────────────────────────────────────
#  Pack constants
# ─────────────────────────────────────────────────────────────
CAPACITY_AH          = 100.0     # nominal capacity (Ah)
CELL_COUNT           = 8         # cells in series
V_NOM_PACK           = 48.0      # nominal pack voltage (V)
V_MAX_PACK           = 54.4      # max charge voltage
V_MIN_PACK           = 38.4      # min discharge cutoff
CELL_V_NOM           = 3.6       # cell nominal voltage
CELL_V_MAX           = 4.20      # cell OVP threshold
CELL_V_MIN           = 2.80      # cell UVP threshold
SOC_UPPER_LIMIT      = 95.0      # charge target (%)
SOC_LOWER_LIMIT      = 10.0      # discharge cutoff (%)
MAX_CURRENT_A        = 200.0     # peak continuous current (A)

# Thermal constants (RC network)
R_THERMAL            = 2.5       # thermal resistance  (K/W)
C_THERMAL            = 800.0     # thermal capacitance (J/K)
T_COOLING_ON         = 45.0      # active-cooling activation threshold (°C)
T_CRITICAL           = 60.0      # overtemperature fault threshold (°C)
COOLING_POWER        = 0.8       # cooling power coefficient (W/K above threshold)

# Degradation
CYCLE_FADE_PER_CYCLE = 0.00035   # capacity fade fraction per equivalent full cycle
TEMP_STRESS_PER_C    = 0.002     # extra degradation fraction per °C above 40 °C

# Balancing
BALANCE_DV_THRESHOLD = 0.020     # passive balancing triggers above this ΔV (V)


# ─────────────────────────────────────────────────────────────
#  Enumerations
# ─────────────────────────────────────────────────────────────
class Mode(Enum):
    IDLE      = auto()
    CHARGE    = auto()
    DISCHARGE = auto()


class FaultCode(Enum):
    OVERVOLTAGE     = "OVP  – pack voltage > 53 V"
    UNDERVOLTAGE    = "UVP  – pack voltage < 39 V"
    OVERTEMPERATURE = "OTP  – cell temp > 60 °C"
    LOW_SOC         = "Low SOC – discharge cutoff"
    CELL_OVP        = "Cell OVP – individual cell > 4.20 V"
    CELL_UVP        = "Cell UVP – individual cell < 2.80 V"
    CELL_IMBALANCE  = "Cell imbalance – ΔV > 50 mV"
    SOH_LOW         = "SOH degraded – below 70 %"


# ─────────────────────────────────────────────────────────────
#  Data classes
# ─────────────────────────────────────────────────────────────
@dataclass
class Cell:
    cell_id:  int
    voltage:  float = 0.0      # V
    temp:     float = 25.0     # °C
    r_int:    float = 0.0015   # Ω  (internal resistance)
    capacity: float = 1.0      # fractional vs nominal

    def status_str(self) -> str:
        v_ok = CELL_V_MIN < self.voltage < CELL_V_MAX
        t_ok = self.temp < T_CRITICAL
        flag = "OK   " if (v_ok and t_ok) else "FAULT"
        return (f"  Cell {self.cell_id:2d}  │  {self.voltage:.4f} V  │"
                f"  {self.temp:5.1f} °C  │  {self.r_int*1000:.2f} mΩ  │  [{flag}]")


@dataclass
class BMSState:
    # Electrical
    soc:     float = 85.0    # %
    soh:     float = 96.0    # %
    voltage: float = 48.6    # V
    current: float = 0.0     # A  (positive = discharge)
    mode:    Mode  = Mode.IDLE

    # Thermal
    temp:         float = 25.0   # °C  (pack average)
    ambient_temp: float = 25.0   # °C

    # Ageing
    cycle_count: float = 0.0
    r_int_pack:  float = 0.012   # Ω  (total series resistance)

    # Sub-components
    cells:  List[Cell]      = field(default_factory=list)
    faults: List[FaultCode] = field(default_factory=list)
    log:    List[str]       = field(default_factory=list)

    # Private tracking
    _ah_throughput: float = 0.0
    _balancing:     bool  = False


# ─────────────────────────────────────────────────────────────
#  OCV model  (polynomial fit, NMC cell)
# ─────────────────────────────────────────────────────────────
def ocv_cell(soc_pct: float) -> float:
    """Open-circuit voltage of one cell as a function of SOC (%)."""
    s = max(0.0, min(1.0, soc_pct / 100.0))
    return 3.2 + 0.8 * s + 0.3 * s * s - 0.1 * (s - 0.5) ** 2


def ocv_pack(soc_pct: float) -> float:
    return ocv_cell(soc_pct) * CELL_COUNT


# ─────────────────────────────────────────────────────────────
#  SOC Estimation  (Coulomb counting + OCV correction)
# ─────────────────────────────────────────────────────────────
def estimate_soc_ocv(voltage: float, current: float, r_int: float) -> float:
    """
    Estimate SOC from terminal voltage using OCV-model inversion.
    Uses binary search to invert the polynomial OCV(s).
    """
    ocv       = voltage + current * r_int      # back-calculate OCV
    cell_ocv  = ocv / CELL_COUNT
    lo, hi    = 0.0, 100.0
    for _ in range(40):
        mid = (lo + hi) / 2.0
        if ocv_cell(mid) < cell_ocv:
            lo = mid
        else:
            hi = mid
    return (lo + hi) / 2.0


def coulomb_count(state: BMSState, dt_s: float) -> float:
    """Integrate current to update SOC (Coulomb counting)."""
    usable_capacity = CAPACITY_AH * (state.soh / 100.0)
    d_soc = -(state.current * dt_s / 3600.0) / usable_capacity * 100.0
    return max(0.0, min(100.0, state.soc + d_soc))


# ─────────────────────────────────────────────────────────────
#  SOH Estimation
# ─────────────────────────────────────────────────────────────
def estimate_soh(cycle_count: float, temp: float,
                 nominal_cap_frac: float = 0.985) -> float:
    """
    SOH = f(cycles, temperature):
      - Linear capacity fade per cycle
      - Temperature-stress acceleration above 40 °C
    Returns SOH in percent.
    """
    cycle_deg   = max(0.0, 1.0 - cycle_count * CYCLE_FADE_PER_CYCLE)
    temp_excess = max(0.0, temp - 40.0)
    temp_stress = max(0.0, 1.0 - temp_excess * TEMP_STRESS_PER_C)
    return min(100.0, cycle_deg * temp_stress * nominal_cap_frac * 100.0)


# ─────────────────────────────────────────────────────────────
#  Thermal Model  (first-order RC network)
# ─────────────────────────────────────────────────────────────
def thermal_step(state: BMSState, dt_s: float) -> float:
    """
    RC thermal model:
        C_th * dT/dt = Q_gen - Q_diss - Q_cool

    Q_gen  = I² × R_int             (Joule heating)
    Q_diss = (T - T_amb) / R_th     (natural convection / conduction)
    Q_cool = active cooling when T > T_COOLING_ON
    """
    I      = abs(state.current)
    q_gen  = I * I * state.r_int_pack
    q_diss = (state.temp - state.ambient_temp) / R_THERMAL
    q_cool = 0.0
    if state.temp > T_COOLING_ON:
        q_cool = (state.temp - T_COOLING_ON) * COOLING_POWER

    dT = (q_gen - q_diss - q_cool) / C_THERMAL * dt_s
    return max(state.ambient_temp - 2.0, state.temp + dT)


# ─────────────────────────────────────────────────────────────
#  Cell-level simulation
# ─────────────────────────────────────────────────────────────
def init_cells(ambient_temp: float = 25.0) -> List[Cell]:
    """Initialise 8 cells with realistic manufacturing spread."""
    cells = []
    for i in range(CELL_COUNT):
        cells.append(Cell(
            cell_id  = i + 1,
            voltage  = ocv_cell(85.0) + random.gauss(0, 0.005),
            temp     = ambient_temp   + random.gauss(0, 1.0),
            r_int    = 0.0015         + abs(random.gauss(0, 0.0002)),
            capacity = 1.0            - abs(random.gauss(0, 0.008)),
        ))
    return cells


def update_cells(state: BMSState, dt_s: float) -> None:
    """Update per-cell voltage and temperature each time step."""
    cell_current = state.current / CELL_COUNT
    for c in state.cells:
        # Terminal voltage = OCV - I·R  (noise models measurement uncertainty)
        c.voltage = (ocv_cell(state.soc)
                     - cell_current * c.r_int
                     + random.gauss(0, 0.0003))
        c.voltage = max(CELL_V_MIN - 0.1, min(CELL_V_MAX + 0.1, c.voltage))

        # Per-cell thermal (simplified RC, coupled to pack temperature)
        q_cell = cell_current ** 2 * c.r_int
        dT_cell = ((q_cell - (c.temp - state.ambient_temp) / (R_THERMAL * CELL_COUNT))
                   / (C_THERMAL / CELL_COUNT) * dt_s)
        c.temp = max(state.ambient_temp - 1.0, c.temp + dT_cell)


def check_balancing(state: BMSState) -> bool:
    """Return True if passive balancing should be active (ΔV > threshold)."""
    voltages = [c.voltage for c in state.cells]
    return (max(voltages) - min(voltages)) > BALANCE_DV_THRESHOLD


# ─────────────────────────────────────────────────────────────
#  Fault detection
# ─────────────────────────────────────────────────────────────
def detect_faults(state: BMSState) -> List[FaultCode]:
    faults: List[FaultCode] = []
    if state.voltage > 53.0:
        faults.append(FaultCode.OVERVOLTAGE)
    if state.voltage < 39.0:
        faults.append(FaultCode.UNDERVOLTAGE)
    if state.temp > T_CRITICAL:
        faults.append(FaultCode.OVERTEMPERATURE)
    if state.soc < SOC_LOWER_LIMIT and state.mode == Mode.DISCHARGE:
        faults.append(FaultCode.LOW_SOC)
    if state.soh < 70.0:
        faults.append(FaultCode.SOH_LOW)
    voltages = [c.voltage for c in state.cells]
    if max(voltages) > CELL_V_MAX:
        faults.append(FaultCode.CELL_OVP)
    if min(voltages) < CELL_V_MIN:
        faults.append(FaultCode.CELL_UVP)
    if max(voltages) - min(voltages) > 0.050:
        faults.append(FaultCode.CELL_IMBALANCE)
    return faults


# ─────────────────────────────────────────────────────────────
#  BMS simulation step
# ─────────────────────────────────────────────────────────────
def bms_step(state: BMSState, load_current: float, dt_s: float = 1.0) -> BMSState:
    """
    Advance the BMS simulation by dt_s seconds.

    Parameters
    ----------
    state        : current BMSState (mutated in-place and returned)
    load_current : requested load/charge magnitude in Amperes
    dt_s         : simulation time step in seconds

    Algorithm
    ---------
    1. Set current from mode and load request
    2. SOC – Coulomb counting + periodic OCV blend
    3. Terminal voltage from OCV model and I·R drop
    4. Thermal model (RC network)
    5. Per-cell voltage and temperature update
    6. Ageing: count equivalent full cycles, update R_int and SOH
    7. Passive cell balancing check
    8. Fault detection
    9. Automatic protection actions (trip on OTP / UVP / low SOC)
    """
    # ── 1. Current ───────────────────────────────────────────
    if state.mode == Mode.DISCHARGE:
        state.current = max(0.0, min(MAX_CURRENT_A, load_current))
    elif state.mode == Mode.CHARGE:
        state.current = -max(0.0, min(MAX_CURRENT_A, load_current))
    else:
        state.current = 0.0

    # ── 2. SOC (Coulomb counting + OCV blend) ────────────────
    new_soc = coulomb_count(state, dt_s)
    state._ah_throughput += abs(state.current) * dt_s / 3600.0

    if abs(state.current) < 5.0:
        ocv_soc = estimate_soc_ocv(state.voltage, state.current, state.r_int_pack)
        new_soc = 0.80 * new_soc + 0.20 * ocv_soc

    state.soc = max(0.0, min(100.0, new_soc))

    # ── 3. Terminal voltage ───────────────────────────────────
    state.voltage = (ocv_pack(state.soc)
                     - state.current * state.r_int_pack * CELL_COUNT)
    state.voltage = max(V_MIN_PACK, min(V_MAX_PACK, state.voltage))

    # ── 4. Thermal model ─────────────────────────────────────
    state.temp = thermal_step(state, dt_s)

    # ── 5. Cell-level ────────────────────────────────────────
    update_cells(state, dt_s)

    # ── 6. Ageing ────────────────────────────────────────────
    if state._ah_throughput >= CAPACITY_AH:
        state.cycle_count     += 1.0
        state._ah_throughput  -= CAPACITY_AH
        state.r_int_pack       = 0.012 + state.cycle_count * 0.000025
        state.soh              = estimate_soh(state.cycle_count, state.temp)

    # ── 7. Balancing ─────────────────────────────────────────
    state._balancing = check_balancing(state)

    # ── 8. Fault detection ───────────────────────────────────
    state.faults = detect_faults(state)

    # ── 9. Protection actions ────────────────────────────────
    if FaultCode.LOW_SOC in state.faults and state.mode == Mode.DISCHARGE:
        state.mode    = Mode.IDLE
        state.current = 0.0
        state.log.append("[PROTECTION] Low SOC – discharge halted")

    if FaultCode.OVERTEMPERATURE in state.faults:
        state.mode    = Mode.IDLE
        state.current = 0.0
        state.log.append("[PROTECTION] Overtemperature – BMS tripped")

    if state.soc >= SOC_UPPER_LIMIT and state.mode == Mode.CHARGE:
        state.mode    = Mode.IDLE
        state.current = 0.0
        state.log.append("[BMS] Charge complete – SOC target reached")

    return state


# ─────────────────────────────────────────────────────────────
#  Display helpers
# ─────────────────────────────────────────────────────────────
def bar(value: float, width: int = 30) -> str:
    filled = max(0, min(width, int(round(value / 100.0 * width))))
    return "█" * filled + "░" * (width - filled)


def print_dashboard(state: BMSState, elapsed_s: float) -> None:
    """Render a text-based BMS dashboard to the terminal."""
    sys.stdout.write("\033[2J\033[H")

    soc_bar  = bar(state.soc)
    soh_bar  = bar(state.soh)
    temp_pct = min(100.0, max(0.0, (state.temp - 10.0) / 70.0 * 100.0))
    temp_bar = bar(temp_pct)
    fault_str = "None" if not state.faults else ", ".join(f.name for f in state.faults)
    fc  = "\033[91m" if state.faults else "\033[92m"
    rst = "\033[0m"
    power_w  = abs(state.voltage * state.current)

    if state.current > 0.1 and state.mode == Mode.DISCHARGE:
        usable = CAPACITY_AH * state.soh / 100.0
        ah_left = (state.soc - SOC_LOWER_LIMIT) / 100.0 * usable
        time_rem_str = f"{ah_left / abs(state.current) * 60.0:.0f} min"
    else:
        time_rem_str = "—"

    print("╔══════════════════════════════════════════════════════════════════╗")
    print("║         BATTERY MANAGEMENT SYSTEM  –  BMS Simulation            ║")
    print("║    Li-Ion 48V / 100Ah  ·  NMC  ·  8-Cell Series Pack            ║")
    print("╠══════════════════════════════════════════════════════════════════╣")
    print(f"║  Mode: {state.mode.name:<12s}  Elapsed: {elapsed_s:6.1f} s"
          f"  Cycles: {state.cycle_count:6.1f}              ║")
    print("╠══════════════════════════════════════════════════════════════════╣")
    print(f"║  SOC  [{soc_bar}] {state.soc:5.1f} %  ║")
    print(f"║  SOH  [{soh_bar}] {state.soh:5.1f} %  ║")
    print(f"║  Temp [{temp_bar}] {state.temp:5.1f} °C ║")
    print("╠════════════════════════╦═════════════════════════════════════════╣")
    print(f"║  Voltage  {state.voltage:7.3f} V      ║  Current   {state.current:8.2f} A               ║")
    print(f"║  R_int    {state.r_int_pack*1000:7.2f} mΩ     ║  Power     {power_w:8.1f} W               ║")
    print(f"║  Ambient  {state.ambient_temp:7.1f} °C     ║  Time rem  {time_rem_str:>8s}               ║")
    print("╠════════════════════════╩═════════════════════════════════════════╣")
    bal_s  = "ACTIVE" if state._balancing else "idle  "
    cool_s = "ON " if state.temp > T_COOLING_ON else "off"
    print(f"║  Balancing: {bal_s}      Cooling: {cool_s}"
          "                                  ║")
    print("╠══════════════════════════════════════════════════════════════════╣")
    print("║  Cell-level data                                                 ║")
    print("║  ID   │   Voltage    │   Temp    │  R_int   │  Status            ║")
    print("║───────────────────────────────────────────────────────────────── ║")
    for c in state.cells:
        v_flag = "⚠" if c.voltage < 3.0 or c.voltage > 4.15 else " "
        print(f"║{v_flag} Cell {c.cell_id:2d}  │  {c.voltage:.4f} V  │"
              f"  {c.temp:5.1f} °C  │  {c.r_int*1000:.2f} mΩ  │                  ║")
    print("╠══════════════════════════════════════════════════════════════════╣")
    print(f"║  Faults: {fc}{fault_str:<57s}{rst}║")
    print("╠══════════════════════════════════════════════════════════════════╣")
    print("║  Recent log                                                      ║")
    for entry in state.log[-4:]:
        print(f"║  {entry:<66s}║")
    print("╚══════════════════════════════════════════════════════════════════╝")
    print("  Controls: [d] discharge  [c] charge  [i] idle  "
          "[+/-] current  [a] ambient  [q] quit")


# ─────────────────────────────────────────────────────────────
#  Interactive mode  (Unix / macOS terminal)
# ─────────────────────────────────────────────────────────────
def run_interactive(state: BMSState) -> None:
    """Real-time interactive simulation driven by keyboard input."""
    import tty
    import termios
    import select

    load_current = 50.0
    dt_s         = 0.5
    start_time   = time.time()

    def get_key() -> Optional[str]:
        if select.select([sys.stdin], [], [], 0)[0]:
            return sys.stdin.read(1)
        return None

    old = termios.tcgetattr(sys.stdin)
    try:
        tty.setcbreak(sys.stdin.fileno())
        while True:
            key = get_key()
            if key:
                if key == 'q':
                    break
                elif key == 'd':
                    state.mode = Mode.DISCHARGE
                    state.log.append(f"[USER] Discharge → {load_current:.0f} A")
                elif key == 'c':
                    state.mode = Mode.CHARGE
                    state.log.append(f"[USER] Charge → {load_current:.0f} A")
                elif key == 'i':
                    state.mode = Mode.IDLE
                    state.log.append("[USER] Mode → IDLE")
                elif key == '+':
                    load_current = min(MAX_CURRENT_A, load_current + 10.0)
                    state.log.append(f"[USER] Load current → {load_current:.0f} A")
                elif key == '-':
                    load_current = max(0.0, load_current - 10.0)
                    state.log.append(f"[USER] Load current → {load_current:.0f} A")
                elif key == 'a':
                    state.ambient_temp = min(50.0, state.ambient_temp + 5.0)
                    state.log.append(f"[USER] Ambient temp → {state.ambient_temp:.0f} °C")
                state.log = state.log[-30:]

            state = bms_step(state, load_current, dt_s)
            print_dashboard(state, time.time() - start_time)
            time.sleep(dt_s)
    finally:
        termios.tcsetattr(sys.stdin, termios.TCSADRAIN, old)

    print("\nSimulation ended.")


# ─────────────────────────────────────────────────────────────
#  Automated demo  (works on all platforms)
# ─────────────────────────────────────────────────────────────
def run_automated_demo(state: BMSState, duration_s: float = 60.0) -> None:
    """
    Automated 60-second demo scenario:
      0 – 20 s  │ Discharge at  80 A  (normal operation)
     20 – 40 s  │ Discharge at 150 A  (high load, triggers thermal warning)
     40 – 60 s  │ Charge at    50 A   (recovery)
    """
    print("═" * 72)
    print("  BMS Simulation  –  Automated Demo  (60 s, 10× real-time)")
    print("═" * 72)
    print(f"  {'t(s)':>5}  │  SOC%   SOH%  │  Voltage    Current   Power    │"
          f"  Temp   │  Status")
    print("  " + "─" * 70)

    schedule = [
        ( 0, 20, Mode.DISCHARGE,  80.0, "Discharge  80 A – normal operation"),
        (20, 40, Mode.DISCHARGE, 150.0, "Discharge 150 A – high load"),
        (40, 60, Mode.CHARGE,     50.0, "Charge      50 A – recovery"),
    ]

    dt_s  = 0.5
    tick  = 0
    prev_label = ""

    while tick * dt_s < duration_s:
        elapsed = tick * dt_s
        load_current = 0.0

        for (t0, t1, mode, current, label) in schedule:
            if t0 <= elapsed < t1:
                load_current = current
                if label != prev_label:
                    print(f"\n  [{elapsed:5.1f}s] ► {label}\n")
                    state.log.append(f"[SCHED] {label}")
                    prev_label = label
                    state.mode = mode
                break

        state = bms_step(state, load_current, dt_s)

        if tick % 4 == 0:
            faults = (", ".join(f.name for f in state.faults)
                      if state.faults else "—")
            bal    = "BAL " if state._balancing else "    "
            cool   = "COOL" if state.temp > T_COOLING_ON else "    "
            power  = abs(state.voltage * state.current)
            print(f"  {elapsed:5.1f}  │ {state.soc:5.1f}%  {state.soh:5.1f}%  │"
                  f"  {state.voltage:6.2f} V  {state.current:7.1f} A  {power:7.0f} W  │"
                  f"  {state.temp:5.1f}°C  │  {bal}{cool}  {faults}")

        tick += 1
        time.sleep(dt_s / 10.0)   # 10× speed

    print("\n" + "═" * 72)
    print("  Demo complete — final pack state:")
    print(f"    SOC  = {state.soc:.1f} %")
    print(f"    SOH  = {state.soh:.1f} %")
    print(f"    V    = {state.voltage:.3f} V")
    print(f"    T    = {state.temp:.1f} °C")
    print(f"    R_int= {state.r_int_pack*1000:.2f} mΩ")
    print(f"    Cycles = {state.cycle_count:.1f}")
    print()
    print("  Cell-level summary:")
    print("  " + "─" * 58)
    print(f"  {'':3s}Cell  │   Voltage    │   Temp    │  R_int   │  Status")
    print("  " + "─" * 58)
    for c in state.cells:
        print(" " + c.status_str())
    if state.log:
        print()
        print("  BMS event log (last 10 entries):")
        for entry in state.log[-10:]:
            print(f"    {entry}")
    print("═" * 72)


# ─────────────────────────────────────────────────────────────
#  Entry point
# ─────────────────────────────────────────────────────────────
def main() -> None:
    parser = argparse.ArgumentParser(
        description="Battery Management System (BMS) Simulation",
        formatter_class=argparse.ArgumentDefaultsHelpFormatter,
    )
    parser.add_argument("--auto",    action="store_true",
                        help="Run automated 60-second demo (cross-platform)")
    parser.add_argument("--soc",     type=float, default=85.0,
                        help="Initial SOC %%")
    parser.add_argument("--cycles",  type=float, default=124.0,
                        help="Initial cycle count")
    parser.add_argument("--ambient", type=float, default=25.0,
                        help="Ambient temperature °C")
    args = parser.parse_args()

    random.seed(42)

    state = BMSState(
        soc          = args.soc,
        soh          = estimate_soh(args.cycles, args.ambient),
        voltage      = ocv_pack(args.soc),
        temp         = args.ambient + 5.0,
        ambient_temp = args.ambient,
        cycle_count  = args.cycles,
        r_int_pack   = 0.012 + args.cycles * 0.000025,
        cells        = init_cells(args.ambient),
    )
    state.log.append(
        f"[INIT] BMS ready – SOC={state.soc:.1f}%  SOH={state.soh:.1f}%"
        f"  Cycles={state.cycle_count:.0f}  T_amb={args.ambient:.0f}°C"
    )

    if args.auto:
        run_automated_demo(state)
    else:
        try:
            run_interactive(state)
        except Exception:
            print("Interactive mode unavailable – running automated demo.\n")
            run_automated_demo(state)


if __name__ == "__main__":
    main()
