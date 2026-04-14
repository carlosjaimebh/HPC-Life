#Plauing with HW Constants and sites to propose a comparative analysis
import numpy as np
import matplotlib.pyplot as plt
import pandas as pd

# ==========================================
# 1. HARDWARE-SPECIFIC CONSTANTS (NVIDIA H100 / Sapphire Rapids)
# ==========================================
# Higher Ea (0.85) reflects increased sensitivity of <7nm nodes
Ea_NEXT_GEN = 0.85    
k = 8.617e-5
MTBF_H100 = 35000     # Reduced baseline for high-density accelerators
YEARS = 10
IT_LOAD_KW = 1000     # 1 MW Cluster

# Site & Grid Config
GRID_CO2 = {"Colombia": 150, "Mexico": 450}

# Cooling Scenarios for H100 at 2660m (Toluca)
# Note: Air cooling H100s at altitude often leads to Tj > 85C
TOLUCA_H100_SCENARIOS = {
    "Air (Severe Throttling)": [1.90, 88.0], 
    "Direct-to-Chip (Liquid)": [1.18, 65.0],
    "Immersion (Optimized)": [1.03, 58.0]
}

# ==========================================
# 2. LIFECYCLE SIMULATION
# ==========================================
def calculate_next_gen_rel(tj_c, alt):
    tj_k, base_k = tj_c + 273.15, 75.0 + 273.15 # Baseline H100 Tj is ~75C
    af_t = np.exp((Ea_NEXT_GEN/k) * (1/base_k - 1/tj_k))
    af_r = np.exp(alt / 1500.0) # Altitude Radiation
    # 0.08 scaling for radiation as H100 memory density is ultra-high (HBM3)
    l_total = (1/MTBF_H100) * (af_t + 0.08 * (af_r - 1))
    
    t_hours = np.linspace(0, YEARS * 8760, 100)
    return t_hours / 8760, np.exp(-l_total * t_hours) * 100

# ==========================================
# 3. COMPARATIVE ANALYSIS
# ==========================================
results = []
plt.figure(figsize=(10, 6))

for name, (pue, tj) in TOLUCA_H100_SCENARIOS.items():
    years_x, rel_y = calculate_next_gen_rel(tj, 2660)
    plt.plot(years_x, rel_y, label=f"{name} (Tj: {tj}C)", linewidth=2)
    
    # Calculate Year when reliability hits 50%
    l_calc = -np.log(0.5) / (years_x[np.argmin(np.abs(rel_y - 50))] * 8760) if any(rel_y < 50) else 0
    half_life = years_x[np.argmin(np.abs(rel_y - 50))] if any(rel_y < 50) else 10
    
    energy = (IT_LOAD_KW * pue * YEARS * 8760) / 1000
    co2 = (energy * 1000 * GRID_CO2["Mexico"]) / 1e6
    results.append([name, tj, pue, round(half_life, 1), int(co2)])

# Formatting
plt.title("H100 GPU Cluster Survival Probability (Toluca, MX - 2,660m)")
plt.ylabel("Reliability (%)"); plt.xlabel("Years"); plt.grid(True, alpha=0.3); plt.legend()
plt.show()

df_h100 = pd.DataFrame(results, columns=["Cooling", "Tj (C)", "PUE", "Half-Life (Yrs)", "CO2 (Tons)"])
print(df_h100)
