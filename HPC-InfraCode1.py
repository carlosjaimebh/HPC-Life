#The First HPC-Analysis Code made by CJ, NW and the others...
import numpy as np
import matplotlib.pyplot as plt
import pandas as pd

# Constants
k = 8.617e-5  # Boltzmann constant in eV/K
Ea = 0.7      # Activation energy for silicon degradation (eV)
T_amb_c = 25  # Ambient temperature in Celsius
T_j_sea_c = 70 # Junction temperature at sea level in Celsius (baseline)
T_j_sea_k = T_j_sea_c + 273.15
T_amb_k = T_amb_c + 273.15
R = 8.31447   # Universal gas constant
g = 9.80665   # Earth-surface gravity
M = 0.0289644 # Molar mass of dry air
P0 = 101325   # Sea level standard atmospheric pressure (Pa)
L = 0.0065    # Temperature lapse rate (K/m)
T0 = 288.15   # Sea level standard temperature (K)
MTBF_baseline = 50000 # Hours at sea level (approx 5.7 years)

altitudes = np.array([0, 1000, 3000, 5000]) # meters

def get_air_density(h):
    # Barometric formula
    P = P0 * (1 - L * h / T0)**(g * M / (R * L))
    T = T0 - L * h
    rho = (P * M) / (R * T)
    return rho

rho_vals = np.array([get_air_density(h) for h in altitudes])
rho_ratio = rho_vals[0] / rho_vals

# 1. Thermal Degradation
# Assuming constant fan speed, convective heat transfer coefficient h ~ rho^0.6
# Delta T (Tj - Tamb) is inversely proportional to h
delta_T_sea = T_j_sea_c - T_amb_c
delta_T_alt = delta_T_sea * (rho_ratio**0.6)
T_j_alt_k = T_amb_k + delta_T_alt

# Arrhenius Acceleration Factor (AF_thermal)
AF_thermal = np.exp((Ea / k) * (1/T_j_sea_k - 1/T_j_alt_k))

# 2. Soft Error Rate (SER) Scaling
# Empirical model: SER doubles roughly every 1500m
AF_radiation = np.exp(altitudes / 1500.0)

# 3. Combined Wear Effect (Simplified as multiplicative for demonstration)
# Note: MTBF is 1/FailureRate. Total failure rate = sum of individual rates.
total_AF = AF_thermal * (1 + 0.1 * (AF_radiation - 1)) # Simplified weighting
MTBF_alt = MTBF_baseline / total_AF

# 4. Fan Wear (Heuristic: Fan must spin faster to maintain cooling)
# Power needed scales with 1/rho. Mechanical wear scales roughly with RPM^3.
fan_wear_factor = (rho_ratio)**3

# Results Table
df = pd.DataFrame({
    'Altitude (m)': altitudes,
    'Air Density (kg/m3)': np.round(rho_vals, 3),
    'Junction Temp (C)': np.round(T_j_alt_k - 273.15, 1),
    'Thermal AF': np.round(AF_thermal, 2),
    'Radiation (SER) Multiplier': np.round(AF_radiation, 2),
    'Estimated MTBF (Hours)': np.round(MTBF_alt, 0),
    'Fan Wear Factor': np.round(fan_wear_factor, 2)
})

df.to_csv('hpc_wear_simulation.csv', index=False)

# Plotting
fig, ax1 = plt.subplots(figsize=(10, 6))

color = 'tab:red'
ax1.set_xlabel('Altitude (m)')
ax1.set_ylabel('Junction Temp (°C)', color=color)
ax1.plot(altitudes, T_j_alt_k - 273.15, marker='o', color=color, label='Junction Temp')
ax1.tick_params(axis='y', labelcolor=color)

ax2 = ax1.twinx()
color = 'tab:blue'
ax2.set_ylabel('MTBF (Hours)', color=color)
ax2.plot(altitudes, MTBF_alt, marker='s', color=color, label='MTBF')
ax2.tick_params(axis='y', labelcolor=color)

plt.title('HPC Wear Simulation: Altitude vs. Reliability')
fig.tight_layout()
plt.grid(True, linestyle='--', alpha=0.6)
plt.savefig('hpc_altitude_wear.png')

print(df)
