#Testing original code for E-waste and lifecycle. Revisited using AI.

import numpy as np
import matplotlib.pyplot as plt
import pandas as pd

def simulate_hpc_sustainability(years, altitude, temp_ambient, rh, salinity, embodied_co2=500, ops_co2_per_year=200):
    # 1. Atmospheric and Thermal Modeling
    pressure = 760 * (1 - 0.0065 * altitude / 288.15)**5.255
    density_ratio = pressure / 760
    r_theta_mult = 1 / (density_ratio**0.6)
    tj = temp_ambient + (40 * r_theta_mult)
    
    # 2. Reliability & Lifecycle (Peck + Arrhenius)
    Ea, k = 0.7, 8.617e-5
    af_temp = np.exp((Ea/k) * (1/(25+273.15) - 1/(tj+273.15)))
    af_hum = (rh / 50)**2.5
    total_stress = af_temp * af_hum * salinity
    
    # Characteristic Life (eta) - Baseline 12 years
    eta = 12 / (total_stress**0.5)
    t = np.linspace(0, years, 100)
    survival = 100 * np.exp(-(t / eta)**2.5)
    
    # 3. Sustainability Metrics (CO2 Impact)
    # Embodied CO2 per system is 'embodied_co2' kg
    # When survival falls below a threshold (e.g. 50%), we consider the hardware replaced.
    # For a simplified continuous model: Impact = (t / expected_life) * embodied_co2
    # Expected life is roughly eta * Gamma(1 + 1/beta)
    expected_life = eta * 0.887 # Gamma(1.4) for beta=2.5
    
    # Cumulative E-waste impact (CO2e from manufacturing)
    cumulative_ewaste_co2 = (t / expected_life) * embodied_co2
    
    # Operational impact (energy for operation and cooling)
    # Cooling penalty at high altitude (fans spinning faster)
    cooling_penalty = 1 / density_ratio # simplified: power proportional to 1/density
    ops_impact = t * ops_co2_per_year * cooling_penalty
    
    total_sustainability_impact = cumulative_ewaste_co2 + ops_impact
    
    return t, survival, total_sustainability_impact, tj, expected_life

# Scenarios
scenarios = {
    'Cartagena (Coastal)': {'alt': 0,    't': 31, 'rh': 85, 's': 2.5, 'color': 'blue'},
    'Bucaramanga (Mid)':    {'alt': 959,  't': 24, 'rh': 70, 's': 1.0, 'color': 'orange'},
    'Toluca (High)':       {'alt': 2660, 't': 14, 'rh': 55, 's': 1.0, 'color': 'green'}
}

fig, (ax1, ax2) = plt.subplots(2, 1, figsize=(12, 12))

for name, p in scenarios.items():
    t, surv, impact, tj, life = simulate_hpc_sustainability(10, p['alt'], p['t'], p['rh'], p['s'])
    
    # Plot Survival
    ax1.plot(t, surv, label=f"{name} (Life: {life:.1f}y)", color=p['color'], lw=2.5)
    # Markers for 5 and 10 years
    idx5 = np.argmin(np.abs(t - 5))
    idx10 = np.argmin(np.abs(t - 10))
    ax1.scatter([5, 10], [surv[idx5], surv[idx10]], color=p['color'], edgecolors='black', zorder=5)
    ax1.text(5, surv[idx5]+2, f"{surv[idx5]:.0f}%", color=p['color'], fontsize=9, fontweight='bold')
    ax1.text(10, surv[idx10]+2, f"{surv[idx10]:.0f}%", color=p['color'], fontsize=9, fontweight='bold')
    
    # Plot Sustainability Impact
    ax2.plot(t, impact, label=name, color=p['color'], lw=2.5)
    ax2.scatter([5, 10], [impact[idx5], impact[idx10]], color=p['color'], edgecolors='black', zorder=5)
    ax2.text(5, impact[idx5]+50, f"{int(impact[idx5])} kg", color=p['color'], fontsize=9, fontweight='bold')
    ax2.text(10, impact[idx10]+50, f"{int(impact[idx10])} kg", color=p['color'], fontsize=9, fontweight='bold')

# Ax1 Formatting
ax1.set_title('HPC Reliability: Survival Probability at 5 and 10 Years', fontsize=14)
ax1.set_ylabel('Survival Probability (%)', fontsize=12)
ax1.set_xlabel('Years', fontsize=12)
ax1.axvline(5, color='gray', linestyle='--', alpha=0.3)
ax1.axvline(10, color='gray', linestyle='--', alpha=0.3)
ax1.legend()
ax1.grid(True, alpha=0.2)

# Ax2 Formatting
ax2.set_title('Sustainability Impact: Cumulative CO2e (Embodied + Operational)', fontsize=14)
ax2.set_ylabel('Total CO2 Equivalent (kg)', fontsize=12)
ax2.set_xlabel('Years', fontsize=12)
ax2.axvline(5, color='gray', linestyle='--', alpha=0.3)
ax2.axvline(10, color='gray', linestyle='--', alpha=0.3)
ax2.legend()
ax2.grid(True, alpha=0.2)

plt.tight_layout()
plt.savefig('hpc_sustainability_lifecycle.png')
plt.show()

# Output table for confirmation
results = []
for name, p in scenarios.items():
    t, surv, impact, tj, life = simulate_hpc_sustainability(10, p['alt'], p['t'], p['rh'], p['s'])
    results.append({
        'Site': name,
        'Survival 5y (%)': surv[np.argmin(np.abs(t - 5))],
        'Survival 10y (%)': surv[np.argmin(np.abs(t - 10))],
        'Impact 5y (kg CO2e)': impact[np.argmin(np.abs(t - 5))],
        'Impact 10y (kg CO2e)': impact[np.argmin(np.abs(t - 10))],
        'Mean Life (y)': life
    })

print(pd.DataFrame(results).to_string(index=False))
