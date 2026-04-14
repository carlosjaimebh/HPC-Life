#Now, regarding sites (Cartagena, Bucaramanga and Toluca sites) 

import numpy as np
import matplotlib.pyplot as plt

def simulate_hpc_survival(years, altitude, temp_ambient, rh, salinity):
    # 1. Atmospheric Pressure Calculation
    pressure = 760 * (1 - 0.0065 * altitude / 288.15)**5.255
    
    # 2. Junction Temperature (Tj) Modeling
    # Cooling efficiency drops as density (pressure) drops
    density_ratio = pressure / 760
    # Thermal resistance increases: thinner air removes less heat
    r_theta_mult = 1 / (density_ratio**0.6)
    # Assume a standard 40C rise above ambient for high-load GPUs
    tj = temp_ambient + (40 * r_theta_mult)
    
    # 3. Reliability Acceleration Factors (AF)
    Ea, k = 0.7, 8.617e-5
    # Thermal Acceleration (Arrhenius) relative to 25C baseline
    af_temp = np.exp((Ea/k) * (1/(25+273.15) - 1/(tj+273.15)))
    # Humidity Acceleration (Peck's Law) relative to 50% baseline
    af_hum = (rh / 50)**2.5
    
    # Total combined stress factor
    total_stress = af_temp * af_hum * salinity
    
    # 4. Weibull Survival Probability
    # eta represents 'Characteristic Life' (Baseline 12 years)
    eta = 12 / (total_stress**0.5) 
    t = np.linspace(0, years, 100)
    survival = 100 * np.exp(-(t / eta)**2.5) # beta=2.5 implies wear-out phase
    
    # End of Life (EOL) at 20% survival
    eol_year = t[np.where(survival <= 20)[0][0]] if any(survival <= 20) else years
    
    return t, survival, tj, eol_year

# Scenario Definitions
scenarios = {
    'Cartagena (Sea Level)': {'alt': 0,    't': 31, 'rh': 85, 's': 2.5, 'c': 'blue'},
    'Bucaramanga (900m)':    {'alt': 959,  't': 24, 'rh': 70, 's': 1.0, 'c': 'orange'},
    'Toluca (2700m)':       {'alt': 2660, 't': 14, 'rh': 55, 's': 1.0, 'c': 'green'}
}

plt.figure(figsize=(12, 7))
for label, p in scenarios.items():
    t, surv, tj, eol = simulate_hpc_survival(10, p['alt'], p['t'], p['rh'], p['s'])
    plt.plot(t, surv, label=f"{label} (Tj: {tj:.1f}°C)", color=p['c'], lw=3)
    plt.axvline(eol, color=p['c'], linestyle='--', alpha=0.4)
    plt.text(eol, 5, f' EOL: {eol:.1f}y', color=p['c'], fontweight='bold', rotation=90)

plt.axhline(20, color='red', linestyle=':', label='20% Reliability (Critical Failure)')
plt.title('HPC Infrastructure Survival Prediction: 10-Year Lifecycle Analysis')
plt.ylabel('Probability of Hardware Survival (%)')
plt.xlabel('Years of Service')
plt.grid(True, alpha=0.3)
plt.legend()
plt.show()

