#Testing Wear Simulation by Nico and CJ
import numpy as np
import matplotlib.pyplot as plt
import pandas as pd

def simulate_material_degradation(altitude_m, temp_c, rh_percent, salinity_mult):
    # 1. Atmospheric Pressure (mmHg)
    P0, L, T0 = 760, 0.0065, 288.15
    pressure = P0 * (1 - L * altitude_m / T0)**5.255
    
    # 2. Corrosion Kinetics (Simplified Peck's Model)
    # Ea: Activation Energy for atmospheric corrosion (~0.4 eV)
    # R_k: Boltzmann constant (eV/K)
    Ea, R_k = 0.4, 8.617e-5
    temp_k = temp_c + 273.15
    
    # Thermal Acceleration (Arrhenius)
    thermal_acc = np.exp(-Ea / (R_k * temp_k))
    
    # Humidity Impact (Non-linear increase in electrolytic film)
    humidity_acc = (rh_percent / 100)**3 
    
    # Combined Degradation Index
    # Scaled by salinity (1.5 for coastal Cartagena, 1.0 for inland)
    degradation_rate = thermal_acc * humidity_acc * salinity_mult * 1e6
    
    return pressure, degradation_rate

# Site Data Input
sites = {
    'City': ['Cartagena', 'Bucaramanga', 'Toluca'],
    'Altitude': [10, 959, 2660],
    'Temp': [31, 24, 13],
    'RH': [85, 75, 60],
    'Salinity': [1.5, 1.0, 1.0] # High salinity for coastal sites
}

df = pd.DataFrame(sites)
results = [simulate_material_degradation(a, t, rh, s) 
           for a, t, rh, s in zip(df['Altitude'], df['Temp'], df['RH'], df['Salinity'])]
df['Pressure'], df['Degradation_Rate'] = zip(*results)

# 3D Visualization
fig = plt.figure(figsize=(12, 8))
ax = fig.add_subplot(111, projection='3d')

# Color represents the Intensity of Degradation; Size represents Humidity (RH)
scatter = ax.scatter(df['Altitude'], df['Temp'], df['Pressure'], 
                    c=df['Degradation_Rate'], cmap='YlOrRd', 
                    s=df['RH']*8, edgecolors='black', alpha=0.9)

for i, city in enumerate(df['City']):
    ax.text(df['Altitude'][i], df['Temp'][i], df['Pressure'][i], 
            f'  {city}\n  Rate: {df["Degradation_Rate"][i]:.2f}', fontsize=10)

ax.set_xlabel('Altitude (m)')
ax.set_ylabel('Temperature (°C)')
ax.set_zlabel('Pressure (mmHg)')
plt.title('HPC Infrastructure Degradation Risk Analysis\n(Corrosion Kinetics: Temp + RH + Salinity)')
plt.colorbar(scatter, label='Normalized Degradation Rate', shrink=0.5)

plt.show()
