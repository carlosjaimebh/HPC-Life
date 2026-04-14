#Another code observing temperature and Altitude

import numpy as np
import matplotlib.pyplot as plt
import pandas as pd

def calculate_hpc_physics(altitude_m, temp_c):
    # Constants
    P0 = 760        # Sea level pressure (mmHg)
    L = 0.0065      # Temperature lapse rate (K/m)
    T0 = 288.15     # Sea level standard temp (K)
    R_spec = 287.05 # J/(kg·K) for dry air
    
    # 1. Calculate Atmospheric Pressure (mmHg)
    pressure = P0 * (1 - L * altitude_m / T0)**5.255
    
    # 2. Calculate Air Density (kg/m^3)
    temp_k = temp_c + 273.15
    pressure_pa = pressure * 133.322 # Convert mmHg to Pascals
    density = pressure_pa / (R_spec * temp_k)
    
    # 3. Cooling Efficiency (Normalized to sea level @ 20°C ≈ 1.204 kg/m^3)
    cooling_eff = (density / 1.204) * 100
    
    return pressure, density, cooling_eff

# Site Data: Cartagena, Bucaramanga, Toluca
data = {
    'City': ['Cartagena', 'Bucaramanga', 'Toluca'],
    'Altitude': [10, 959, 2660],
    'Temperature': [31, 24, 13],
    'Humidity': [85, 75, 60]
}

df = pd.DataFrame(data)
results = [calculate_hpc_physics(a, t) for a, t in zip(df['Altitude'], df['Temperature'])]
df['Pressure'], df['Density'], df['Cooling_Eff'] = zip(*results)

# 3D Visualization
fig = plt.figure(figsize=(12, 8))
ax = fig.add_subplot(111, projection='3d')

# Plotting sites
# Size of points represents Humidity; Color represents Cooling Efficiency
img = ax.scatter(df['Altitude'], df['Temperature'], df['Pressure'], 
                c=df['Cooling_Eff'], cmap='RdYlGn', s=df['Humidity']*10, edgecolors='k')

# Labels and Annotations
for i, city in enumerate(df['City']):
    ax.text(df['Altitude'][i], df['Temperature'][i], df['Pressure'][i], 
            f'  {city}\n  Eff: {df["Cooling_Eff"][i]:.1f}%', fontsize=10)

ax.set_xlabel('Altitude (m)')
ax.set_ylabel('Temperature (°C)')
ax.set_zlabel('Pressure (mmHg)')
plt.title('Multiphysics Analysis of HPC Sites\nColor=Cooling Efficiency | Size=Humidity')
plt.colorbar(img, label='Relative Cooling Efficiency (%)', shrink=0.6)

plt.show()
