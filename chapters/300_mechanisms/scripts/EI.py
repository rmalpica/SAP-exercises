import cantera as ct
import numpy as np
import matplotlib.pyplot as plt
from efficiency_functions import * 

# Define the gas using a mechanism that includes dodecane and NOx chemistry
# Replace 'nDodecane.cti' with your mechanism file name
mech = 'CRECK_2003_TOT_HT_NOX.yaml' 
gas = ct.Solution(mech)

# Initial conditions
T0 = 1000.0  # Initial temperature in K
P0 = ct.one_atm  # Initial pressure in Pa
eqratio = 0.4  # Equivalence ratio for stoichiometric mixture

# Define fuel and oxidizer
fuel = 'NC12H26'
oxidizer = 'O2:1.0, N2:3.76'  # Air composition

# Set the gas state to the specified temperature, pressure, and composition
gas.TP = T0, P0
gas.set_equivalence_ratio(eqratio, fuel, oxidizer)

LHV, HHV = heating_value(mech, fuel, eqratio) 
theoretical_heat_release = LHV * 1e6 #J/Kg

LHV, HHV = heating_value(mech, fuel, 1) 
theoretical_heat_release = LHV * 1e6  #J/Kg
print("theoretical_heat_release (LHV) [MJ/kg]: ", theoretical_heat_release * 1e-6)

# Create a constant-pressure reactor and insert the gas
reactor = ct.IdealGasConstPressureReactor(gas, clone=False)

# Create a reactor network
sim = ct.ReactorNet([reactor])

# Initial fuel mass for EI calculation
initial_fuel_mass_fraction = gas[fuel].Y[0] 
initial_mass = reactor.mass
initial_fuel_mass = initial_mass * initial_fuel_mass_fraction

# Initial enthalpy of the system
Tstd = 298.15
gas_inlet = ct.Solution(mech)
gas_inlet.TPY = Tstd, P0, gas.Y 
H_initial = gas_inlet.enthalpy_mass

# Initialize storage for results
times = []
temperatures = []
EI = {'CO2': [], 'H2O': [], 'NO': [], 'CO': []}
combustion_efficiency = []

# Time parameters
t_end = 0.1  # End time in seconds
time_step = 1e-6  # Time step in seconds

gas_outlet = ct.Solution(mech)

# Time integration
t = 0.0
while t < t_end:
    t = sim.step()
    times.append(t)
    temperatures.append(reactor.T)

    # Current mass fractions
    Y = reactor.phase.Y
    species_names = reactor.phase.species_names

    # Mass fractions of species of interest
    Y_CO2 = gas['CO2'].Y[0]
    Y_H2O = gas['H2O'].Y[0] 
    Y_NO = gas['NO'].Y[0] 
    Y_CO = gas['CO'].Y[0] 
    Y_fuel = gas[fuel].Y[0]

    # Current mass of fuel
    current_fuel_mass = reactor.mass * Y_fuel
    m_fuel_consumed = initial_fuel_mass - current_fuel_mass

    # Heat released so far
    gas_outlet.TPY = Tstd, P0, reactor.phase.Y
    H_current = gas_outlet.enthalpy_mass 
    Q = (H_initial - H_current)  # Positive if heat is released

    # Avoid division by zero
    if m_fuel_consumed > 1e-20 and initial_fuel_mass > 1e-20:
        # Calculate emission indices
        EI_CO2 = (reactor.mass * Y_CO2) / m_fuel_consumed
        EI_H2O = (reactor.mass * Y_H2O) / m_fuel_consumed
        EI_NO = (reactor.mass * Y_NO) / m_fuel_consumed
        EI_CO = (reactor.mass * Y_CO) / m_fuel_consumed

        # Calculate combustion efficiency
        combustion_efficiency_current = Q / (theoretical_heat_release * initial_fuel_mass_fraction)
    else:
        EI_CO2 = 0.0
        EI_H2O = 0.0
        EI_NO = 0.0
        EI_CO = 0.0
        combustion_efficiency_current = 0.0

    EI['CO2'].append(EI_CO2)
    EI['H2O'].append(EI_H2O)
    EI['NO'].append(EI_NO)
    EI['CO'].append(EI_CO)
    combustion_efficiency.append(combustion_efficiency_current)


# Set the font to Times New Roman
plt.rcParams["font.family"] = "STIXGeneral"
plt.rcParams["font.size"] = 14

plt.figure(figsize=(10, 12))

# Plot emission indices with log scale on Y-axis
plt.subplot(4, 1, 1)
plt.plot(times, EI['CO2'], label='EI CO$_2$')
plt.plot(times, EI['H2O'], label='EI H$_2$O')
plt.plot(times, EI['CO'], label='EI CO')
plt.plot(times, EI['NO'], label='EI NO')
#plt.yscale('log')  # Set Y-axis to log scale
plt.xlabel('Time [s]')
plt.ylabel('Emission Index [kg/kg]')
plt.legend()
plt.title('Emission Indices over Time')

# Plot temperature
plt.subplot(4, 1, 2)
plt.plot(times, temperatures, 'k-', label='Temperature')
plt.xlabel('Time [s]')
plt.ylabel('Temperature [K]')
plt.legend()
plt.title('Temperature over Time')

# Plot combustion efficiency
plt.subplot(4, 1, 3)
plt.plot(times, [ce * 100 for ce in combustion_efficiency], 'g-', label='Combustion Efficiency')
plt.xlabel('Time [s]')
plt.ylabel('Combustion Efficiency [%]')
plt.legend()
plt.title('Combustion Efficiency over Time')

# Set transparent background
plt.subplots_adjust(hspace=0.5) 
plt.savefig('../outputs/results.png',dpi=800)  # Save figure with transparent background

plt.tight_layout()
plt.show()
