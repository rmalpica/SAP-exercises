import cantera as ct
import numpy as np
import matplotlib.pyplot as plt
#%%
def heating_value(mech, fuel, equivalence_ratio):
    """Returns the LHV and HHV for the specified fuel and equivalence ratio"""
    gas = ct.Solution(mech)
    gas.TP = 298.15, ct.one_atm
    gas.set_equivalence_ratio(equivalence_ratio, fuel, "O2:1.0, N2:3.76")
    h1 = gas.enthalpy_mass
    Y_fuel = gas[fuel].Y[0]

    # Adjust product composition for complete combustion
    if equivalence_ratio < 1.0:
        # Fuel lean: excess O2
        X_products = {
        "CO2": gas.elemental_mole_fraction("C"),
        "H2O": 0.5 * gas.elemental_mole_fraction("H"),
        "N2": 0.5 * gas.elemental_mole_fraction("N"),
        "O2": 0.5 * gas.elemental_mole_fraction("O") - 0.5 * gas.elemental_mole_fraction("H") 
        }
    elif equivalence_ratio > 1.0:
        # Fuel rich: remaining fuel
        X_products = {
        "CO2": 0.25 * gas.elemental_mole_fraction("O"),
        "H2O": 0.5 * gas.elemental_mole_fraction("O"),
        "N2": 0.5 * gas.elemental_mole_fraction("N"),
        #"CH4": (gas.elemental_mole_fraction("C") - 0.25 * gas.elemental_mole_fraction("O") )
        }
        X_products[fuel] = (gas.elemental_mole_fraction("C") - 0.25 * gas.elemental_mole_fraction("O") )
    else:
        X_products = {
        "CO2": gas.elemental_mole_fraction("C"),
        "H2O": 0.5 * gas.elemental_mole_fraction("H"),
        "N2": 0.5 * gas.elemental_mole_fraction("N")
        }
        #X_products[fuel] = (equivalence_ratio - 1.0) * gas[fuel].X[0]

    #print(X_products)

    gas.TPX = None, None, X_products
    Y_H2O = gas["H2O"].Y[0]
    h2 = gas.enthalpy_mass

    # Calculate latent heat of vaporization for water at standard conditions (298 K, 1 atm)
    water_liquid = ct.Water()
    water_vapor = ct.Water()
    water_liquid.TP = 298.15, ct.one_atm
    water_vapor.TP = 298.15, ct.one_atm
    water_vapor.TP = 373.15, ct.one_atm  # Boiling point of water at 1 atm

    h_liquid = water_liquid.h  # Enthalpy of liquid water at 298.15 K
    h_gas = water_vapor.h  # Enthalpy of water vapor at 298.15 K (boiling point)
    latent_heat_vaporization = h_gas - h_liquid

    LHV = -(h2 - h1) / Y_fuel / 1e6  # in MJ/kg
    HHV = -(h2 - h1 + latent_heat_vaporization * Y_H2O) / Y_fuel / 1e6  # in MJ/kg

    return LHV, HHV

def get_enthalpy_of_formation(mech, species_name):
    gas = ct.Solution(mech)
    T0 = 298.15  # Standard temperature in K
    P0 = ct.one_atm  # Standard pressure in Pa
    gas.TP = T0, P0
    species_index = gas.species_index(species_name)
    Hf_standard = gas.standard_enthalpies_RT[species_index] * ct.gas_constant * T0
    Hf_standard_J_per_mol = Hf_standard / 1e3
    return Hf_standard_J_per_mol

# Function to get the enthalpy of formation of a mixture
def get_mixture_enthalpy_of_formation(mech, species_composition):
    gas = ct.Solution(mech)
    T0 = 298.15  # Standard temperature in K
    P0 = ct.one_atm  # Standard pressure in Pa
    gas.TP = T0, P0
    gas.X = species_composition  # Set the mole fractions of the species in the mixture

    # Calculate the total molar mass of the mixture (in g/mol)
    total_molar_mass = gas.mean_molecular_weight 

    # Calculate the enthalpy of formation of the mixture per unit mass (in kJ/kg)
    Hf_mixture_per_mass = 0.0
    for species_name, mole_fraction in zip(gas.species_names, species_composition):
        Hf_species = get_enthalpy_of_formation(mech,species_name)  # in J/mol
        Hf_mixture_per_mass += 1e3 * (mole_fraction * Hf_species) / total_molar_mass # J/kg

    return Hf_mixture_per_mass

def compute_combustion_efficiency(mech, fuel, inlet_composition, outlet_composition, T_inlet, P_inlet):
    debug = False
    T0 = 298.15  # Standard temperature in K
    P0 = ct.one_atm  # Standard pressure in Pa
    # Create gas objects for inlet and outlet conditions
    gas_inlet = ct.Solution(mech)
    gas_outlet = ct.Solution(mech)
    gas_ideal = ct.Solution(mech)

    
    # Set inlet conditions
    gas_inlet.TPX = T0, P0, inlet_composition
    Y_fuel = gas_inlet[fuel].Y[0]
    if(debug): print("inlet X ",inlet_composition)
    inlet_enthalpy = gas_inlet.enthalpy_mass
    if(debug): print("inlet_enthalpy ",inlet_enthalpy)
    if(debug): inlet_enthalpy = get_mixture_enthalpy_of_formation(mech, inlet_composition)
    if(debug): print("inlet_enthalpy ",inlet_enthalpy)

    # Set outlet conditions
    gas_outlet.TPX = T0, P0, outlet_composition
    if(debug): print("outlet X ",outlet_composition)
    outlet_enthalpy = gas_outlet.enthalpy_mass
    if(debug): print("outlet_enthalpy ",outlet_enthalpy)
    if(debug): outlet_enthalpy = get_mixture_enthalpy_of_formation(mech, outlet_composition)
    if(debug): print("outlet_enthalpy ",outlet_enthalpy)
    
    # Calculate actual heat released (equivalent to absolute enthalpy difference at const temperature)
    actual_heat_release = (inlet_enthalpy - outlet_enthalpy) / Y_fuel
    if(debug): print("actual_heat_release [MJ/kg]: ", actual_heat_release * 1e-6)
    
    # Assume complete combustion as equilibrium condition
    # Set ideal combustion products composition
    gas_ideal.TPX = T_inlet, P_inlet, inlet_composition 
    gas_ideal.equilibrate('HP')
    equilibrium_composition = gas_ideal.X
    if(debug): print("equilibrium X ",equilibrium_composition)
    gas_ideal.TPX = T0, P0, equilibrium_composition 
    ideal_enthalpy = gas_ideal.enthalpy_mass
    if(debug): print("ideal_enthalpy ",ideal_enthalpy)
    
    # Calculate theoretical heat release
    theoretical_heat_release = (inlet_enthalpy - ideal_enthalpy) / Y_fuel
    if(debug): print("theoretical_heat_release [MJ/kg]: ", theoretical_heat_release * 1e-6)

    # Instead, assume LHV as theoretical heat released
    gas_ideal.TPX = T_inlet, P_inlet, inlet_composition 
    eqratio = gas_ideal.equivalence_ratio()
    if(debug): print("Equiv ratio :", eqratio)
    LHV, HHV = heating_value(mech, fuel, eqratio) 
    theoretical_heat_release = LHV * 1e6 #J/Kg
    if(debug): print("theoretical_heat_release (LHV) [MJ/kg]: ", theoretical_heat_release * 1e-6)

    
    # Calculate combustion efficiency
    combustion_efficiency = actual_heat_release / theoretical_heat_release
    
    return combustion_efficiency

