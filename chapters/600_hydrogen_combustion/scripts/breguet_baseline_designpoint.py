import math

def calculate_wetted_area(gtow, d_factor=0.7531, c_factor=0.0199):
    """
    Calculate wetted area (S_wet) based on GTOW using the formula:
    log10(S_wet [ft^2]) = D * log10(GTOW [lb]) + C.

    Args:
        gtow: Gross Take-Off Weight in kg
        d_factor: Coefficient D in the wetted area equation (default: 0.85)
        c_factor: Coefficient C in the wetted area equation (default: -1.5)

    Returns:
        Wetted area in m^2
    """
    # Convert GTOW to pounds
    gtow_lb = gtow * 2.20462

    # Calculate wetted area in ft^2
    log_swet_ft2 = d_factor * math.log10(gtow_lb) + c_factor
    swet_ft2 = 10**log_swet_ft2

    # Convert to m^2
    swet_m2 = swet_ft2 * 0.092903
    return swet_m2

def calculate_fuselage_wetted_area(diameter_m, length_m):
    diameter_ft = diameter_m * 3.28084
    length_ft = length_m * 3.28084
    swet_ft2 = math.pi * diameter_ft * length_ft * ((1.0-(2*(diameter_ft/length_ft)))**(2./3.)) * (1+(diameter_ft/length_ft)**2)
    swet_m2 = swet_ft2 * 0.092903
    return swet_m2 

def calculate_fuselage_weight(swet_m2):
    swet_ft2 = swet_m2 / 0.092903 
    weight_lbs = 5 * swet_ft2
    weight_kg = weight_lbs / 2.20462 
    return weight_kg 


def calculate_zero_lift_drag(wetted_area, wing_area, cf):
    """
    Compute the zero-lift drag coefficient (C_D0).
    """
    return (wetted_area / wing_area) * cf

def calculate_lift_coefficient(weight, density, speed, wing_area):
    """
    Compute the lift coefficient (C_L).
    """
    return (2 * weight * 9.81) / (density * speed**2 * wing_area)

def calculate_total_drag_coefficient(cd0, cl, oswald_efficiency, aspect_ratio):
    """
    Compute the total drag coefficient (C_D).
    """
    induced_drag = cl**2 / (math.pi * oswald_efficiency * aspect_ratio)
    return cd0 + induced_drag

def calculate_l_d(cl, cd):
    """
    Compute the lift-to-drag ratio (L/D).
    """
    return cl / cd

def calculate_breguet_range(calorific_value, efficiency, ld, winitial, wfinal):
    """
    Calculate the range using the Breguet range equation:

    Args:
        calorific_value: Calorific value of the fuel (MJ/kg)
        efficiency: Propulsion efficiency
        ld: Lift-to-drag ratio
        winitial: Initial weight (kg)
        wfinal: Final weight (kg)

    Returns:
        Range in kilometers
    """
    range_m = (calorific_value / 9.81) * efficiency * ld * math.log(winitial / wfinal)
    return range_m / 1000  # Convert to kilometers

def calculate_design_range(
    mtow, oew, payload, fuel_weight, cruise_speed, density, wing_area, cf,
    oswald_efficiency, aspect_ratio, calorific_value, efficiency, fuselage_diameter, fuselage_length
):
    """
    Perform the on-design analysis, incorporating fuselage calculation.

    Args:
        mtow: Maximum Take-Off Weight (kg)
        oew: Operating Empty Weight (kg)
        payload: Payload weight (kg)
        fuel_weight: Known fuel weight (kg)
        cruise_speed: Cruise speed (m/s)
        density: Air density (kg/m^3)
        wing_area: Reference wing area (m^2)
        cf: Skin-friction coefficient
        oswald_efficiency: Oswald efficiency factor
        aspect_ratio: Wing aspect ratio
        calorific_value: Calorific value of jet fuel (MJ/kg)
        efficiency: Propulsion efficiency
        fuselage_diameter (m)
        fuselage_length (m)

    Returns:
        A dictionary containing:
          - GTOW: Gross Take-Off Weight (kg)
          - Range: Achievable range (km)
          - L/D: Lift-to-drag ratio
    """
    # Calculate GTOW
    gtow = oew + payload + fuel_weight

    # Check if GTOW exceeds MTOW
    if gtow > mtow:
        print("Error: GTOW exceeds MTOW. Design is invalid.")
        return None

    # Calculate wetted area
    wetted_area = calculate_wetted_area(gtow)
    fuselage_wetted_area = calculate_fuselage_wetted_area(fuselage_diameter, fuselage_length)
    fuselage_weight = calculate_fuselage_weight(fuselage_wetted_area)


    # Compute C_D0 (Zero-lift drag coefficient)
    cd0 = calculate_zero_lift_drag(wetted_area, wing_area, cf)

    # Estimate initial and final weights
    lost_fuel = 0.022 * gtow  # 2.2% for other flight phases 
    winitial = gtow - lost_fuel
    wfinal = gtow - 0.9 * fuel_weight  # Residual fuel is 10%

    average_cruise_weight = wfinal+(0.5*(winitial-wfinal)) 

    # Compute C_L (Lift coefficient)
    cl = calculate_lift_coefficient(average_cruise_weight, density, cruise_speed, wing_area)

    # Compute C_D (Total drag coefficient)
    cd = calculate_total_drag_coefficient(cd0, cl, oswald_efficiency, aspect_ratio)

    # Compute L/D
    ld = calculate_l_d(cl, cd)

    # Calculate range using Breguet equation
    range_km = calculate_breguet_range(
        calorific_value, efficiency, ld, winitial, wfinal
    )

    return {
        "GTOW": gtow,
        "OEW": oew,
        "OEW/GTOW": oew/gtow,
        "Fuel": fuel_weight,
        "Range (km)": range_km,
        "L/D": ld,
        "C_D0": cd0,
        "C_L": cl,
        "C_D": cd,
        "wetted area (m^2)": wetted_area,
        "fuselage wetted area (m^2)": fuselage_wetted_area,
        "fuselage weight (kg)": fuselage_weight,
        "wing loading (kg/m2)": gtow/wing_area,
        "SEC (MJ/ton-km)": calorific_value * 1e-6 * fuel_weight / (payload*range_km/1000) 
    }


# Example inputs
if __name__ == "__main__":

    case = 'A350-1000'


    if(case == 'A320'):
        ## A320 Aircraft specifications
        mtow = 73500.0  # Maximum Take-Off Weight (kg)
        oew = 44200.0  # Operating Empty Weight (kg)
        payload = 16565.0  # Payload weight (kg)
        fuel_weight = mtow - oew - payload  # Known fuel weight (kg)
        cruise_speed = 234.0  # Cruise speed (m/s)
        density = 0.348  # Air density at cruise altitude (kg/m^3)
        wing_area = 122.4  # Reference wing area (m^2)
        cf = 0.0035  # Skin-friction coefficient
        oswald_efficiency = 0.85  # Oswald efficiency factor
        aspect_ratio = 9.5  # Wing aspect ratio
        calorific_value = 43.2e6  # Calorific value of jet fuel (MJ/kg)
        efficiency = 0.3  # Propulsion efficiency
        fuselage_diameter = 3.95
        fuselage_length = 37.57

    if(case == 'A350-1000'):
    # A350-1000 Aircraft specifications
        mtow = 316000.0  # Maximum Take-Off Weight (kg)
        oew = 155129.0  # Operating Empty Weight (kg)
        payload = 34770.0  # Payload weight (kg)
        fuel_weight = mtow - oew - payload  # Known fuel weight (kg)
        cruise_speed = 252.1  # Cruise speed (m/s)
        density = 0.38  # Air density at cruise altitude (kg/m^3)
        wing_area = 465  # Reference wing area (m^2)
        cf = 0.003  # Skin-friction coefficient
        oswald_efficiency = 0.85  # Oswald efficiency factor
        aspect_ratio = 9.12  # Wing aspect ratio
        calorific_value = 43.2e6  # Calorific value of jet fuel (MJ/kg)
        efficiency = 0.4  # Propulsion efficiency
        fuselage_diameter = 5.96
        fuselage_length = 72.25

    # Run the analysis
    result = calculate_design_range(
        mtow, oew, payload, fuel_weight, cruise_speed, density, wing_area, cf,
        oswald_efficiency, aspect_ratio, calorific_value, efficiency, fuselage_diameter, fuselage_length
    )

    if result:
        print(f"\nFinal Design Outputs {case}:")
        for key, value in result.items():
            print(f"{key}: {value:.2f}")
