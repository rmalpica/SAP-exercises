import math

def calculate_OEW_cryogenic(oew_base, fuel_weight_cryo, fuselage_weight_cryo, fuselage_weight_base, eta_grav):
    fuselage_penalty = 0.06
    oew_cryo = oew_base + ((1-eta_grav)*fuel_weight_cryo)/eta_grav + (fuselage_penalty * fuselage_weight_base) + (fuselage_weight_cryo - fuselage_weight_base)
    return oew_cryo 

def compute_fuselage_length_increase(
    liq_fuel_mass, liq_fuel_density, fuselage_diameter, tank_diameter_ratio=1.0, margin_factor=1.02
):
    """
    Compute the fuselage length increase required to accommodate a given amount of liquid hydrogen.

    Args:
        lh2_mass: Mass of liquid hydrogen (kg)
        lh2_density: Density of liquid hydrogen (kg/m³), default is 71 kg/m³
        fuselage_diameter: Diameter of the fuselage (m)
        tank_diameter_ratio: Ratio of tank diameter to fuselage diameter, default is 0.9
        margin_factor: Margin factor for structural integration, default is 1.1

    Returns:
        tank_length: Required length of the cylindrical tank (m)
        fuselage_length_increase: Total fuselage length increase (m)
    """
    # Step 1: Compute the volume of LH2
    lh2_volume = liq_fuel_mass / liq_fuel_density  # m³

    # Step 2: Compute the tank radius
    tank_diameter = tank_diameter_ratio * fuselage_diameter
    tank_radius = tank_diameter / 2  # m

    # Step 3: Compute the required tank length
    tank_length = lh2_volume / (math.pi * tank_radius**2)  # m

    # Step 4: Apply the margin factor to account for structural integration
    fuselage_length_increase = tank_length * margin_factor

    return fuselage_length_increase

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
    target_range, mtow, oew_base, payload, cruise_speed, density, wing_area, cf,
    oswald_efficiency, aspect_ratio, calorific_value, liq_fuel_density, eta_grav, efficiency, fuselage_diameter, fuselage_length, max_iterations
):
    """
    Perform the on-design analysis

    Args:
        mtow: Maximum Take-Off Weight (kg)
        oew: Operating Empty Weight (kg)
        payload: Payload weight (kg)
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

    # Calculate baseline fuselage
    fuselage_wetted_area_base = calculate_fuselage_wetted_area(fuselage_diameter, fuselage_length)
    fuselage_weight_base = calculate_fuselage_weight(fuselage_wetted_area_base)
    wetted_area_base = calculate_wetted_area(mtow)

    # Guess fuel weight
    fuel_weight_cryo = 1000
    iteration = 0
    while iteration < max_iterations:
        iteration += 1

        # Calculate tank length -> fuselage length increase
        fuselage_length_increase = compute_fuselage_length_increase(fuel_weight_cryo, liq_fuel_density, fuselage_diameter)

        # Calculate new fuselage
        fuselage_wetted_area_cryo = calculate_fuselage_wetted_area(fuselage_diameter, fuselage_length + fuselage_length_increase)
        fuselage_weight_cryo = calculate_fuselage_weight(fuselage_wetted_area_cryo)

        oew_cryo = calculate_OEW_cryogenic(oew_base, fuel_weight_cryo, fuselage_weight_cryo, fuselage_weight_base, eta_grav)

        # Calculate GTOW
        gtow = oew_cryo + payload + fuel_weight_cryo

        # Check if GTOW exceeds MTOW
        if gtow > mtow:
            break
    

        lost_fuel = 0.014 * gtow  # 1.4% for other flight phases. LNH3=5.5, LNH=1.89, methanol=4,78, SAF/jet-A = 2.2
        winitial = gtow - lost_fuel
        wfinal = gtow - 0.9 * fuel_weight_cryo  # Residual fuel is 10%
    

        # Compute C_D0 (Zero-lift drag coefficient)
        wetted_area_cryo = wetted_area_base + (fuselage_wetted_area_cryo - fuselage_wetted_area_base) 

        cd0 = calculate_zero_lift_drag(wetted_area_cryo, wing_area, cf)

        # Estimate initial and final weights
        lost_fuel = 0.014 * gtow  # 1.4% for operational losses
        winitial = gtow - lost_fuel
        wfinal = gtow - 0.9 * fuel_weight_cryo  # Residual fuel is 10%

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

        if abs(range_km - target_range) < 1:
            break
        else:
            fuel_weight_cryo += 10.0


    print(f"Design converged after {iteration} iterations.")
    return {
        "GTOW": gtow,
        "OEW": oew_cryo,
        "OEW/GTOW": oew_cryo/gtow,
        "Fuel": fuel_weight_cryo,
        "Range (km)": range_km,
        "L/D": ld,
        "C_D0": cd0,
        "C_L": cl,
        "C_D": cd,
        "wetted area (m^2)": wetted_area_cryo,
        "fuselage length increase (m)": fuselage_length_increase,
        "fuselage length increase (%)": 100*fuselage_length_increase/fuselage_length,
        "fuselage wetted area (m^2)": fuselage_wetted_area_cryo,
        "fuselage weight (kg)": fuselage_weight_cryo,
        "wing loading (kg/m2)": gtow/wing_area,
        "SEC (MJ/ton-km)": calorific_value * 1e-6 * fuel_weight_cryo / (payload*range_km/1000) 
    }


    


# Example inputs
if __name__ == "__main__":

    case = 'A350-1000'

    if(case == 'A320'):
        ## A320 Aircraft specifications
        target_range = 3060 #km
        mtow = 73500.0  # Maximum Take-Off Weight (kg)
        oew_base = 44200.0  # Operating Empty Weight (kg)
        payload = 16565.0  # Payload weight (kg)
        cruise_speed = 234.0  # Cruise speed (m/s)
        density = 0.348  # Air density at cruise altitude (kg/m^3)
        wing_area = 122.4  # Reference wing area (m^2)
        cf = 0.0035  # Skin-friction coefficient
        oswald_efficiency = 0.85  # Oswald efficiency factor
        aspect_ratio = 9.5  # Wing aspect ratio
        calorific_value = 120e6  # Calorific value of H2-liq (MJ/kg)
        liq_fuel_density = 71.0 #kg/m3 
        eta_grav = 0.78 # tank gravimetric efficiency
        efficiency = 0.3  # Propulsion efficiency
        fuselage_diameter = 3.95
        fuselage_length = 37.57
        max_iterations = 100000
     
    if(case == 'A350-1000'):
        # A350-1000 Aircraft specifications
        target_range = 13870 #km
        mtow = 316000.0  # Maximum Take-Off Weight (kg)
        oew_base = 155129.0  # Operating Empty Weight (kg)
        payload = 34770.0  # Payload weight (kg)
        cruise_speed = 252.1  # Cruise speed (m/s)
        density = 0.38  # Air density at cruise altitude (kg/m^3)
        wing_area = 465  # Reference wing area (m^2)
        cf = 0.003  # Skin-friction coefficient
        oswald_efficiency = 0.85  # Oswald efficiency factor
        aspect_ratio = 9.12  # Wing aspect ratio
        calorific_value = 120e6  # Calorific value of H2-liq (MJ/kg)
        liq_fuel_density = 71.0 #kg/m3 
        eta_grav = 0.78 # tank gravimetric efficiency
        efficiency = 0.4  # Propulsion efficiency
        fuselage_diameter = 5.96
        fuselage_length = 72.25
        max_iterations = 100000

    # Run the analysis
    result = calculate_design_range(
        target_range, mtow, oew_base, payload, cruise_speed, density, wing_area, cf,
        oswald_efficiency, aspect_ratio, calorific_value, liq_fuel_density, eta_grav, efficiency, fuselage_diameter, fuselage_length, max_iterations
    )

    if result:
        print(f"\nFinal Design Outputs retrofitted {case}:")
        for key, value in result.items():
            print(f"{key}: {value:.2f}")
