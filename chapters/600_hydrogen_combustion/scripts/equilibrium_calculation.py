import numpy as np
import cantera as ct

from pathlib import Path

HERE = Path(__file__).resolve().parent
CHAPTER_DIR = HERE.parent
DATA = CHAPTER_DIR / "data"
OUTPUTS = CHAPTER_DIR / "outputs"
OUTPUTS.mkdir(parents=True, exist_ok=True)
	
fuel = 'NC10H22:0.74,PHC3H7:0.15,CYC9H18:0.11'
mechfile = DATA / "kerosene.yaml" 

species = {S.name: S for S in ct.Species.list_from_file(str(mechfile))}

# Create an ideal gas phase object with species representing complete combustion
complete_species = [species[S] for S in ("NC10H22","PHC3H7","CYC9H18", "O2", "N2", "CO2", "H2O")]
gas1 = ct.Solution(thermo="ideal-gas", species=complete_species)

phi = np.linspace(0.2, 1.5, 100)
T_complete_jetA1 = np.zeros(phi.shape)
for i in range(len(phi)):
    gas1.TP = 300, ct.one_atm
    gas1.set_equivalence_ratio(phi[i], fuel, "O2:1, N2:3.76")
    gas1.equilibrate("HP")
    T_complete_jetA1[i] = gas1.T

gas2 = ct.Solution(thermo="ideal-gas", species=species.values())
T_incomplete_jetA1 = np.zeros(phi.shape)
for i in range(len(phi)):
    gas2.TP = 300, ct.one_atm
    gas2.set_equivalence_ratio(phi[i], fuel, "O2:1, N2:3.76")
    gas2.equilibrate("HP")
    T_incomplete_jetA1[i] = gas2.T
    

fuel = 'H2'

complete_species = [species[S] for S in ("H2", "O2", "N2", "H2O")]
gas1 = ct.Solution(thermo="ideal-gas", species=complete_species)

phi = np.linspace(0.2, 1.5, 100)
T_complete_H2 = np.zeros(phi.shape)
for i in range(len(phi)):
    gas1.TP = 300, ct.one_atm
    gas1.set_equivalence_ratio(phi[i], fuel, "O2:1, N2:3.76")
    gas1.equilibrate("HP")
    T_complete_H2[i] = gas1.T

gas2 = ct.Solution(thermo="ideal-gas", species=species.values())
T_incomplete_H2 = np.zeros(phi.shape)
for i in range(len(phi)):
    gas2.TP = 300, ct.one_atm
    gas2.set_equivalence_ratio(phi[i], fuel, "O2:1, N2:3.76")
    gas2.equilibrate("HP")
    T_incomplete_H2[i] = gas2.T

import matplotlib.pyplot as plt
fig, ax = plt.subplots(figsize=(6, 6))

ax.plot(phi, T_complete_jetA1, label="kerosene complete", lw=2, ls = '-', color = 'dimgray')
ax.plot(phi, T_incomplete_jetA1, label="kerosene equilibrium", lw=2, ls = '--',color = 'dimgray')
ax.plot(phi, T_complete_H2, label="hydrogen complete", lw=2, ls = '-', color = 'forestgreen')
ax.plot(phi, T_incomplete_H2, label="hydrogen equilibrium", lw=2, ls = '--',color = 'forestgreen')

ax.grid(True)
ax.set_xlabel(r"Equivalence ratio, $\phi$")
ax.set_ylabel("Temperature [K]")
plt.legend()

plt.savefig(OUTPUTS / "Tad-H2_vs_kerosene.png", dpi=800, transparent=False)
plt.show()

