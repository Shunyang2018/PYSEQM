# Perform Molecular Dynamics with batch of systems, see test8.py
# For furthur usage, see examples in tests/test*
import torch
from seqm.seqm_functions.constants import Constants
from seqm.XLBOMD import XL_BOMD

# torch setting
torch.set_default_dtype(torch.float64)
if torch.cuda.is_available():
    device = torch.device('cuda')
else:
    device = torch.device('cpu')

# Specify SEQM parameters
seqm_parameters = {
                   'method' : 'AM1',  # AM1, MNDO, PM3
                   'scf_eps' : 1.0e-6,  # unit eV, SCF converging threshold
                   'scf_converger' : [2,0.0], # converger used for scf loop
                                         # [0, 0.1], [0, alpha] constant mixing, P = alpha*P + (1.0-alpha)*Pnew
                                         # [1], adaptive mixing
                                         # [2], adaptive mixing, then pulay
                   'sp2' : [True, 1.0e-5],  # whether to use sp2 algorithm in scf loop,
                                            #[True, eps] or [False], eps for SP2 conve criteria
                   'elements' : [0,1,6,8], # element atomic number, ascending order
                   'learned' : [], # learned parameters name list, e.g ['U_ss'],
                        # name lists in seqm.basics.parameterlist
                        # if empty, loaded from ./seqm/params,
                        # otherwise provide dictionary for modules Energy, Force or MD, see test5.py
                        # or provide callable function for modules, f(species, coordinates), which return a same type dictionary as in test5.py
                   'pair_outer_cutoff' : 1.0e10, # consistent with the unit on coordinates
                   }

# Prepare configurations
# Note configurations are prepared in batch mode, i.e. first dimension: molecules
#example with two molecule systems
# [O, C, H, H]
# species in descending order
species = torch.as_tensor([[8,6,1,1],[8,6,1,1]],dtype=torch.int64, device=device)
coordinates = torch.tensor([
                  [
                   [0.014497970281389074, 3.208059797520201e-05, -1.0697192468126102e-07],
                   [1.3364260161590171, -3.26283382508022e-05, 8.510168803526663e-07],
                   [1.7576599286132542, 1.0395080227523756, -5.348699492766755e-07],
                   [1.757558154681721, -1.039614513603968, 2.8473584469483316e-06]
                  ],
                  [
                   [0.014497970281389074, 3.208059797520201e-05, -1.0697192468126102e-07],
                   [1.3364260161590171, -3.26283382508022e-05, 8.510168803526663e-07],
                   [1.7576599286132542, 1.0395080227523756, -5.348699492766755e-07],
                   [1.757558154681721, -1.039614513603968, 2.8473584469483316e-06]
                  ]
                 ], device=device)

# Prepare some constant parameters
const = Constants().to(device)


# Setup Molecular Dynamics Engine
#md =  Molecular_Dynamics_Basic(seqm_parameters, timestep=1.0).to(device)
#md =  Molecular_Dynamics_Langevin(seqm_parameters, timestep=1.0, damp=100.0, T=300.0, output={'molid':[0, 1], 'thermo':1, 'dump':10, 'prefix':'md'}).to(device)
md = XL_BOMD(seqm_parameters, timestep=1.0, k=9).to(device)

# Initialize velocities (can also be provided)
velocities = md.initialize_velocity(const, coordinates, species, Temp=300.0)

# NVE or NVT
#coordinates, velocities, accelaration =  md.run(const, 20, coordinates, velocities, species)

# XL-BOMD
coordinates, velocities, accelaration, P, Pt =  md.run(const, 20, coordinates, velocities, species)
