#!/usr/bin/env python
from __future__ import print_function
import sys
import numpy as np
import geom
import solve
import constants as c
from scipy.interpolate import interp1d
from scipy.interpolate import InterpolatedUnivariateSpline as interpSpl

#-------------------------------------------------------------------#
# Inputs:                                                           #
# -------                                                           #
# coord_file.xyz : all atoms in rotor must be labelled with r at    # 
#                  end of line, input units assumed to be Angstrom  #
# pot_file.txt:    potential energy values in Hartree, computed at  #
#                  equally spaced points along torsional coordinate #
#                  with offsets ranging from -180 to +180           # 
# n_fine_grid_pts: integer specifying number of grid points to use  #
#                  for Fourier grid Hamiltonian procedure, ideally  #
#                  a sensible multiple of 180 (up to 3600)          # 
#-------------------------------------------------------------------#

if __name__ == "__main__":

  if len(sys.argv) != 4:

    print('Usage: torsion_1d.py <coord_file.xyz> <pot_file.txt> <n_fine_grid_pts>')
    sys.exit()

  else:
 
    # Store command line arguments

    xyzfile = sys.argv[1]
    potfile = sys.argv[2]
    n_fine = int(sys.argv[3])

    # Process inputs

    geometries = geom.parse(xyzfile)
    with open(potfile,'r') as f: vals = f.readlines()
    V_coarse = np.array(list(map(float,vals)))
    E_min = np.min(V_coarse)
    i_min = np.argmin(V_coarse)
    V_coarse -= E_min

    # Check inputs in required format

    if (len(V_coarse)%2 == 0) or (abs(V_coarse[0]-V_coarse[-1]) >= 1.0e-12):
      print('Error: Must have odd number of input points with last value = repeat of first')
      sys.exit()
 
    if len(geometries) == 1: 
      equilibrium_geometry = geometries[0]
    elif len(geometries) == len(V_coarse):
      equilibrium_geometry = geometries[i_min]
    else:
      print('Error: Must supply either minimum energy geometry or all geometries')
      sys.exit()

    # Compute torsional moments of inertia 

    moment,reduced_moment = geom.compute_moments(equilibrium_geometry) 

    # Interpolate potential energy curve on finely-spaced grid
    # Note both interpolation schemes give very similar results
    # The advantage of spline interpolation is that derivatives are available

    n_coarse = len(V_coarse)-1
    phi_coarse = [float(i)/float(n_coarse) for i in range(0,n_coarse+1)]
    phi_fine = [float(i)/float(n_fine) for i in range(0,n_fine+1)] 
#    f = interp1d(phi_coarse, V_coarse, kind='cubic')
    f = interpSpl(phi_coarse, V_coarse, k=3)
    V_fine = f(phi_fine)

    np.savetxt('V_fine.csv',V_fine,delimiter=',')

    # Perform Fourier grid Hamiltonian calculations assuming moment of 
    # inertia does not change substantially during bond rotation

    moments = [moment for i in range(0,n_fine+1)]
    reduced_moments = [reduced_moment for i in range(0,n_fine+1)]

    E_fix_all,E_fix,Psi_fix = solve.cp(V_fine,moments)
    E_rel_all,E_rel,Psi_rel = solve.cp(V_fine,reduced_moments)

    print('---------------------------------------------------------------------')
    print('Fundamental torsional frequencies (cm-1)')
    print('---------------------------------------------------------------------')
    print('Assuming constant moment of inertia, fixed base (infinite mass):',(E_fix[1]-E_fix[0])*c.HA_TO_CM)
    print('Assuming constant moment of inertia, moving base:',(E_rel[1]-E_rel[0])*c.HA_TO_CM)

    np.savetxt('E_all_const_I_fixed_base.txt',E_fix_all)
    np.savetxt('E_all_const_I_moving_base.txt',E_rel_all)
    np.savetxt('E_const_I_fixed_base.txt',E_fix)
    np.savetxt('E_const_I_moving_base.txt',E_rel)
    np.savetxt('Wfn_const_I_fixed_base.csv',Psi_fix,delimiter=',')
    np.savetxt('Wfn_const_I_moving_base.csv',Psi_rel,delimiter=',')

    # Perform calculations with variable moment of inertia (if multiple geoms provided)

    if len(geometries) > 1:

      I_coarse = []; I_red_coarse = []
      for geometry in geometries:
        moment,reduced_moment = geom.compute_moments(geometry)
        I_coarse.append(moment)
        I_red_coarse.append(reduced_moment)

      fI  = interpSpl(phi_coarse, I_coarse, k=3)
      fIr = interpSpl(phi_coarse, I_red_coarse, k=3)
      I_fine = fI(phi_fine)
      I_red_fine = fIr(phi_fine)

    #  np.savetxt('I_fixed_base.txt',np.array(moments))
    #  np.savetxt('I_moving_base.txt',np.array(reduced_moments))

      E_fix_var,Psi_fix_var = solve.cp(V_fine,I_fine)
      E_rel_var,Psi_rel_var = solve.cp(V_fine,I_red_fine)

      print('---------------------------------------------------------------------')
      print('Fundamental torsional frequencies (cm-1)')
      print('---------------------------------------------------------------------')
      print('Variable moment of inertia, fixed base (infinite mass):',(E_fix_var[1]-E_fix_var[0])*c.HA_TO_CM)
      print('Variable moment of inertia, moving base:',(E_rel_var[1]-E_rel_var[0])*c.HA_TO_CM)

      np.savetxt('E_var_I_fixed_base.txt',E_fix_var)
      np.savetxt('E_var_I_moving_base.txt',E_rel_var)
      np.savetxt('Wfn_var_I_fixed_base.csv',Psi_fix_var,delimiter=',')
      np.savetxt('Wfn_var_I_moving_base.csv',Psi_rel_var,delimiter=',')

    print('---------------------------------------------------------------------')
    print('Note: all energy levels printed to file are in atomic units          ')
    print('      and computed relative to the global minimum                    ')
    print('---------------------------------------------------------------------')
