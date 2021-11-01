import numpy as np
from constants import mass_lib
from constants import ANGST_TO_BOHR as a2b

def parse(xyzfile):

  with open(xyzfile,'r') as f: fc = f.readlines()
   
  xyz = []; geoms = []
  for line in fc:
    if len(line.split()) < 4:
      if xyz != []:
        geoms.append(xyz) 
        xyz = []
    else:
      xyz.append(line)
  if xyz != []:
    geoms.append(xyz)

  return geoms

def get_masses_coords(xyz_split,line_length):

  masses = [mass_lib[line[0].upper()] for line in xyz_split if len(line) == line_length]
  coords = [list(map(float,line[1:4])) for line in xyz_split if len(line) == line_length]
  coords_bohr = [[crd*a2b for crd in line] for line in coords]

  return masses,coords_bohr

def distance_matrix(A,B):

  distances = np.sum(A**2,axis=1) + np.sum(B**2,axis=1)[:,np.newaxis] - 2*np.dot(B,A.T)
  
  return distances
  
def compute_moments(xyz):

  # Extract coordinates and masses in atomic units

  xyz_split = [line.split() for line in xyz]
  masses_base,coords_base = get_masses_coords(xyz_split,4)
  masses_rotor,coords_rotor = get_masses_coords(xyz_split,5)
  masses = masses_rotor + masses_base
  coords = coords_rotor + coords_base
  masses3 = []
  for mass in masses:
    masses3 += [mass,mass,mass]
  coords_rotor = np.asarray(coords_rotor)
  coords_base = np.asarray(coords_base)
  coords = np.asarray(coords)
  
  # Set up some useful parameters

  n_atom = len(masses)
  n_rot = len(masses_rotor)
  n_dim = len(masses3)

  # Find axis of rotation as closest pair of atoms between base and rotor

  distances = distance_matrix(coords_base,coords_rotor)
  bonded_atoms = np.unravel_index(np.argmin(distances, axis=None), distances.shape)
  rotation_centre = coords_base[bonded_atoms[0],:]
  rotation_axis = coords_rotor[bonded_atoms[1],:] - rotation_centre
  rotation_vector = rotation_axis/np.linalg.norm(rotation_axis)

  # Construct basis for global translations and infinitesimal rotations

  basis = np.zeros((6, 3*n_atom))
  basis[0, 0::3] = 1
  basis[1, 1::3] = 1
  basis[2, 2::3] = 1
  basis[3, 1::3] = np.transpose( coords[:,2])
  basis[3, 2::3] = np.transpose(-coords[:,1])
  basis[4, 2::3] = np.transpose( coords[:,0])
  basis[4, 0::3] = np.transpose(-coords[:,2])
  basis[5, 0::3] = np.transpose( coords[:,1])
  basis[5, 1::3] = np.transpose(-coords[:,0])

  # Construct vectors perpendicular to axis of rotation for each rotating atom 

  rot_tangent = np.zeros((n_dim))
  for i in range(0,n_rot):
    rot_tangent[3*i:3*i+3] = np.cross(coords[i,:]-rotation_centre,rotation_vector)
  
  # Do projection and compute moments of inertia

  m3 = np.asarray(masses3)
  mwbasis = basis*np.asarray(m3)
  A = np.dot(mwbasis, basis.T) 
  B = np.dot(mwbasis, rot_tangent)
  alphas = np.linalg.solve(A,B)
  rot_tangent_rel = rot_tangent - np.dot(alphas,basis)
  moment = (m3*rot_tangent**2).sum()
  reduced_moment = (m3*rot_tangent_rel**2).sum()

  return moment, reduced_moment 
