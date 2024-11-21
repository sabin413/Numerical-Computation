### This is a collection of 9 utility functions needed to process calculations in between the automated simulations. The main code will import this file as a python module.
### Writing a class is not a good idea here as the functions are not all related

import numpy as np
from numpy.linalg import norm
import matplotlib.pyplot as plt
from scipy.integrate import dblquad
from scipy.spatial import distance

# function1: a function that reads the trajectory file line by line, extracts numerical data, and calculates the bond vectors

def get_data(data_path, r_inner = 10, r_outer = 50):   
    
    '''reads data, and outputs the bond vectors'''
    
    cell_position = np.array([100, 100, 100])
    
    f = open(data_path,'r')
    big_string = f.read()
    f.close()
    big_string_splitted = big_string.split("Bonds")
    bonds_data = big_string_splitted[1]

    pos_and_vel_data = big_string_splitted[0] # atom coordinates and velocities
    
    big_string_splitted = bonds_data.split("Angles")
    bonds_data = big_string_splitted[0]

    bonds_data.strip() # removes both - trailing and preceeding (right and left) empty strings
    bonds_data = bonds_data.split('\n')
    #print(bonds_data[-10:])
    bonds_data = bonds_data[2:-2]
    print(len(bonds_data))

    #print(bonds_data[:10])
    bond_pairs_data = []
    for i in bonds_data:
        lst0 = i.split()
        lst1 = [int(j) for j in lst0]
        #print(lst0)
        if lst1[1] == 1:           # correct it later
            bond_pairs_data.append([lst1[2], lst1[3]]) # bond pairs
            
    atoms_data = pos_and_vel_data.split("Velocities")
    atoms_data = atoms_data[0]
    atoms_string_splitted = atoms_data.split("Atoms # full")
    #print(len(atoms_string_splitted))
    atoms_data_string = atoms_string_splitted[1]

    atoms_data_string.strip() # removes both - trailing and preceeding (right and left) empty strings
    atoms_data_lines = atoms_data_string.split('\n')
    #print(atoms_data_lines[:10])
    atoms_data = atoms_data_lines[2:-2]
    print(len(atoms_data))

    #print(bonds_data[:10])
    atoms_position_data = []
    for i in atoms_data:
        lst0 = i.split()
        lst1 = [float(j) for j in lst0]
        lst_xyz = [lst1[0], lst1[-6], lst1[-5], lst1[-4]] # atom no, x, y, z
        atoms_position_data.append(lst_xyz)
    atoms_position_data.sort(key=lambda x:x[0])
    
    
    bond_vectors_list = []#np.empty((len(bond_pairs_data), 3))

    for index, elem in enumerate(bond_pairs_data):
        atom1, atom2 = elem[0], elem[1] # atom numbers
        data1, data2 = atoms_position_data[atom1-1], atoms_position_data[atom2-1] # indexing starts from 0 and 1 in 
        #python and lammps, resp
        #print(data1, data2)
        pos1, pos2 = np.array(data1[1:]), np.array(data2[1:])
        #print(norm(pos1 - cell_position))
        bond_vec = np.subtract(pos2, pos1) # subtract pos1 from pos2
        #print(bond_vec)
        if (r_inner <= norm(pos1 - cell_position) <= r_outer) and (r_inner <= norm(pos2 - cell_position) <= r_outer):
            #bond_vectors_array[index] = bond_vec
            bond_vectors_list.append(bond_vec)    
            
    bond_vectors = np.array(bond_vectors_list) # all the inner bond vectors
    
    return bond_vectors

# function2: calculate the alignment tensor

def alignment_tensor(vec_arr):
    
    norm_array = np.array([norm(elem) for elem in vec_arr])
    norm_array_reshaped = norm_array.reshape(-1,1)
    vec_arr_normalized = vec_arr / norm_array_reshaped
    norm_list = [norm(i) for i in vec_arr_normalized]
    #print(vec_arr, norm_array, vec_arr_normalized, norm_list)
    
    x, y, z = vec_arr_normalized[:,0], vec_arr_normalized[:,1], vec_arr_normalized[:,2]
    
    x_sq, y_sq, z_sq = x**2, y**2, z**2
    xy, yz, xz = x*y, y*z, x*z
    
    Qxx = np.average(norm_array*x_sq)
    Qyy = np.average(norm_array*y_sq)
    Qzz = np.average(norm_array*z_sq)
    Qxy = np.average(norm_array*xy)
    Qxz = np.average(norm_array*xz)
    Qyz = np.average(norm_array*yz)
    
    Q = np.array(( (Qxx, Qxy, Qxz), (Qxy, Qyy, Qyz), (Qxz, Qyz, Qzz) ))
    w, v = np.linalg.eigh(Q) # eigen value, eigen vec
    
    w_v_pair = [(abs(w[i]), v[:,i]) for i in range(3)]
    w_v_pair.sort(key = lambda x: x[0]) # ascending order
    
    return w_v_pair

### Now find cell bead positions
# parametric eqns for an ellipsoid 
# functions 3:

def sin(x):
    return np.sin(x)

def cos(x):
    return np.cos(x)

def x_cor(a, theta, phi):
    return a*sin(theta)*cos(phi)

def y_cor(b, theta, phi):
    return b*sin(theta)*sin(phi)

def z_cor(c, theta, phi):
    return c*cos(theta)

# function 4: see docstring

def p(theta, phi, a, b, c):
    '''Calculates the unnormalized probability of choosing a given point (theta, phi)
    at the surface of an ellipsoid characterized by axes a, b, c'''

    term_1 = (a**2 * (cos(theta) * cos(phi))**2 + b**2 * (cos(theta) * sin(phi))**2 + c**2 * (sin(theta))**2)**0.5

    term_2 = (a**2 * (sin(theta) * sin(phi))**2 + b**2 * (sin(theta) * cos(phi))**2 )**0.5

    return term_1 * term_2

# function 5: do sampling

def metropolis_hastings_sampling(pdf, num_samples=1, num_iter=5000, proposal_std=0.5, x_range=(0, np.pi), y_range=(0, 2*np.pi)):
    samples = []
    x, y = np.pi / 2, np.pi  # Initial point for Metropolis-Hastings within specified ranges
    
    for i in range(num_iter):
        x_new, y_new = np.random.normal([x, y], proposal_std)

        # Check if proposed point is within specified x and y ranges
        if x_range[0] <= x_new <= x_range[1] and y_range[0] <= y_new <= y_range[1]:
            pdf_x_y = pdf(x, y)
            pdf_x_new_y_new = pdf(x_new, y_new)

            if pdf_x_y > 0 and pdf_x_new_y_new > 0:  # Check for non-zero probability densities
                acceptance_ratio = pdf_x_new_y_new / pdf_x_y
                if acceptance_ratio >= np.random.rand():
                    x, y = x_new, y_new
                    if i >= num_iter - num_samples:
                        samples.append([x, y])

    return samples

# function 6: find minimum distance

def min_distance(p1, lis):
    if lis == []:
        return 1e12
    distance_list = [distance.euclidean(p1, p2) for p2 in lis]
    # print(np.min(distance_list))
    return np.min(distance_list)

# function 7: see docstring

def create_tractor_beads(a, b, c, no_of_points,  bead_diameter = 1, overlap_threshold = 1.0):
    '''creates tractor beads on the surface of an ellipsoid '''
          
    final_list = []
    
    custom_pdf = lambda theta, phi: p(theta, phi, a,  b,  c)
    
    for i in range(no_of_points):
       # print(i)
        elem = []
        while elem == []:
            elem = metropolis_hastings_sampling(custom_pdf)

        theta, phi = elem[0][0], elem[0][1]
        point = [x_cor(a, theta, phi), y_cor(b, theta, phi), z_cor(c, theta, phi)]

        while min_distance(point, final_list) < overlap_threshold:

            elem = []
            while elem == []:
                elem = metropolis_hastings_sampling(custom_pdf)

            theta, phi = elem[0][0], elem[0][1]
            point = [x_cor(a, theta, phi), y_cor(b, theta, phi), z_cor(c, theta, phi)]


        final_list.append(point)
    
    return np.array(final_list)

# function 9: see docstring

def create_cell_beads(a, b, c, center, coverage = 1, bead_diameter = 1, overlap_threshold = 0.5):
    '''creates beads on the cell surface - this code works even when the cell is ellipsoidal'''
    
    # surface concentration of beads
       
    final_list = []
  
    surface_area = 4/3*np.pi*(a*b + b*c + a*c)

    no_of_points = int(coverage*surface_area / bead_diameter**2)


    custom_pdf = lambda theta, phi: p(theta, phi, a,  b,  c)
    
    for i in range(no_of_points):
        #print(i)
        elem = []
        while elem == []:
            elem = metropolis_hastings_sampling(custom_pdf)

        theta, phi = elem[0][0], elem[0][1]
        point = [x_cor(a, theta, phi), y_cor(b, theta, phi), z_cor(c, theta, phi)]

        while min_distance(point, final_list) < overlap_threshold:

            elem = []
            while elem == []:
                elem = metropolis_hastings_sampling(custom_pdf)

            theta, phi = elem[0][0], elem[0][1]
           
            point = np.array([x_cor(a, theta, phi), y_cor(b, theta, phi), z_cor(c, theta, phi)])
            point = np.add(point, center)

        final_list.append(point)
    
    return np.array(final_list)
