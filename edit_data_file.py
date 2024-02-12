#!/usr/bin/env python
# coding: utf-8

# In[1]:


# redistribute tractors and create new displacement vectors
import numpy as np
gap = 24
no_of_atoms = 88461

tractor_lo, tractor_hi = 88262, 88461
no_of_tractors = tractor_hi - tractor_lo + 1

b1 = 200
tr = 20
alpha = 50000
cell_position = [b1/2, b1/2, b1/2]

xcomp_t = open('xcomp.txt', 'w')
ycomp_t = open('ycomp.txt', 'w')
zcomp_t = open('zcomp.txt', 'w')

xcomp_t.write(f'{no_of_tractors}\n')
ycomp_t.write(f'{no_of_tractors}\n')
zcomp_t.write(f'{no_of_tractors}\n')

f = open('data_from_sim', 'r')
list_of_lines = f.readlines()
#print(list_of_lines[199])
f.close()

new_file = open('data_from_sim_edited.txt', 'w')
n = 1
for ind, line in enumerate(list_of_lines):
    #print(ind)
    if gap < ind < no_of_atoms+gap:
        elem_list = line.split(' ')
        atom_no = int(elem_list[0])
        if tractor_lo <= atom_no <= tractor_hi:
            #print(elem_list)
            theta, phi =  np.pi * np.random.rand(), 2 * np.pi * np.random.rand()
            x = tr*np.sin(theta) * np.cos(phi)
            y = tr*np.sin(theta) * np.sin(phi)
            z = tr*np.cos(theta)
            new_point = cell_position + np.array([x, y ,z])
            x, y, z = new_point[0], new_point[1], new_point[2]
            elem_list[4], elem_list[5], elem_list[6] = str(x), str(y), str(z) # find the positions here
            line = ' '.join(elem_list)
            print(x, y ,z)
            dx, dy, dz = alpha*(cell_position[0] - x)/tr, alpha*(cell_position[1] - y)/tr, alpha*(cell_position[2] - z)/tr
            xcomp_t.write('{0} {1}\n'.format(int(atom_no), dx))
            ycomp_t.write('{0} {1}\n'.format(int(atom_no), dy))
            zcomp_t.write('{0} {1}\n'.format(int(atom_no), dz))
    
    print(line)
    new_file.write(f'{line}')
    print(n)
    n = n+1
    
#new_file.writelines(big_string0)
new_file.close()
xcomp_t.close()
ycomp_t.close()
zcomp_t.close()


# In[ ]:




