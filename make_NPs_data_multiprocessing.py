import numpy as np

from make_NPs_data_utils import Random_NP, NP_HRTEM, NP_Labels, NP_Data

from ase.visualize import view

from pyqstem import PyQSTEM

import matplotlib.pyplot as plt
from mpl_toolkits.axes_grid1 import make_axes_locatable

import platform
import multiprocessing as mp

import random

import argparse



print("Running on host '{}'".format(platform.node()))

n_processors = mp.cpu_count()
num_data = n_processors

print("Number of available processors: ", n_processors)
print("Number of used processors: ", num_data)


path = 'test_data/'
first_number = 1
num_data_all = 8000

# test
#path = 'test_data/'
#first_number = 1
#num_data_all = 2000

crystal_structure = 'fcc'

material = 'Au'

lc = 4.065

spatial_domain = (51.2, 51.2)  # A

qstem = PyQSTEM('TEM')

image_size = (256, 256)  # px

resolution = spatial_domain[0] / image_size[0]  # [A/px]


def HRTEM_multiprocessing(data_index):

    print('Processing data [{}/{}]'.format(data_index, num_data_all))

    random_size = random.uniform(10, 20)  # A

    random_NP = Random_NP(crystal_structure, material, lc, random_size, spatial_domain)

    random_NP_model = random_NP.get_model()

    random_Cs = random.uniform(-15e4, 15e4)

    random_defocus = random.uniform(180,200)

    random_focal_spread = random.uniform(20, 40)

    random_a22 = random.uniform(0, 50)

    random_phi22 = random.uniform(0, 2 * np.pi)

    random_aberrations = {'a22': random_a22, 'phi22': random_phi22}

    random_blur = random.uniform(0, 1)

    random_dose = random.uniform(1e2, 1e3)

    random_c1 = random.uniform(0.95, 1)

    random_c2 = random.uniform(0, 1e-1)

    random_c3 = random.uniform(5e-1, 6e-1)

    random_c4 = random.uniform(2, 3)

    random_MTF_param = [random_c1, random_c2, random_c3, random_c4]

    NP_hrtem = NP_HRTEM(qstem,
                        random_NP_model,
                        image_size,
                        resolution,
                        random_Cs,
                        random_defocus,
                        random_focal_spread,
                        random_aberrations,
                        random_blur,
                        random_dose,
                        random_MTF_param)

    img = NP_hrtem.get_HRTEM_img()

    NP_labels = NP_Labels(random_NP_model,
                          image_size,
                          resolution)

    chs_lbl = NP_labels.get_CHs_labels()

    NP_data = NP_Data(random_NP_model,
                      img,
                      chs_lbl,
                      path,
                      data_index)

    NP_data.save_NP()

if __name__ == '__main__':

    for counter in range(2):

        first_number = num_data * counter + 1

        pool = mp.Pool(n_processors)
        pool.map(HRTEM_multiprocessing, [data_index for data_index in range(first_number,num_data + first_number)])
        pool.close()

