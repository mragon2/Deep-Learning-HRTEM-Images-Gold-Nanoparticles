import numpy as np

import random

from ase.build import bulk
from ase import Atoms
from ase.io import write

from pyqstem.imaging import CTF

from skimage.filters import gaussian
from skimage.feature import peak_local_max

from scipy.spatial.distance import cdist

import matplotlib.pyplot as plt
from mpl_toolkits.axes_grid1 import make_axes_locatable

import os


class Random_NP(object):

    directions = np.array(

        [[1, 0, 0], [0, 1, 0], [0, 0, 1], [-1, 0, 0],
         [0, -1, 0], [0, 0, -1], [1, 1, 1], [-1, 1, 1],
         [1, -1, 1], [1, 1, -1], [-1, -1, 1], [-1, 1, -1],
         [1, -1, -1], [-1, -1, -1]]
    )

    directions = (directions.T / np.linalg.norm(directions, axis=1)).T

    def __init__(self,crystal_structure,material, lc, random_size, spatial_domain):

        self.material = material

        self.lc = lc

        self.crystal_structure = crystal_structure

        self.random_size = random_size

        self.spatial_domain = spatial_domain

        self.sites = self.get_sites()

        self.bonds = self.get_bonds()

    def get_sites(self):

        grid_size = 22

        cluster = bulk(self.material, self.crystal_structure, a = self.lc, cubic = True)

        cluster *= (grid_size,) * 3

        cluster.center()

        self.center = np.diag(cluster.get_cell()) / 2

        positions = cluster.get_positions()

        return positions

    def get_bonds(self):

        bond_length = 3.91 / np.sqrt(2)

        bonds = []

        for i, s in enumerate(self.sites):

            distances = np.linalg.norm(self.sites - s, axis=1)

            indices = np.where(distances <= bond_length * 1.05)[0]

            bonds.append(indices)

        return bonds

    def create_seed(self, lengths100, lengths111):

        self.active = np.ones(len(self.sites), dtype=bool)

        lengths = np.hstack((lengths100, lengths111))

        for length, direction in zip(lengths, self.directions):

            r0 = self.center + length * direction

            for i, site in enumerate(self.sites):

                if self.active[i]:

                    self.active[i] = np.sign(np.dot(direction, site - r0)) == -1

        self.active_bonds = np.array([self.active[b] for b in self.bonds],dtype = object)

        self.available_sites = np.where([any(ab) & (not a) for ab, a in zip(self.active_bonds, self.active)])[0]

    def build(self, grid_size, T0, T1=None):

        if T1 is None:

            T1 = T0

        for i in range(grid_size):

            T = T0 + (T1 - T0) * i / grid_size

            coordination = self.get_coordination(self.available_sites)

            p = np.zeros_like(coordination, dtype=np.float)

            p[coordination > 2] = np.exp(coordination[coordination > 2] / T)

            p = p / float(np.sum(p))

            p[p < 0] = 0

            n = np.random.choice(len(p), p=p)

            k = self.available_sites[n]

            self.available_sites = np.delete(self.available_sites, n)

            self.expand(k)

    def expand(self, k):

        self.active[k] = True

        new_avail = self.bonds[k][self.active[self.bonds[k]] == 0]

        self.available_sites = np.array(list(set(np.append(self.available_sites, new_avail))))

        if len(new_avail) > 0:

            to_update = np.array([np.where(self.bonds[x] == k)[0] for x in new_avail]).T[0]

            for i, j in enumerate(to_update):

                self.active_bonds[new_avail][i][j] = True

    def get_coordination(self, sites):

        return np.array([sum(self.active_bonds[site]) for site in sites])

    def get_cluster(self):

        return Atoms([self.material] * len(self.sites[self.active]), self.sites[self.active])

    def get_model(self):

        radius = self.random_size/2

        lengths100 = np.random.uniform(radius, radius + .2 * radius, 6)

        lengths111 = np.random.uniform(radius, radius + .2 * radius, 8)

        self.create_seed(lengths100, lengths111)

        self.build(int(np.sum(self.active) / 4.), 10, 2)

        self.model = self.get_cluster()


        self.model.rotate(v='y', a=45, center='COP')

        self.model.rotate(v='z', a=random.random() * 360, center='COP')

        self.model.center(vacuum=0)

        cell = self.model.get_cell()

        size = np.diag(cell)

        self.model.set_cell((self.spatial_domain[0],) * 3)

        self.model.center()

        t = 1 / 6

        tx = random.uniform(-self.spatial_domain[0] * t, self.spatial_domain[0] * t)

        ty = random.uniform(-self.spatial_domain[1] * t, self.spatial_domain[1] * t)

        self.model.translate((tx, ty, 0))

        return self.model


class NP_HRTEM(object):

    def __init__(self,qstem,NP_model,image_size, resolution, Cs, defocus, focal_spread,aberrations,blur,dose,MTF_param):

        self.qstem = qstem

        self.NP_model = NP_model

        self.image_size = image_size

        self.resolution = resolution

        self.Cs = Cs

        self.defocus = defocus

        self.focal_spread = focal_spread

        self.aberrations = aberrations

        self.blur = blur

        self.dose = dose

        self.MTF_param = MTF_param


    def get_HRTEM_img(self):

        self.qstem.set_atoms(self.NP_model)

        self.wave_size = (int(self.NP_model.get_cell()[0,0]/self.resolution),int(self.NP_model.get_cell()[1,1]/self.resolution))

        self.qstem.build_wave('plane', 300, self.wave_size)
        self.qstem.build_potential(int(self.NP_model.get_cell()[2, 2] * 2))
        self.qstem.run()

        wave = self.qstem.get_wave()

        wave.array = wave.array.astype(np.complex64)

        self.ctf = CTF(defocus = self.defocus,
                       Cs = self.Cs,
                       focal_spread = self.focal_spread,
                       aberrations = self.aberrations)

        self.img = wave.apply_ctf(self.ctf).detect(resample = self.resolution,
                                              blur = self.blur,
                                              dose = self.dose,
                                              MTF_param = self.MTF_param)

        return self.img


    def get_local_normalization(self):

        self.img = (self.img - np.min(self.img)) / (np.max(self.img) - np.min(self.img))

        self.img = self.img - gaussian(self.img, 12 / self.resolution)

        self.img = self.img / np.sqrt(gaussian(self.img ** 2, 12 / self.resolution))

        return self.img

    def get_NP_hrtem(self):

        self.get_HRTEM_img()

        self.get_local_normalization()

        return self.img

class NP_Labels(object):

    def __init__(self,NP_model,image_size,resolution):

        self.NP_model = NP_model

        self.image_size = image_size

        self.resolution = resolution

    def get_CHs_labels(self):

        positions = self.NP_model.get_positions()[:,:2]/self.resolution

        spotsize = 0.52

        width = int(spotsize/self.resolution)

        x, y = np.mgrid[0:self.image_size[0], 0:self.image_size[1]]

        labels = np.zeros(self.image_size)

        for p in (positions):

            p_round = np.round(p).astype(int)

            min_xi = np.max((p_round[0] - width * 4, 0))
            max_xi = np.min((p_round[0] + width * 4 + 1, self.image_size[0]))
            min_yi = np.max((p_round[1] - width * 4, 0))
            max_yi = np.min((p_round[1] + width * 4 + 1, self.image_size[1]))

            xi = x[min_xi:max_xi, min_yi:max_yi]
            yi = y[min_xi:max_xi, min_yi:max_yi]

            v = np.array([xi.ravel(), yi.ravel()])

            labels[xi, yi] += np.exp(-cdist([p], v.T) ** 2 / (2 * width ** 2)).reshape(xi.shape)

        return labels

class NP_Data(object):

    def __init__(self,model,img,lbl,path,data_index):

        self.model = model

        self.img = img

        self.lbl = lbl

        self.path = path

        self.data_index = data_index

    def save_NP_model(self):

        if(self.path + 'models/') and not os.path.exists(self.path + 'models/'):
            os.makedirs(self.path + 'models/')

        if self.data_index < 50:

            write(self.path + 'models/NP_model_{}.xyz'.format(self.data_index), self.model)


    def save_NP_data(self):

        self.img = self.img.reshape((1,) + self.img.shape + (1,))

        self.lbl = self.lbl.reshape((1,) + self.lbl.shape + (1,))

        self.data = np.concatenate([self.img,self.lbl], axis = 3)

        if (self.path + 'data/') and not os.path.exists(self.path + 'data/'):
            os.makedirs(self.path + 'data/')

        np.save(self.path + 'data/data_{}.npy'.format(self.data_index),self.data)

        return self.data

    def save_NP_plot(self):

        fig = plt.figure(figsize=(14, 7))
        ax = fig.add_subplot(1, 2, 1)

        im = ax.imshow(self.img[0, :, :, 0], cmap='gray')
        plt.title('HRTEM image', fontsize=20)
        divider = make_axes_locatable(ax)
        cax1 = divider.append_axes("right", size="5%", pad=0.05)
        cbar = plt.colorbar(im, cax=cax1)

        ax = fig.add_subplot(1, 2, 2)
        im = ax.imshow(self.lbl[0, :, :, 0], cmap='jet')
        plt.title('CHs labels',fontsize = 20)
        divider = make_axes_locatable(ax)
        cax2 = divider.append_axes("right", size="5%", pad=0.05)
        cbar = plt.colorbar(im, cax=cax2)

        plt.tight_layout()

        if (self.path + 'plots/') and not os.path.exists(self.path + 'plots/'):
            os.makedirs(self.path + 'plots/')

        if self.data_index < 50:

            fig.savefig(self.path + 'plots/data_{}.png'.format(self.data_index), bbox_inches='tight')

        plt.close(fig)

    def save_NP(self):

        self.save_NP_model()

        self.save_NP_data()

        self.save_NP_plot()






















