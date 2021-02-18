import numpy as np

from skimage.feature import peak_local_max

import os


class CHs_Distribution(object):

	def __init__(self,path,ch_min,ch_max):


		self.path = os.path.join(path,'data/')

		self.ch_min = ch_min

		self.ch_max = ch_max

	def get_peaks_pos(self,image):

		peaks_pos = peak_local_max(image, min_distance = 1, threshold_abs = 1e-6)

		return peaks_pos

	def get_CHs(self,peaks_pos,lbl):


		CHs = np.round(lbl[peaks_pos[:, 0], peaks_pos[:, 1]])

		return CHs

	def get_CHs_distribution(self):


		self.num_data = len(os.listdir(self.path))


		absolute_ch_distr = np.zeros([self.ch_max - self.ch_min + 1, self.num_data])
		relative_ch_distr = np.zeros([self.ch_max - self.ch_min + 1, self.num_data])

		for data_index,data_file in enumerate(os.listdir(self.path)):

			data = np.load(os.path.join(self.path,data_file), allow_pickle=True,fix_imports=True)

			img = data[0,:,:,0]

			lbl = data[0,:,:,1]

			peaks_pos = self.get_peaks_pos(lbl)

			CHs = self.get_CHs(peaks_pos,lbl)

			n_columns = len(CHs)

			for ch in range(self.ch_min, self.ch_max + 1):

				n_ch = len(np.where(CHs == ch)[0])

				absolute_ch_distr[ch, data_index] = n_ch
				relative_ch_distr[ch, data_index] = np.round(n_ch/n_columns, decimals = 3)

		return absolute_ch_distr, relative_ch_distr


class I_CHs_Correlation(object):

	def __init__(self,path):

		self.path = os.path.join(path,'data/')


	def get_peaks_pos(self,labels):

		peaks_pos = peak_local_max(labels, min_distance = 1, threshold_abs = 1e-6)

		return peaks_pos

	def get_I_CHs_correlation(self):

		self.I_ch_correlation = []

		for data_index, data_file in enumerate(os.listdir(self.path)):

			data = np.load(os.path.join(self.path,data_file))

			img =  data[0, :, :, 0]

			lbl = data[0, :, :, 1]

			peaks_pos = self.get_peaks_pos(lbl)

			I_ch_correlation_single_data = np.zeros((len(peaks_pos),2))

		
			for peaks_index,peaks_pos in enumerate(peaks_pos):

				ch = np.round(lbl[peaks_pos[0],peaks_pos[1]])

				I = img[peaks_pos[0],peaks_pos[1]]
                
				I_ch_correlation_single_data[peaks_index, 0] = ch
				I_ch_correlation_single_data[peaks_index,1] = I



			self.I_ch_correlation.append(I_ch_correlation_single_data)


		return self.I_ch_correlation



class I_CHs_Correlation_Prediction(object):

	def __init__(self, img, pred, n_elements=5):

		self.img = img[0, :, :, 0]
		self.pred = pred[0]

		self.n_elements = n_elements

	def get_peaks_pos(self, labels_single_element):

		peaks_pos = peak_local_max(labels_single_element, min_distance=1, threshold_abs=1e-6)

		return peaks_pos

	def get_I_CHs_correlation(self):

		self.I_absolute_ch_correlation = []

		self.I_relative_ch_correlation = []

		pred_all_elements = self.pred[:, :, 0] 
        
		peaks_pos_all_elements = self.get_peaks_pos(pred_all_elements)

		I_absolute_ch_correlation_single_data = np.zeros((self.n_elements, len(peaks_pos_all_elements), 2))

		I_relative_ch_correlation_single_data = np.zeros((self.n_elements, len(peaks_pos_all_elements), 2))

		for peaks_index, peaks_pos in enumerate(peaks_pos_all_elements):

			ch_tot = np.round(pred_all_elements[peaks_pos[0], peaks_pos[1]])

			I = self.img[peaks_pos[0], peaks_pos[1]]

			for ce in range(self.n_elements):
				ch = np.round(self.pred[peaks_pos[0], peaks_pos[1], ce])

				I_absolute_ch_correlation_single_data[ce, peaks_index, 0] = ch 
				I_absolute_ch_correlation_single_data[ce, peaks_index, 1] = I

				I_relative_ch_correlation_single_data[ce, peaks_index, 0] = ch / ch_tot
				I_relative_ch_correlation_single_data[ce, peaks_index, 1] = I

		self.I_absolute_ch_correlation.append(I_absolute_ch_correlation_single_data)
		self.I_relative_ch_correlation.append(I_relative_ch_correlation_single_data)

		return self.I_absolute_ch_correlation, self.I_relative_ch_correlation






































