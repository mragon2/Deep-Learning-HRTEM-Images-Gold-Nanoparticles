
import numpy as np

import torch
import torchvision
import torch.nn as nn

from fcn import FCN

import os
from glob import glob

import platform

from skimage.feature import peak_local_max
from skimage.filters import gaussian

from sklearn.metrics import r2_score

import matplotlib.pyplot as plt
from mpl_toolkits.axes_grid1 import make_axes_locatable

def print_devices():

    print('Running on host {}'.format(platform.node()))

    if torch.cuda.is_available():

        num_devices = torch.cuda.device_count()

        if num_devices == 1:

            print('Running on 1 GPU')

        else:

            print('Running on {} GPUs'.format(num_devices))

    else:

        print('Running on 1 CPU')

def make_folder(path):
    if path and not os.path.exists(path):
        os.makedirs(path)

def make_results_folder(path,train = True):

    make_folder(path)

    debug_path = os.path.join(path, 'debug/')
    make_folder(debug_path)

    if train:

        weights_path = os.path.join(path, 'weights/')
        make_folder(weights_path)

    learning_curve_path = os.path.join(path, 'learning_curve/')
    make_folder(learning_curve_path)

def make_dataset(path,batch_size):

    dataset = np.sort(np.array(glob(os.path.join(path,'data*.npy'))))

    num_batches = len(dataset) // batch_size

    dataset = torch.utils.data.DataLoader(dataset,shuffle = True, batch_size = batch_size)

    return dataset,num_batches


def make_dataset_hvd(path, batch_size, hvd):


    dataset = np.sort(np.array(glob(os.path.join(path, 'data*.npy'))))

    num_batches = len(dataset) // batch_size

    data_sampler = torch.utils.data.distributed.DistributedSampler(dataset,
                                                                   num_replicas = hvd.size(),
                                                                   rank = hvd.rank())

    dataset = torch.utils.data.DataLoader(dataset, shuffle = False,
                                          batch_size = batch_size,
                                          sampler = data_sampler)


    return dataset,num_batches

def make_batch(batch):

    batch = np.array(batch)

    batch_images = []
    batch_labels = []

    for i in range(len(batch)):

        data = np.load(batch[i])

        img = data[0,:,:,0]
        img = img.reshape(1,1,img.shape[0],img.shape[1])

        lbl = data[0,:,:,1]
        lbl_torch = lbl.reshape(1,1,lbl.shape[0],lbl.shape[1])


        rnd_imgng = Random_Imaging(image=img,labels=lbl_torch)
        img,lbl_torch = rnd_imgng.get_trasform()

        batch_images.append(img)
        batch_labels.append(lbl_torch)

    batch_images = torch.Tensor(np.concatenate(batch_images))

    batch_labels = torch.Tensor(np.concatenate(batch_labels))

    if torch.cuda.is_available():

        batch_images = batch_images.cuda()

        batch_labels = batch_labels.cuda()


    return batch_images,batch_labels

def make_model():

    model = FCN()

    if torch.cuda.is_available():

        model.to('cuda')

        model = nn.DataParallel(model)

    return model

def detatch(x,mp):

    x = x.detach().cpu().numpy()

    if mp:

        x = x.astype('float32')

    return x


def train_step(batch,model,mp,criterion,optimizer):

    batch_images,batch_labels  = batch

    model.zero_grad()

    if mp:

        loss_scaler = torch.cuda.amp.GradScaler()

        with torch.cuda.amp.autocast():

            batch_predictions = model(batch_images)

            train_loss = criterion(batch_predictions,batch_labels)

            loss_scaler.scale(train_loss).backward()

            loss_scaler.step(optimizer)

            loss_scaler.update()

    else:

        batch_predictions = model(batch_images)

        train_loss = criterion(batch_predictions,batch_labels)

        train_loss.backward()

        optimizer.step()

    batch_predictions = detatch(batch_predictions, mp)

    if torch.cuda.is_available():

        torch.cuda.empty_cache()

    return train_loss,batch_predictions

def test_step(batch,model,mp,criterion):

    batch_images,batch_labels  = batch

    with torch.no_grad():

        if mp:

            with torch.cuda.amp.autocast():

                batch_predictions = model(batch_images)

        else:

            batch_predictions = model(batch_images)

    test_loss = criterion(batch_predictions,batch_labels)

    batch_predictions = detatch(batch_predictions, mp)

    if torch.cuda.is_available():

        torch.cuda.empty_cache()

    return test_loss,batch_predictions

class R2_CHs(object):

    """
    * R2_CHs: class to calculate the regression metric R^2 between the predicted CHs
    and the true CHs for each chemical element. Input parameters:

    - batch_predictions: batch of predictions maps from the FCN (batch_size, 256,256,5).
    - batch_labels: batch of ground truth maps (batch_size, 256,256,5).
    - num_chemical_elements: number of chemical_elements and output channels of the predictions and labels.

    """

    def __init__(self,batch_predictions,batch_labels):

        self.batch_predictions = np.array(batch_predictions)

        self.batch_labels = np.array(batch_labels)


    def get_peaks_pos(self,target):

        peaks_pos = peak_local_max(target,min_distance = 1,threshold_abs = 1e-6)

        return peaks_pos


    def get_CHs(self,output, peaks_pos):

        """
        * get_CHs: function to extract the CHs, which are the values of the outputs
        in correspondence of the peaks. Input parameters:

        - output: predictions or labels.
        - peaks_pos: pixel positions of the column's peaks.

        """

        CHs = np.round(output[peaks_pos[:, 0],
                                  peaks_pos[:, 1]])

        return CHs

    def get_r2(self,peaks_predictions):

        """
        * get_r2: function to calculate the R^2 between the predicted and true CHs for a single element
        Input parameters:

        - predictions: predicted CHs mao of a single element.
        - labels: true CHs map of a single element.
        - peaks_predictions: bool (True or False). If True, the R^2 is calculated in the peaks
          of the predictions, if False the R^2 is calculated in the peaks if the ground truth.

          True: taking into account the false positive.
          False: taking into account the true negative.

          The R^2 is calculated for both the cases and the final value is the average of the two
          If there are less than 2 columns in the prediction, or if the R^2 is negative, the value
          is set to 0.

        """

        if peaks_predictions:

            peaks_pos = self.get_peaks_pos(self.predictions[0,:,:])

        else:

            peaks_pos = self.get_peaks_pos(self.labels[0,:,:])

        if len(peaks_pos) > 2:

            CHs_predictions = self.get_CHs(self.predictions[0,:,:], peaks_pos)

            CHs_labels = self.get_CHs(self.labels[0,:,:], peaks_pos)

            r2 = r2_score(CHs_predictions,CHs_labels)

            if r2 < 0.0:

                r2 = 0.0
        else:

            r2 = 0.0

        return r2

    def get_avg_r2(self):

        """
       * get_avg_r2: function to calculate the average of the R^2 between the case
        of CHs calculated in the predicted peaks and the true peaks.

        """

        r2_1 = self.get_r2(peaks_predictions = True)
        r2_2 = self.get_r2(peaks_predictions = False)

        avg_r2 = (r2_1 + r2_2)/2

        return avg_r2

    def get_r2_batch(self):

        """
        * get_r2_all_elements_batch: function to calculate the R^2 for each element
        and for each data in the batch. Once the R^2 is calculated for each element,
        an average R^2 among all the elements is considered.

        """

        r2 = 0.0


        num_images = self.batch_predictions.shape[0]

        for self.predictions,self.labels in zip(self.batch_predictions,self.batch_labels):


            r2 += self.get_avg_r2()


        r2 = r2/num_images


        return r2


class Random_Imaging():

    """
    * Random_Imaging: class to perform random transformations on the images.
    The random transformations include random blur, random brightness, random contrast,
    random gamma and random flipping/rotation. Input parameters:

    - image: image to which apply the random transformations.
    - labels: included for the random flipping/rotation.

    """

    def __init__ (self,image,labels):

        self.image = image
        self.labels = labels

    def random_blur(self,image,low,high):

        sigma = np.random.uniform(low,high)

        image = gaussian(image,sigma)

        return image

    def random_brightness(self,image, low, high, rnd=np.random.uniform):

        image = image+rnd(low,high)

        return image

    def random_contrast(self,image, low, high, rnd=np.random.uniform):

        mean = np.mean(image)

        image = (image-mean)*rnd(low,high)+mean

        return image

    def random_gamma(self,image, low, high, rnd=np.random.uniform):

        min = np.min(image)

        image = (image-min)*rnd(low,high)+min

        return image

    def random_flip(self,image, labels, rnd=np.random.rand()):

        if rnd < 0.5:

            image[0,:,:,:] = np.fliplr(image[0,:,:,:])

            labels[0,:,:,:] = np.fliplr(labels[0,:,:,:])

        if rnd > 0.5:

            image[0,:,:,:] = np.flipud(image[0,:,:,:])

            labels[0,:,:,:] = np.flipud(labels[0,:,:,:])

        return image,labels

    def random_crop(self,image,labels,size):

        orig_size = image.shape[1:3]

        if orig_size>size:

            n = np.random.randint(0,orig_size[0]-size[0])
            m = np.random.randint(0,orig_size[1]-size[1])


            image = image[0,n:n+size[0],m:m+size[1],:]

            labels = labels[0,n:n+size[0],m:m+size[1],:]

        image, labels = image.reshape((1,)+image.shape),labels.reshape((1,)+labels.shape)

        return image, labels

    def get_trasform(self):

        self.image = self.random_brightness(self.image,low = -1,high = 1)

        self.image = self.random_contrast(self.image,low = 0.5,high = 1.5)

        self.image = self.random_gamma(self.image,low = 0.5,high = 1.5)

        self.image,self.labels = self.random_flip(self.image,self.labels)

        self.image = self.random_blur(self.image,low = 0, high = 2)

        return self.image,self.labels


def plot_debug(batch_images,batch_labels,batch_predictions,debug_folder_path):

    for i in range(len(batch_images)):


        fig = plt.figure(figsize=(21,7))
        ax = fig.add_subplot(1, 3, 1)

        im = ax.imshow(batch_images[i,0,:,:], cmap='gray')
        divider = make_axes_locatable(ax)
        cax1 = divider.append_axes("right", size="5%", pad=0.05)
        cbar = plt.colorbar(im, cax = cax1)

        ax = fig.add_subplot(1, 3, 2)
        im = ax.imshow(batch_labels[i,0,:,:], cmap='jet')
        divider = make_axes_locatable(ax)
        cax2 = divider.append_axes("right", size="5%", pad=0.05)
        cbar = plt.colorbar(im, cax = cax2)

        ax = fig.add_subplot(1, 3, 3)
        im = ax.imshow(batch_predictions[i, 0, :,:], cmap='jet')
        divider = make_axes_locatable(ax)
        cax2 = divider.append_axes("right", size="5%", pad=0.05)
        cbar = plt.colorbar(im, cax=cax2)

        plt.tight_layout()

        fig.savefig(debug_folder_path +'data_{}.png'.format(i + 1), bbox_inches='tight')

        plt.close(fig)

def plot_learning_curves(train_loss,test_loss,train_r2,test_r2,path):

    epochs = np.arange(1,len(train_loss) + 1)
    fig = plt.figure(figsize=(21, 7))

    fig.add_subplot(1, 2, 1)
    plt.plot(epochs,train_loss,'bo-')
    plt.plot(epochs, test_loss, 'ro-')
    plt.xlabel('Epochs')
    plt.ylabel('Loss')
    plt.legend(['Training','Test'])

    fig.add_subplot(1, 2, 2)
    plt.plot(epochs, train_r2, 'bo-')
    plt.plot(epochs, test_r2, 'ro-')
    plt.xlabel('Epochs')
    plt.ylabel('R2')
    plt.legend(['Training', 'Test'])

    plt.tight_layout()

    fig.savefig(path + 'learning_curves.png', bbox_inches='tight')

    plt.close(fig)