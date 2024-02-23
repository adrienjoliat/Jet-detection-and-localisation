import os

import pandas as pd
import matplotlib.pyplot as plt
import numpy as np

import astropy.units as u
from astropy.coordinates import SkyCoord
from astropy.time import Time
from astropy.visualization import ImageNormalize, SqrtStretch
from astropy.io import fits

import sunpy.coordinates  # NOQA
import sunpy.map
from sunpy.net import Fido
from sunpy.net import attrs as a

import torch

# Animations
from matplotlib.animation import FuncAnimation
from IPython.display import HTML
from astropy.visualization import ImageNormalize, SqrtStretch

def load_data():
    """
    Function to load the data with jets
    """
    data=pd.read_csv("../data/events/Jet_clusters_3.0_2.0_paperID.csv", sep=",")
    data.columns=data.columns.str.strip()
    data=data.drop(columns="velocity")

    return data

def load_data_nojet():
    """
    Function to load the data without jets
    """
    data=pd.read_csv("../data/events/No_jet_df.csv")
    return data


def get_images(data, num_image, jsoc_email):
    """
    Inputs:
    data:       1xD, an line of data from dataframe (will be iterate in the main code)
    num_image:  scalar, number of images wanted in a sequence
    jsoc_email: string, verified mail address from the user

    
    Return:
    files:      sunpy content, Files extracted from the Jsoc database 
    """

    date=data["date"]

    start_time = Time(date, scale='utc', format='isot') #“CCYY-MM-DDThh:mm:ss[.sss. . . ]”, Coordinated Universal Time (UTC), 

    bottom_x=data["basepoint_X"]  # ew position
    bottom_y=data["basepoint_Y"]   # ns position
        
    bottom_left = SkyCoord((bottom_x-150)*u.arcsec, (bottom_y-150)*u.arcsec, obstime=start_time, observer="earth", frame="helioprojective")
    top_right = SkyCoord((bottom_x+150)*u.arcsec, (bottom_y+150)*u.arcsec, obstime=start_time, observer="earth", frame="helioprojective")

    cutout = a.jsoc.Cutout(bottom_left=bottom_left, top_right=top_right, tracking=False)
    query = Fido.search(
            a.Time(start_time , start_time + (num_image-1)*24*u.s), #we stop for n images
            a.Wavelength(304*u.angstrom), #Wavelength
            a.Sample(24*u.s), #one image /24 s 
            a.jsoc.Series.aia_lev1_euv_12s,
            a.jsoc.Notify(jsoc_email),
            a.jsoc.Segment.image,
            cutout,
        )
    
    files = Fido.fetch(query,overwrite=True)
    files.sort()

    return files

def array_file(files): 
    """
    Inputs:
    files:              sunpy content, Files extracted from the Jsoc database 

    Return:     
    sequence_array:     numpy array, Shape [height, width, num_images] (166,166,30)      
    """
    # Initialize the 3D matrix
    sequence_array = np.zeros((166, 166, len(files)))
    downsampling_layer = torch.nn.MaxPool2d(3, stride=3)
    # "files" is (class <parfive>) and contains N <HDUList> objects (where N is the nb of images in the sequence) 
    for i in range(len(files)):
        with fits.open(files[i]) as f:
            array = (f[1].data).astype(np.float32)
            array = torch.from_numpy(array).unsqueeze(0).unsqueeze(0)
            array = downsampling_layer(array)
            sequence_array[:,:,i] = array.numpy()

    return sequence_array


def plot_array(array):
    """
    Inputs:
    array               numpy array, Shape [height, width] (166,166)

    Function to plot a numpy array image       
    """
    plt.figure()
    vmin, vmax=np.percentile(array, [1, 99.9])
    norm=ImageNormalize(vmin=vmin, vmax=vmax, stretch=SqrtStretch())
    plt.imshow(array, norm=norm, cmap="sdoaia304")

    plt.show()
