#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
contributing authors: dcp5303, ken38, seg246, Austin Gerlt, Simon Mason
"""
"""
    A few notes:
        - Depending on how many images you have this script can eat up a lot of RAM
        - Be sure to double check the meta data that is being read in to ensure it 
            is correct from the .par files and your logbook
        - Change the intensities in the plotting cells to scale to the data you have
        - Take time to double check that the spots you want are being binarized 
            correctly
        - In general, gaussian filtering is the easiest and often the most robust,
            just be careful not to blur your images
        - Non-local means filtering takes more time to optimize the parameters,
            but can do a very good job of detecting the spot edges
        - Errosion/dilation is there if the others fail
        - It is generally suggested to dilate through omega (only one frame on 
            either side) as this will handle any small omega differences between
            FF and NF
        - Removing small features may be needed for the gaussian filter as that
            function can blur hot pixels/noise to look like a very small spot.  
            Be very careful not to remove spots from grains which are small so 
            it is suggested to do some quick math on how many pixels to set as 
            the threshold in comarison to your grain size.  
"""
# %% ==========================================================================
# IMPORTS - DO NOT CHANGE
# =============================================================================
# Gneral imports
import matplotlib.pyplot as plt
import matplotlib
import importlib
import nf_config
import nfutil
import numpy as np
import os

# HEXRD Imports
importlib.reload(nfutil)  # This reloads the file if you made changes to it
importlib.reload(nf_config)
# Matplotlib
# This is to allow interactivity of inline plots in your gui
# the import ipywidgets as widgets line is not needed - however, you do need to run a pip install ipywidgets
# the import ipympl line is not needed - however, you do need to run a pip install ipympl
# import ipywidgets as widgets
# import ipympl 24
# The next lines are formatted correctly, no matter what your IDE says
# For inline, interactive plots (if you use these, make sure to run a plt.close() to prevent crashing)
# %matplotlib widget
# For inline, non-interactive plots
# %matplotlib inline
# For pop out, interactive plots (cannot be used with an SSH tunnel)
# %matplotlib qt
# %% ===========================================================================
# USER INPUT - CAN BE EDITED
# ==============================================================================
# What is the file path to the configuration file?
configuration_filepath = 'V-Y-2_S0_L2.yml'

# %% ===========================================================================
# LOAD CONFIGURATION - DO NOT EDIT
# ==============================================================================
# Go ahead and load the configuration
configuration = nf_config.open_file(configuration_filepath)[0]

# %% ===========================================================================
# LOAD METADATA AND DOWNSELECT - CAN BE EDITED
# ==============================================================================
downselection_number = None  # None if you want to do all images, else int
filenames, omega_edges_deg = nfutil.generate_filepaths_and_omegas(
    configuration, downselection_number)


# for i, filename in enumerate(filenames):
#    filenames[i] = filenames[i].replace(
#        '/p/lustre3/lim37/lim-4252-a/V-TT-1/88/nf/', '/p/lustre1/lim37/88/')

# %% ===========================================================================
# LOAD IMAGES - DO NOT EDIT
# ==============================================================================
# Load all of the images
controller = nfutil.build_controller(configuration)
raw_image_stack = nfutil.load_all_images(filenames, controller)

# %% ===========================================================================
# PLOTTING - CAN BE EDITED
# ==============================================================================
if configuration.output_plot_check:
    img_num = 50
    fig = plt.figure()
    plt.title('Raw Image: ' + str(img_num))
    plt.imshow(raw_image_stack[img_num, :, :],
               interpolation='none', clim=[0, 50], cmap='bone')
    plt.show(block=False)
# %% ===========================================================================
# INTENSITY CHECK - DO NOT EDIT
# ==============================================================================
if configuration.output_plot_check:
    summed_image_int = np.sum(np.sum(raw_image_stack, axis=1), axis=1)
    plt.figure()
    plt.scatter(np.arange(0, np.shape(filenames)[0], 1), summed_image_int)
    plt.ylim(0, np.max(summed_image_int))
    plt.title('Image Intensity vs Image Number')
    plt.xlabel('Image Number')
    plt.ylabel('Summed Intensity')
    plt.show(block=False)

    # What is the median intensity, how many are well below that and what could be the expected confidence drop
    median_int = np.median(summed_image_int)
    num_bad_images = np.sum(summed_image_int < median_int*0.75)
    print('There are potentially ' + str(num_bad_images) +
          ' images with poor intensity.')
    print('Potential confidence maximum around ' +
          str(np.round(1 - num_bad_images/np.shape(filenames)[0], 2)))

# %% ===========================================================================
# MEDIAN DARKFIELD REMOVAL - DO NOT EDIT
# ==============================================================================
# Perform median darkfield subtraction
cleaned_image_stack = nfutil.remove_median_darkfields(
    raw_image_stack, controller, configuration)

# %% ===========================================================================
# PLOTTING - CAN BE EDITED
# ==============================================================================
if configuration.output_plot_check:
    fig, axs = plt.subplots(1, 2)
    img_num = 10
    axs[0].imshow(raw_image_stack[img_num, :, :],
                  interpolation='none', clim=[0, 50], cmap='bone')
    axs[1].imshow(cleaned_image_stack[img_num, :, :],
                  interpolation='none', clim=[2, 20], cmap='bone')
    axs[0].title.set_text('Raw Image: ' + str(img_num))
    axs[1].title.set_text('Cleaned Image: ' + str(img_num))
    plt.show(block=False)
# %% ===========================================================================
# IMAGE CLEANING AND BINARIZATION - DO NOT EDIT
# ==============================================================================
# Perform image filtering, small object removal, and binarization
binarized_image_stack = nfutil.filter_and_binarize_images(
    cleaned_image_stack, controller, configuration.images.processing.method)

# %% ===========================================================================
# PLOTTING - CAN BE EDITED
# ==============================================================================
if configuration.output_plot_check:
    fig, axs = plt.subplots(1, 2)
    img_num = 50
    axs[0].imshow(cleaned_image_stack[img_num, :, :],
                  interpolation='none', clim=[1, 10], cmap='bone')
    axs[1].imshow(binarized_image_stack[img_num, :, :],
                  interpolation='none', clim=[0, 1], cmap='bone')
    axs[0].title.set_text('Cleaned Image: ' + str(img_num))
    axs[1].title.set_text('Binarized Image: ' + str(img_num))
    plt.show(block=False)

# %% ==========================================================================
# OMEGA DILATION - DO NOT EDIT
# =============================================================================
# Dilate the image stack in omega
dilated_image_stack = nfutil.dilate_image_stack(
    binarized_image_stack, configuration.images.processing.dilate_omega)

# %% ===========================================================================
# PLOTTING - CAN BE EDITED
# ==============================================================================
if configuration.output_plot_check == True and configuration.images.processing.dilate_omega > 0:
    fig, axs = plt.subplots(1, 2)
    img_num = 10
    axs[0].imshow(binarized_image_stack[img_num, :, :],
                  interpolation='none', clim=[0, 1], cmap='bone')
    axs[1].imshow(dilated_image_stack[img_num, :, :],
                  interpolation='none', clim=[0, 1], cmap='bone')
    axs[0].title.set_text('Binarized Image: ' + str(img_num))
    axs[1].title.set_text('Dilated Image: ' + str(img_num))
    plt.show(block=False)

# %% ==========================================================================
# SAVING - DO NOT EDIT
# =============================================================================
print(
    f'Saving image stack and omega edges to: {configuration.output_directory}')
nfutil.save_image_stack(configuration, dilated_image_stack, omega_edges_deg)

# %%
