"""
Python Dictionary for FTH reconstructions in Python using functions defined in fth_reconstroction

2019/2020
@authors:   KG: Kathinka Gerlinger (kathinka.gerlinger@mbi-berlin.de)
            RB: Riccardo Battistelli (riccardo.battistelli@helmholtz-berlin.de)
            MS: Michael Schneider (michaelschneider@mbi-berlin.de)
"""

import os, sys
import numpy as np
import matplotlib.pyplot as plt
import pandas as pd
import h5py
#interactive stuff
import ipywidgets as widgets
from IPython.display import display
#fth dictionary
import fth_reconstruction as fth
import cameras as cam
import pymaxi as maxi




###########################################################################################

#                         RECONSTRUCT WITH MATLAB FILE                                    #

###########################################################################################


def allNew(image_folder, image_numbers, folder_matlab, matlab_number = None, size=[2052,2046], auto_factor = True, spe_prefix = None):
    '''
    This function reconstructs a hologram using values in a matlab file. This function is currently not used in the reconstruction files.
    INPUT:  image_folder: folder where the raw data is stored
            image_numbers: list or array of the images [positive helicity image, negative helicity image]
            folder_matlab: folder where the matlab parameter files are saved
            matlab_number: number of the matlab file, default is None (takes positive helicity image number)
            size: size of the hologram in pixels, default is the pixel dimensions of our greateyes camera.
            auto_factor:  determine the factor by which neg is multiplied automatically, if FALSE: factor is set to 0.5, default is TRUE
            spe_prefix: default is NONE in which case it opens greateyes files, otherwise, it opens spe files with the given prefix
    OUTPUT: difference hologram that is centered and multiplied with the beamstop mask, center coordinates, beamstop diameter and factor as a tuple
    -------
    author: KG 2019
    '''
    if spe_prefix is None:
        pos = cam.load_greateyes(image_folder + 'holo_%04d.dat'%image_numbers[0], size=size)
        neg = cam.load_greateyes(image_folder + 'holo_%04d.dat'%image_numbers[1], size=size)
    else: 
        pos = cam.load_spe(image_folder + spe_prefix + '%04d.spe'%image_numbers[0], return_header=False)
        neg = cam.load_spe(image_folder + spe_prefix + '%04d.spe'%image_numbers[1], return_header=False)

    holo, factor = fth.load_both(pos, neg, auto_factor = auto_factor)
    if matlab_number is None:
        matlab_number = image_numbers[0]
    center, bs_diam = fth.load_mat(folder_matlab, matlab_number)
    
    print("Start reconstructing the image using the center from the Matlab reconstruction.")
    holoN = fth.set_center(holo, center)
    #holoN = fth.mask_beamstop(holoN, beamstop, sigma=10)

    print("Please adapt the beamstop executing the function set_beamstop.")
    print("Please chose a ROI by executing the function set_roi.")
    return (holoN, center, bs_diam, factor)


def change_beamstop(bs_diameter):
    '''
    Change the beamstop diameter with a jupyter widget input field.
    Returns the input field. When you are finished, you can save the positions of the field.
    -------
    author: KG 2019
    '''
    style = {'description_width': 'initial'}

    bs_input = widgets.IntText(value=bs_diameter, description='ROI x1 coordinate:', disabled=False, style=style)

    button = widgets.Button(description='Finished')
    display(bs_input, button)

    return (bs_input, button)

def set_beamstop(holo, bs_diameter, sigma = 10):
    '''
    Input center-shifted hologram and the diameter of the beamstop. You may change the sigma of the gauss filter (default is 10).
    Returns the hologram where the beamstop is masked.
    -------
    author: KG 2019
    '''
    print("Masking beamstop with a diameter of %i pixels."%bs_diameter)
    holoB = fth.mask_beamstop(holo, bs_diameter, sigma=sigma)
    return holoB


def set_roi(holo, scale = (1,99)):
    """ 
    Select a ROI somewhat interactively, not used anymore!
    Input the shfited and masked hologram as returned from recon_allNew.
    Returns the four input fields. When you are finished, you can save the positions of the fields.
    -------
    author: KG 2019
    """
    recon=fth.reconstruct(holo)
    
    mi, ma = np.percentile(np.real(recon), scale)
    fig, ax = plt.subplots()
    ax = plt.imshow(np.real(recon), cmap='gray', vmin=mi, vmax=ma)
    plt.colorbar()

    style = {'description_width': 'initial'}

    ROIx1 = widgets.IntText(value=None, description='ROI x1 coordinate:', disabled=False, style=style)
    ROIx2 = widgets.IntText(value=None, description='ROI x2 coordinate:', disabled=False, style=style)
    ROIy1 = widgets.IntText(value=None, description='ROI y1 coordinate:', disabled=False, style=style)
    ROIy2 = widgets.IntText(value=None, description='ROI y2 coordinate:', disabled=False, style=style)

    button = widgets.Button(description='Finished')
    display(ROIx1, ROIx2, ROIy1, ROIy2, button)

    return (ROIx1, ROIx2, ROIy1, ROIy2, button)


def propagate(holo, ROI, phase=0, prop_dist=0, scale=(0,100), experimental_setup = {'ccd_dist': 18e-2, 'energy': 779.5, 'px_size' : 20e-6}):
    '''
    starts the quest for the right propagation distance and global phase shift.
    Input:  the shifted and masked hologram as returned from recon_allNew
            coordinates of the ROI in the order [Xstart, Xstop, Ystart, Ystop]
    Returns the two sliders. When you are finished, you can save the positions of the sliders.
    -------
    author: KG 2019
    '''
    ph_flip = False
    style = {'description_width': 'initial'}
    fig, axs = plt.subplots(1,2)
    def p(x,y):
        image = fth.reconstruct(fth.propagate(holo, x*1e-6, experimental_setup = experimental_setup)*np.exp(1j*y))
        mir, mar = np.percentile(np.real(image[ROI[2]:ROI[3], ROI[0]:ROI[1]]), scale)
        mii, mai = np.percentile(np.imag(image[ROI[2]:ROI[3], ROI[0]:ROI[1]]), scale)

        ax1 = axs[0].imshow(np.real(image[ROI[2]:ROI[3], ROI[0]:ROI[1]]), cmap='gray', vmin = mir, vmax = mar)
        #fig.colorbar(ax1, ax=axs[0], fraction=0.046, pad=0.04)
        axs[0].set_title("Real Part")
        ax2 = axs[1].imshow(np.imag(image[ROI[2]:ROI[3], ROI[0]:ROI[1]]), cmap='gray', vmin = mii, vmax = mai)
        #fig.colorbar(ax2, ax=axs[1], fraction=0.046, pad=0.04)
        axs[1].set_title("Imaginary Part")
        fig.tight_layout()
        print('REAL: max=%i, min=%i'%(np.max(np.real(image)), np.min(np.real(image))))
        print('IMAG: max=%i, min=%i'%(np.max(np.imag(image)), np.min(np.imag(image))))
        return
    
    layout = widgets.Layout(width='90%')
    style = {'description_width': 'initial'}
    slider_prop = widgets.FloatSlider(min=-10, max=10, step=0.01, value=prop_dist, layout=layout, description='propagation[um]', style=style)
    slider_phase = widgets.FloatSlider(min=-np.pi, max=np.pi, step=0.001, value=phase, layout=layout, description='phase shift', style=style)
    
    widgets.interact(p, x=slider_prop, y=slider_phase)

    #input("Press the <ENTER> key to continue...")
    button = widgets.Button(description="Finished")
    display(button) 
    def on_button_clicked(b):
        slider_prop.close()
        slider_phase.close()
        return
    button.on_click(on_button_clicked)
    
    #ph_flip_button = widgets.Button(description="Flip Phase")
    #display(ph_flip_button)
    #def flip_phase(b):
    #    ph_flip = not(ph_flip)
    #    return
    #button.on_click(flip_phase)
    
    return (slider_prop, slider_phase, button)#, ph_flip_button, ph_flip)



###########################################################################################

#                      RECONSTRUCT WITH PREVIOUS PARAMETERS                               #

###########################################################################################

def fromParameters(pos, neg, fname_param, new_bs=False, old_prop=True, topo=None, auto_factor=False, experimental_setup = {'ccd_dist': 18e-2, 'energy': 779.5, 'px_size' : 20e-6}):
    '''
    This function reconstructs a hologram using the latest parameters of the given hdf file.
    INPUT:  fname_im: name of the h5 file where the images are stored
            entry_nr: list or array of the entry numbers [positive helicity image, negative helicity image], single helicity reconstruction: just the entry number you want to reconstruct
            fname_param: path and filename of the hdf file where the parameters are stored.
            new_bs: boolean variable to indicate if you want to use the stored beamstop (FALSE) or if you want to change it (TRUE), default is FALSE
            old_prop: boolean variable to indicate if you want to use the stored propagation distance (TRUE) or if you want to change it (FALSE), default is TRUE
            topo_nr: numbers of the topography entries for single helicity reconstruction, default is None, then the topography numbers of the hdf file are used if possible
            helpos: boolean variable to indicate the helicity of the single helicity reconstruction, default is None which indicates a double helicity reconstruction
            auto_factor:  determine the factor by which neg is multiplied automatically, if FALSE: factor is set to 0.5, default is TRUE
            size: size of the hologram in pixels, default is the pixel dimensions of our greateyes camera.
            spe_prefix: default is NONE in which case it opens greateyes files, otherwise, it opens spe files with the given prefix
            holoN, factor, center, bs_diam, roi, prop_dist, phase
    OUTPUT: difference hologram that is centered and multiplied with the beamstop mask (if new_bs is FALSE) and propagated (if old_prop is TRUE), factor determined by the function
            center coordinates, beamstop diameter, ROI coordinates, propagation distance and phase from the hdf file
    -------
    author: KG 2020
    '''

    #Load the parameters from the hdf file
    _, nref, _, center, bs_diam, prop_dist, phase, roi = fth.read_hdf(fname_param)

    #Load the images (spe or greateyes; single or double helicity)
    if pos is None:
        if topo is None:
            print('Please provide topology!')
            return
        else:
            holo, factor = fth.load_single(neg, topo, False, auto_factor=auto_factor)
    elif neg is None:
        if topo is None:
            print('Please provide topology!')
            return
        else:
            holo, factor = fth.load_single(pos, topo, True, auto_factor=auto_factor)
    else:
        holo, factor = fth.load_both(pos, neg, auto_factor=auto_factor)

    print("Start reconstructing the image using the center and beamstop mask from the Matlab reconstruction.")
    holoN = fth.set_center(holo, center)

    if not new_bs:
        print("Using beamstop diameter %i from config file and a sigma of 10."%bs_diam)
        holoN = fth.mask_beamstop(holoN, bs_diam, sigma=10)
    else:
        print("Please adapt the beamstop using the beamstop function and then propagate.")
        return(holoN, factor, center, bs_diam, roi, prop_dist, phase)
        
    if old_prop:
        print("Using propagation distance from config file.")
        holoN = fth.propagate(holoN, prop_dist*1e-6, experimental_setup = experimental_setup)
        print("Now determine the global phase shift by executing phase_shift.")
    else: 
        print("Please use the propagation function to propagate.")

    return(holoN, factor, center, bs_diam, roi, prop_dist, phase)



def fromConfig(image_folder, image_numbers, folder_config, number_config, new_bs=False, old_prop=True, topo_nr=None, helpos=None, auto_factor=False, size=[2052,2046], spe_prefix=None):
    '''
    old function that uses the config file, replaced by funtion FromParameters with hdf file
    opens the image files in image_numbers (either single helicity or double helicity)
    if a spe_prefix is given, it opens spe files, otherwise it opens greateyes files.
    opens the config file for the center, beamstop, the ROI and the propagation etc.

    shifts the center and masks the beamstop
    returns the reconstruction parameters as well as the hologram that was corrected as inidcated (with and without bs mask, propagation)
    -------
    author: KG 2019
    '''

    #Load the config file
    nref, center, bs_diam, prop_dist, phase, roi = fth.read_config(folder_config + '%i_config.ini'%number_config)

    #Load the images (spe or greateyes; single or double helicity)
    if spe_prefix is None: #greateyes
        if helpos==None:
            print("Double Helicity Reconstruction")
            pos = cam.load_greateyes(image_folder + 'holo_%04d.dat'%image_numbers[0], size=size)
            neg = cam.load_greateyes(image_folder + 'holo_%04d.dat'%image_numbers[1], size=size)
            holo, factor = fth.load_both(pos, neg, auto_factor=auto_factor)
        else:
            print("Single Helicity Reconstruction")
            if topo_nr is None:
                if np.logical_or(np.isnan(nref[0]), np.isnan(nref[0])):
                    print("Please put in the numbers for the topography.")
                    return
                else:
                    pos = cam.load_greateyes(image_folder + 'holo_%04d.dat'%nref[0], size=size)
                    neg = cam.load_greateyes(image_folder + 'holo_%04d.dat'%nref[1], size=size)
            else:
                pos = cam.load_greateyes(image_folder + 'holo_%04d.dat'%topo_nr[0], size=size)
                neg = cam.load_greateyes(image_folder + 'holo_%04d.dat'%topo_nr[1], size=size)
    
            image = cam.load_greateyes(image_folder + 'holo_%04d.dat'%image_numbers, size=size)
            holo, factor = fth.load_single(image, pos+neg, helpos, auto_factor=auto_factor)
    else: #spe
        if helpos==None:
            print("Double Helicity Reconstruction")
            pos = cam.load_spe(image_folder + spe_prefix + '%04d.spe'%image_numbers[0], return_header=False)
            neg = cam.load_spe(image_folder + spe_prefix + '%04d.spe'%image_numbers[1], return_header=False)
            holo, factor = fth.load_both(pos, neg, auto_factor=auto_factor)
        else:
            print("Single Helicity Reconstruction")
            if topo_nr is None:
                if np.logical_or(np.isnan(nref[0]), np.isnan(nref[0])):
                    print("Please put in the numbers for the topography.")
                    return
                else:
                    pos = cam.load_spe(image_folder + spe_prefix + '%04d.spe'%n_ref[0], return_header=False)
                    neg = cam.load_spe(image_folder + spe_prefix + '%04d.spe'%n_ref[1], return_header=False)
            else:
                pos = cam.load_spe(image_folder + spe_prefix + '%04d.spe'%topo_nr[0], return_header=False)
                neg = cam.load_spe(image_folder + spe_prefix + '%04d.spe'%topo_nr[1], return_header=False)
    
            image = cam.load_spe(image_folder + spe_prefix + '%04d.spe'%image_numbers, return_header=False)
            holo, factor = fth.load_single(image, pos+neg, helpos, auto_factor=auto_factor)
    
    print("Start reconstructing the image using the center and beamstop mask from the Matlab reconstruction.")
    holoN = fth.set_center(holo, center)

    if not new_bs:
        print("Using beamstop diameter %i from config file and a sigma of 10."%bs_diam)
        holoN = fth.mask_beamstop(holoN, bs_diam, sigma=10)
    else:
        print("Please adapt the beamstop using the beamstop function and then propagate.")
        return(holoN, center, bs_diam, roi, prop_dist, phase)
        
    if old_prop:
        print("Using propagation distance from config file.")
        holoN = fth.propagate(holoN, prop_dist*1e-6)
        print("Now determine the global phase shift by executing phase_shift.")
    else: 
        print("Please use the propagation function to propagate.")

    return(holoN, factor, center, bs_diam, roi, prop_dist, phase)


def phase_shift(holo, roi, phase=0):
    '''
    starts the quest for the global phase shift.
    Input:  the shifted, masked and propagated hologram
            coordinates of the ROI in the order [Xstart, Xstop, Ystart, Ystop]
    Returns the two sliders. When you are finished, you can save the positions of the sliders.
    -------
    author: KG 2019
    '''
    fig, axs = plt.subplots(1,2)
    def p(x):
        image = fth.reconstruct(holo*np.exp(1j*x))
        ax1 = axs[0].imshow(np.real(image[roi[2]:roi[3], roi[0]:roi[1]]), cmap='gray')
        #fig.colorbar(ax1, ax=axs[0], fraction=0.046, pad=0.04)
        axs[0].set_title("Real Part")
        ax2 = axs[1].imshow(np.imag(image[roi[2]:roi[3], roi[0]:roi[1]]), cmap='gray')
        #fig.colorbar(ax2, ax=axs[1], fraction=0.046, pad=0.04)
        axs[1].set_title("Imaginary Part")
        fig.tight_layout()
        print('REAL: max=%i, min=%i'%(np.max(np.real(image)), np.min(np.real(image))))
        print('IMAG: max=%i, min=%i'%(np.max(np.imag(image)), np.min(np.imag(image))))
        return

    layout = widgets.Layout(width='90%')
    style = {'description_width': 'initial'}
    slider_phase = widgets.FloatSlider(min=-np.pi, max=np.pi, step=0.001, value=phase, layout=layout, description='phase shift', style=style)

    widgets.interact(p, x=slider_phase)

    #input("Press the <ENTER> key to continue...")
    button = widgets.Button(description="Finished")
    display(button)
    
    def on_button_clicked(b):
        slider_phase.close()
        return
    button.on_click(on_button_clicked)
    return (slider_phase, button)

###########################################################################################

#                               FINE TUNING                                               #

###########################################################################################


def deconvolve(holo, ROI, E0=0, Q=0.1, phi=0.1, scale=(0,100)):
    '''
    starts the quest for the right deconvolution by the ref. hole point spread function
    (https://www.sciencedirect.com/science/article/pii/S030439911930333X?via%3Dihub)
    INPUT:  holo: array, the centered and masked hologram
            ROI: array, ROI coordinates for your FOV [x1, x2, y1, y1]
            E0: optional, float, starting value for the E0 slider (default is 0)
            Q: optional, float, starting value for the Q slider (default is 0.1)
            phi: optional, float, starting value for the phi slider (default is 0.1)
    OUTPUT: the three sliders E0, Q, phi. When you are finished, you can save the positions of the sliders.
    ----
    author: RB 2020
    '''
    style = {'description_width': 'initial'}
    fig, axs = plt.subplots(1,2)

    npx,npy=holo.shape
    Y,X = np.meshgrid(range(npy),range(npx))
    R=np.sqrt((X-npx/2)**2+(Y-npy/2)**2)
    
    def p(x,y,phi):
        #create bessel function
        J=x*2*spec.j1(R*y*Q)/(R*y*Q)
        J[(npx//2),npy//2]=x
        #Wiener filtering
        J_w=(J**2+phi)/J
        image = fth.reconstruct(holo/J_w)
        mir, mar = np.percentile(np.real(image[ROI[2]:ROI[3], ROI[0]:ROI[1]]), scale)
        mii, mai = np.percentile(np.imag(image[ROI[2]:ROI[3], ROI[0]:ROI[1]]), scale)

        ax1 = axs[0].imshow(np.real(image[ROI[2]:ROI[3], ROI[0]:ROI[1]]), cmap='gray', vmin = mir, vmax = mar)
        #fig.colorbar(ax1, ax=axs[0], fraction=0.046, pad=0.04)
        axs[0].set_title("Real Part")
        ax2 = axs[1].imshow(np.imag(image[ROI[2]:ROI[3], ROI[0]:ROI[1]]), cmap='gray', vmin = mii, vmax = mai)
        #fig.colorbar(ax2, ax=axs[1], fraction=0.046, pad=0.04)
        axs[1].set_title("Imaginary Part")
        fig.tight_layout()
        print('REAL: max=%i, min=%i'%(np.max(np.real(image)), np.min(np.real(image))))
        print('IMAG: max=%i, min=%i'%(np.max(np.imag(image)), np.min(np.imag(image))))
        return
    
    layout = widgets.Layout(width='90%')
    style = {'description_width': 'initial'}
    slider_E0 = widgets.FloatSlider(min=0, max=10, step=0.01, value=E0, layout=layout, description='E0', style=style)
    slider_Q = widgets.FloatSlider(min=0, max=10, step=0.01, value=1, layout=layout, description='Q=pix_size*Rref/(lambda*z)', style=style)
    slider_phi = widgets.FloatSlider(min=0, max=10, step=0.01, value=phi, layout=layout, description='phi', style=style)
    
    widgets.interact(p, x=slider_E0, y=slider_Q, phi=slider_phi)

    #input("Press the <ENTER> key to continue...")
    button = widgets.Button(description="Finished")
    display(button)
    
    def on_button_clicked(b):
        slider_E0.close()
        slider_Q.close()
        slider_phi.close()
        return
    button.on_click(on_button_clicked)

    return (slider_E0, slider_Q, slider_phi, button)


def high_pass_filtering(holo, amp = .5, sig = 60):
    '''
    Applies a high pass Gauss filter to the hologram and lets you determine the amplitude and sigma.
    INPUT:  holo: array, the shifted and masked hologram
            amp: optional, float, starting value for the amplitude (default is .5)
            sig: optional, float, starting value for the sigma (default is 60)
    OUTPUT: the two sliders for the amplitude and the sigma. When you are finished, you can save the positions of the sliders.
    -------
    author: KG 2020
    '''
    style = {'description_width': 'initial'}
    fig, axs = plt.subplots(1,3, figsize = (10, 3))
    def p(amp, sig):
        holo_HP, HP = fth.highpass(holo,amplitude=amp,sigma=sig)

        ax1 = axs[0].imshow(holo, cmap='gray')
        axs[0].set_title("Original hologram")
        ax2 = axs[1].imshow(HP, cmap='gray')
        axs[1].set_title("High Pass")
        ax3 = axs[2].imshow(holo_HP, cmap='gray')
        axs[2].set_title("High Pass Hologram")
        fig.tight_layout()
        return

    layout = widgets.Layout(width='90%')
    style = {'description_width': 'initial'}
    slider_amp = widgets.FloatSlider(min=0, max=1, step=0.001, value=.5, layout=layout,
                                      description='amplitude', style=style)
    slider_sig = widgets.FloatSlider(min=0, max=500, step=0.5, value=60, layout=layout,
                                       description='sigma', style=style)

    widgets.interact(p, amp=slider_amp, sig=slider_sig)

    return(slider_amp, slider_sig)

def sub_pixel_centering(holo, roi, phase=0, prop_dist=0, scale=(0,100), ccd_dist=18e-2, energy=779.5, px_size=20e-6):
    '''
    Applies a sub-pixel centering, propagation distance and global phase shift.
    INPUT:  holo: array, the shifted and masked hologram
            roi: array, coordinates of the ROI in the order [Xstart, Xstop, Ystart, Ystop]
            phase: optional, float, starting value for the phase slider (default is 0)
            prop_dist: optional, float, starting value for the propagation slider (default is 0)
            scale: optional, tuple of floats, values for the scaling using percentiles (default is (0, 100))
            ccd_dist: optional, float, distance between CCD and sample in meter (default is 18e-2 (m))
            energy: optional, float, energy of the x-rays in eV (default is 779.5 (eV))
            px_size: optional, float, physical size of the CCD pixel in m (default is 20e-6 (m))
    OUPUT:  sliders for the propagation, phase, subpixel shift distances in x and y
            When you are finished, you can save the positions of the sliders.
    -------
    author: KG 2020
    '''
    style = {'description_width': 'initial'}
    fig, axs = plt.subplots(1,2)
    def p(x, y, fx, fy):
        image = fth.reconstruct(fth.propagate(holo, x*1e-6)*np.exp(1j*y))
        simage = fth.sub_pixel_centering(image, fx, fy)

        ax1 = axs[0].imshow(np.real(simage[roi[2]:roi[3], roi[0]:roi[1]]), cmap='gray')
        axs[0].set_title("Real Part")
        ax2 = axs[1].imshow(np.imag(simage[roi[2]:roi[3], roi[0]:roi[1]]), cmap='gray')
        axs[1].set_title("Imaginary Part")
        fig.tight_layout()
        print('REAL: max=%i, min=%i'%(np.max(np.real(simage)), np.min(np.real(simage))))
        print('IMAG: max=%i, min=%i'%(np.max(np.imag(simage)), np.min(np.imag(simage))))
        return

    layout = widgets.Layout(width='90%')
    style = {'description_width': 'initial'}
    slider_prop = widgets.FloatSlider(min=-10, max=10, step=0.01, value=prop_dist, layout=layout,
                                      description='propagation[um]', style=style)
    slider_phase = widgets.FloatSlider(min=-np.pi, max=np.pi, step=0.001, value=phase, layout=layout,
                                       description='phase shift', style=style)
    slider_dx = widgets.FloatSlider(min = -2, max = 2, step = 0.1, value = 0, layout = layout,
                                       description = 'x shift', style = style)
    slider_dy = widgets.FloatSlider(min = -2, max = 2, step = 0.1, value = 0, layout = layout,
                                       description = 'y shift', style = style)

    widgets.interact(p, x=slider_prop, y=slider_phase, fx = slider_dx, fy = slider_dy)

    return (slider_prop, slider_phase, slider_dx, slider_dy)

###########################################################################################

#                               SAVE THE CONFIG                                           #

###########################################################################################


def save_parameters(fname, recon, factor, center, bs_diam, prop_dist, phase, roi, image_numbers, comment = '', topo = None):
    '''
    Save everything in a hdf file. If the file already exists, append the reconstruction and parameters to that file (key is always reco%increasing number)
    INPUT:  fname: path and name of the hdf file
            recon: ROI of the reconstructed hologram
            factor: factor of the image and the topography
            center: center coordinates
            bs_diam: beamstop diameter
            prop_dist: propagation distance
            phase: phase
            roi: ROI coordinates
            image_numbers: array or list of the reconstructed images, single helicity reconstruction has one of the two image numbers as np.nan
            comment: string if you want to leave a comment about this reconstruction, default is an empty string
            topo: image numbers of the topography used as array or list, default is None in which case a list of [np.nan, np.nan] is saved
    -------
    author: KG 2020
    '''
    image_numbers = np.array(image_numbers)
    
    if image_numbers.size == 1:
        im = image_numbers
    elif np.isnan(image_numbers[0]):
        im = image_numbers[1]
    else:
        im = image_numbers[0]
    
    if topo is None:
        topo = [np.nan, np.nan]
    
    reco_dict = {
        'reconstruction': recon,
        'image numbers': image_numbers,
        'topo numbers': topo,
        'factor': factor,
        'center': center,
        'beamstop diameter': bs_diam,
        'ROI coordinates': roi,
        'Propagation distance': prop_dist,
        'phase': phase,
        'comment': comment
    }
    
    fth.save_reco_dict_to_hdf(fname, reco_dict)
    return


def save_parameters_config(holo, center, prop_dist, phase, roi, folder, image_numbers, bs_diam, propagate=False):
    '''
    old function to save parameters in a config file and reconstruction as numpy array. Replaced by saving everything in a hdf file with save_parameters()
    
    reconstruct the shifted and masked hologram (propagation and phase shift are performed here.)
    save all parameters in numpy-files (holo and beamstop) and a config file (rest)
    if the folder you put in does not exist, it will be created.
    '''
    image_numbers = np.array(image_numbers)
    if not(os.path.exists(folder)):
        print("Creating folder " + folder)
        os.mkdir(folder)

    if propagate:
        recon = fth.reconstruct(fth.propagate(holo, prop_dist*1e-6)*np.exp(1j*phase))
    else:
        recon = fth.reconstruct(holo*np.exp(1j*phase))
    print('Shifted phase by %f.'%phase)
    
    if image_numbers.size == 1:
        im = image_numbers
    elif np.isnan(image_numbers[0]):
        im = image_numbers[1]
    else:
        im = image_numbers[0]

    np.save(folder + '%i_recon'%im, recon[roi[2]:roi[3], roi[0]:roi[1]])
    
    fth.save_config(image_numbers, center, bs_diam, prop_dist, phase, roi, folder + '%i_config.ini'%im)
    return