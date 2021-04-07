import matplotlib.pyplot as plt
import matplotlib.colors as colors
from matplotlib.gridspec import GridSpec
import numpy as np
import numpy.fft as fft
from scipy.fftpack import fft2, ifftshift, fftshift,ifft2
import scipy.io
import fth_reconstruction as fth
from IPython.display import display
from IPython.display import clear_output
import ipywidgets as widgets
from skimage.draw import circle
from skimage.draw import ellipse
from skimage.draw import polygon
from skimage.draw import rectangle
import h5py
import math
import cupy as cp
import math
from scipy import signal
from skimage.transform import resize

def integer(n):
    return np.int(np.round(n))

#############################################################
#       Fabricate material
#############################################################

def material_hole(hole_type= 'object', SiN_z=200e-9, mask_z=1850e-9,
             n_layers=100, hole_diam=50e-9, beta_mask=1.9474e-3, delta_mask=2.9474e-3, l=1.59e-9, shape='circle', r_diam=25e-9, rotation=0, funnel_factor=0.5, material_size_factor=2, funnel_start=0.25, delta_SiN= 0.00114742666, beta_SiN=  0.000194844441):
    
    '''fabricates obj/reference hole matrix
    the hole is always as big as the matrix, will be accomodated later
    keeps into account if it is an obj. hole or a ref. hole
    magnetic pattern should be ones for ref holes
    hole_diam, r_diam refer to the hole dimensions on the SiN side
    
    OUTPUT: the size of the pixel for the material + a complex 3D numpy array, containing the material refractive indexes in each voxel
    RB_2020'''
    
    #first of all we wanna use an appropriate number of pixels, so that l**2*(ux**2+uy**2)) < 1 e quindi 0.5*ux**-2<l**2 e quindi px_size>l/sqrt(0.5)
    # px size=hole_diam/npx e quindi hole_diam/npx>l/sqrt(0.5) e quindi npx<hole_diam*sqrt(0.5)/l
    
    max_diam = np.maximum(hole_diam, hole_diam*funnel_factor) #always use the bigger part as a reference to choose the pixel number
    npx=math.floor(material_size_factor*max_diam*np.sqrt(0.5)/(2.*l))*2 #only even numbers
    npy=npx
        
    #size of pixel
    px_size=material_size_factor*max_diam/npx
    
    material=np.zeros((npy,npx,n_layers),dtype=complex)
    
    #total thickness of material
    thickness = SiN_z+ mask_z
    #thickness of single layer
    dz= thickness/n_layers
    
    # assign refractive index layer by layer
        
    #SiN membrane
    end_SiN=integer(SiN_z/dz)
    
    #SiN membrane
    material[:,:,:end_SiN] = delta_SiN + 1j*beta_SiN
    #Au (or other) mask
    material[:,:,end_SiN:] = delta_mask + 1j*beta_mask     
    
    # let's mill our hole
    if hole_type == 'object':
        start=end_SiN+1
    else:
        start=0
        
    for j in range(start,n_layers):
        
        #how should we scale the hole dimensions in this layer?
        if funnel_factor>1:
            scale_diam=np.maximum(1+(funnel_factor-1)/(1-funnel_start)*(j-funnel_start*n_layers)/n_layers,1.)
        else:
            scale_diam=np.maximum(1+(funnel_factor-1)/(funnel_start)*(j)/n_layers,funnel_factor)
        
        hole_diam_px=math.floor(hole_diam*np.sqrt(0.5)/(2.*l)*scale_diam)*2
        
        #coordinates of hole
        if shape=='circle':
            yy, xx = circle(npy//2, npx//2, hole_diam_px//2)
            
        elif shape=='ellipse':
            r_diam_px=math.floor(r_diam*np.sqrt(0.5)/(2.*l)*scale_diam)*2
            yy, xx = ellipse(npy//2, npx//2, r_radius=r_diam_px/2, c_radius=hole_diam_px/2, shape=None, rotation=-rotation)
            
        elif shape=='oblonge':
            r_diam_px=math.floor(r_diam*np.sqrt(0.5)/(2.*l)*scale_diam)*2

            x_center= math.floor((hole_diam_px-r_diam_px)/2 * np.cos(rotation))
            y_center= math.floor((hole_diam_px-r_diam_px)/2 * np.sin(rotation))
            shift_x= math.ceil(r_diam_px/2*np.cos(rotation))+1
            shift_y= math.ceil(r_diam_px/2*np.sin(rotation))+1

            yy1, xx1 = circle(y_center-npx//2, x_center-npx//2, r_diam_px//2)
            yy2, xx2 = circle(-y_center-npx//2, -x_center-npx//2, r_diam_px//2)
            yy3, xx3 = polygon([(y_center-shift_x-npx//2)%npx, (y_center+shift_x-npx//2)%npx, (-y_center+shift_x-npx//2)%npx , (-y_center-shift_x-npx//2)%npx], [(x_center-shift_y-npx//2)%npx, (x_center+shift_y-npx//2)%npx, (-x_center+shift_y-npx//2)%npx, (-x_center-shift_y-npx//2)%npx])
            yy=np.append(yy1,yy2)
            yy=np.append(yy,yy3)
            xx=np.append(xx1,xx2)
            xx=np.append(xx,xx3)
            
        elif shape=='rectangle':
            scale_diam_r=np.maximum(1+(funnel_factor*(hole_diam/r_diam)-1)/(1-funnel_start)*(j-funnel_start*n_layers)/n_layers,1.)
            r_diam_px=math.floor(r_diam*np.sqrt(0.5)/(2.*l)*scale_diam_r)*2
            sizes=(hole_diam_px,r_diam_px)
            start=((npx-hole_diam_px)//2,(npy-r_diam_px)//2)
            yy,xx= rectangle(start=start,extent=sizes)
            
        material[yy,xx,j]=0

      
    return px_size, material

#############################################################
#       Multislice
#############################################################
def Multislice_GPU(material=np.zeros((1000,1000,100),dtype=complex),field=np.ones((1000,1000),dtype=complex), l=1.59e-9, thickness=2050e-9, hole_diam=50e-9):
    '''Multislice simulations, computes transmission functions of a complex refractive index material matrix "material"
    of thickness "thickness" for a plane wave of wavelenght l
    can be used to compute transmittance of reference holes and object holes
    for object holes, field should be specified using the transmittance of the magnetized cobalt
    
    OUTPUT: field values at the end of the membrane
    (http://dx.doi.org/10.1364/OE.25.001831)
    RB_2020'''
    
    
    
    #first of all we wanna use an appropriate number of pixels, so that l**2*(ux**2+uy**2)) < 1 e quindi 2ux**-2<l**2 e quindi px_size>l/sqrt(2)
    # px size=hole_diam/npx e quindi hole_diam/npx>l/sqrt(2) e quindi npx<hole_diam*sqrt(2)/l
    #we have to reduce the resolution of the material matrix if it is too detailed. No
    
    npx=material.shape[0]
    npy=material.shape[1]
    n_layers=material.shape[2]
    
    material_cp=cp.asarray(material)
    field_cp=cp.asarray(field)
    
    #rescale entrance field pattern so that is the same number of pixels as the material
    crop=(field.shape[0]-material.shape[0])//2
    #we have to rescale the hole field we have to be hole_px large. We do it by doing the FT, cropping it and FT-1
    field_rescaled_cp=field_cp.copy()
    field_FT_cp= cp.fft.ifftshift(cp.fft.fft2(cp.fft.fftshift(field_cp)))
    if crop>0:
        field_rescaled_cp=cp.fft.fftshift(cp.fft.ifft2(cp.fft.ifftshift(field_FT_cp[crop:-crop,crop:-crop])))
    if crop<0:
        field_rescaled_cp=cp.fft.fftshift(cp.fft.ifft2(cp.fft.ifftshift(cp.pad(field_FT_cp, -crop, mode='constant'))))
    
    #size of pixel
    px_size=hole_diam/npx
    #distance between layers
    dz = thickness/n_layers
    #propagator that will be used often
    prop = 1j*2*np.pi*dz/l
    #spatial frequencies
    Y,X = np.meshgrid(range(npx),range(npx))
    ux =  (np.abs(X-npx/2.)) / ((npx/2.)*px_size)
    uy =  (np.abs(Y-npy/2.)) / ((npy/2.)*px_size)
    
    ux_cp=cp.asarray(ux)
    uy_cp=cp.asarray(uy)
    
    for i in range(n_layers):
        #field enter ith slab and gets modified by material
        field_rescaled_cp *= cp.exp(prop*(material_cp[:,:,i]))
        #field is propagated to next slab
        field_rescaled_cp  = cp.fft.ifftshift(cp.fft.fft2(cp.fft.fftshift(field_rescaled_cp)))
        field_rescaled_cp *= cp.exp( -prop* cp.sqrt(1-l**2*(ux_cp**2+uy_cp**2)))
        field_rescaled_cp  = cp.fft.ifftshift(cp.fft.ifft2(cp.fft.fftshift(field_rescaled_cp)))
        
    field_rescaled=cp.asnumpy(field_rescaled_cp)
    return field_rescaled

############

def Multislice(material=np.zeros((1000,1000,100),dtype=complex),field=np.ones((1000,1000),dtype=complex), l=1.59e-9, thickness=2050e-9, hole_diam=50e-9):
    '''Multislice simulations, computes transmission functions of a complex refractive index material matrix "material"
    of thickness "thickness" for a plane wave of wavelenght l
    can be used to compute transmittance of reference holes and object holes
    for object holes, field should be specified using the transmittance of the magnetized cobalt
    
    OUTPUT: field values at the end of the membrane
    (http://dx.doi.org/10.1364/OE.25.001831)
    RB_2020'''
    
    
    
    #first of all we wanna use an appropriate number of pixels, so that l**2*(ux**2+uy**2)) < 1 e quindi 2ux**-2<l**2 e quindi px_size>l/sqrt(2)
    # px size=hole_diam/npx e quindi hole_diam/npx>l/sqrt(2) e quindi npx<hole_diam*sqrt(2)/l
    #we have to reduce the resolution of the material matrix if it is too detailed. No
    
    npx=material.shape[0]
    npy=material.shape[1]
    n_layers=material.shape[2]
    
    #rescale entrance field pattern so that is the same number of pixels as the material
    crop=(field.shape[0]-material.shape[0])//2
    #we have to rescale the hole field we have to be hole_px large. We do it by doing the FT, cropping it and FT-1
    field_rescaled=field.copy()
    field_FT= np.fft.ifftshift(np.fft.fft2(np.fft.fftshift(field)))
    if crop>0:
        print(crop)
        field_rescaled=np.fft.fftshift(np.fft.ifft2(np.fft.ifftshift(field_FT[crop:-crop,crop:-crop])))
    if crop<0:
        print(crop)
        field_rescaled=np.fft.fftshift(np.fft.ifft2(np.fft.ifftshift(np.pad(field_FT, -crop, mode='constant'))))
    
    #size of pixel
    px_size=hole_diam/npx
    #distance between layers
    dz = thickness/n_layers
    #propagator that will be used often
    prop = 1j*2*np.pi*dz/l
    #spatial frequencies
    Y,X = np.meshgrid(range(npx),range(npx))
    ux =  (np.abs(X-npx/2.)) / ((npx/2.)*px_size)
    uy =  (np.abs(Y-npy/2.)) / ((npy/2.)*px_size)
    
    print('material shape',material.shape)
    print('field rescaled shape',field_rescaled.shape)
    
    for i in range(n_layers):
        #field enter ith slab and gets modified by material
        field_rescaled *= np.exp(prop*(material[:,:,i]))
        #field is propagated to next slab
        field_rescaled  = np.fft.ifftshift(np.fft.fft2(np.fft.fftshift(field_rescaled)))
        field_rescaled *= np.exp( -prop* np.sqrt(1-l**2*(ux**2+uy**2)))
        field_rescaled  = np.fft.ifftshift(np.fft.ifft2(np.fft.fftshift(field_rescaled)))
        
    return field_rescaled


#############

def Multislice2(material=np.zeros((1000,1000,100),dtype=complex),field=np.ones((1000,1000),dtype=complex), l=1.59e-9, thickness=2050e-9, hole_diam=50e-9):
    '''Multislice simulations, computes transmission functions of a complex refractive index material matrix "material"
    of thickness "thickness" for a plane wave of wavelenght l
    can be used to compute transmittance of reference holes and object holes
    for object holes, field should be specified using the transmittance of the magnetized cobalt
    
    OUTPUT: field values at the end of the membrane
    (http://dx.doi.org/10.1364/OE.25.001831)
    RB_2020'''
    
    
    
    #first of all we wanna use an appropriate number of pixels, so that l**2*(ux**2+uy**2)) < 1 e quindi 2ux**-2<l**2 e quindi px_size>l/sqrt(2)
    # px size=hole_diam/npx e quindi hole_diam/npx>l/sqrt(2) e quindi npx<hole_diam*sqrt(2)/l
    #we have to reduce the resolution of the material matrix if it is too detailed. No
    
    npx=material.shape[0]
    npy=material.shape[1]
    n_layers=material.shape[2]
    
    #rescale entrance field pattern so that is the same number of pixels as the material
    field_rescaled=resize(np.real(field), (npx,npy))+1j*resize(np.imag(field), (npx,npy))
    
    #size of pixel
    px_size=hole_diam/npx
    #distance between layers
    dz = thickness/n_layers
    #propagator that will be used often
    prop = 1j*2*np.pi*dz/l
    #spatial frequencies
    Y,X = np.meshgrid(range(npx),range(npx))
    ux =  (np.abs(X-npx/2.)) / ((npx/2.)*px_size)
    uy =  (np.abs(Y-npy/2.)) / ((npy/2.)*px_size)
    
    print('material shape',material.shape)
    print('field rescaled shape',field_rescaled.shape)
    
    for i in range(n_layers):
        #field enter ith slab and gets modified by material
        field_rescaled *= np.exp(prop*(material[:,:,i]))
        #field is propagated to next slab
        field_rescaled  = np.fft.ifftshift(np.fft.fft2(np.fft.fftshift(field_rescaled)))
        field_rescaled *= np.exp( -prop* np.sqrt(1-l**2*(ux**2+uy**2)))
        field_rescaled  = np.fft.ifftshift(np.fft.ifft2(np.fft.fftshift(field_rescaled)))
        
    return field_rescaled
#########
def Multislice_GPU2(material=np.zeros((1000,1000,100),dtype=complex),field=np.ones((1000,1000),dtype=complex), l=1.59e-9, thickness=2050e-9, px_size=1e-9):
    '''Multislice simulations, computes transmission functions of a complex refractive index material matrix "material"
    of thickness "thickness" for a plane wave of wavelenght l
    can be used to compute transmittance of reference holes and object holes
    for object holes, field should be specified using the transmittance of the magnetized cobalt
    
    OUTPUT: field values at the end of the membrane
    (http://dx.doi.org/10.1364/OE.25.001831)
    RB_2020'''
    
    
    
    #first of all we wanna use an appropriate number of pixels, so that l**2*(ux**2+uy**2)) < 1 e quindi 2ux**-2<l**2 e quindi px_size>l/sqrt(2)
    # px size=hole_diam/npx e quindi hole_diam/npx>l/sqrt(2) e quindi npx<hole_diam*sqrt(2)/l
    #we have to reduce the resolution of the material matrix if it is too detailed. No
    
    npx=material.shape[0]
    npy=material.shape[1]
    n_layers=material.shape[2]
    
    #rescale entrance field pattern so that is the same number of pixels as the material
    field_rescaled=resize(np.real(field), (npx,npy))+1j*resize(np.imag(field), (npx,npy))
    
    material_cp=cp.asarray(material)
    field_rescaled_cp=cp.asarray(field_rescaled)

    #distance between layers
    dz = thickness/n_layers
    #propagator that will be used often
    prop = 1j*2*np.pi*dz/l
    #spatial frequencies
    Y,X = np.meshgrid(range(npx),range(npx))
    ux =  (np.abs(X-npx/2.)) / ((npx/2.)*px_size)
    uy =  (np.abs(Y-npy/2.)) / ((npy/2.)*px_size)
    
    ux_cp=cp.asarray(ux)
    uy_cp=cp.asarray(uy)
    
    for i in range(n_layers):
        #field enter ith slab and gets modified by material
        field_rescaled_cp *= cp.exp(prop*(material_cp[:,:,i]))
        #field is propagated to next slab
        field_rescaled_cp  = cp.fft.ifftshift(cp.fft.fft2(cp.fft.fftshift(field_rescaled_cp)))
        field_rescaled_cp *= cp.exp( -prop* cp.sqrt(1-l**2*(ux_cp**2+uy_cp**2)))
        field_rescaled_cp  = cp.fft.ifftshift(cp.fft.ifft2(cp.fft.fftshift(field_rescaled_cp)))
        
    field_rescaled=cp.asnumpy(field_rescaled_cp)
    return field_rescaled

#########
def Multislice_GPU3(material=np.zeros((1000,1000,100),dtype=complex),field=np.ones((1000,1000),dtype=complex), l=1.59e-9, thickness=2050e-9, px_size=1e-9):
    '''Multislice simulations, computes transmission functions of a complex refractive index material matrix "material"
    of thickness "thickness" for a plane wave of wavelenght l
    can be used to compute transmittance of reference holes and object holes
    for object holes, field should be specified using the transmittance of the magnetized cobalt
    
    OUTPUT: field values at the end of the membrane
    (http://dx.doi.org/10.1364/OE.25.001831)
    RB_2020'''
    
    
    
    #first of all we wanna use an appropriate number of pixels, so that l**2*(ux**2+uy**2)) < 1 e quindi 2ux**-2<l**2 e quindi px_size>l/sqrt(2)
    # px size=hole_diam/npx e quindi hole_diam/npx>l/sqrt(2) e quindi npx<hole_diam*sqrt(2)/l
    #we have to reduce the resolution of the material matrix if it is too detailed. No
    
    npx=material.shape[0]
    npy=material.shape[1]
    n_layers=material.shape[2]
    
    #rescale entrance field pattern so that is the same number of pixels as the material
    field_rescaled=resize(np.real(field), (npx,npy))+1j*resize(np.imag(field), (npx,npy))
    field_cube=np.ones(material.shape,dtype=complex)
    field_cube[:,:,0] = field_rescaled
    
    material_cp=cp.asarray(material)
    field_cube_cp=cp.asarray(field_cube)
    
    #distance between layers
    dz = thickness/n_layers
    #propagator that will be used often
    prop = 1j*2*np.pi*dz/l
    #spatial frequencies
    Y,X = np.meshgrid(range(npx),range(npx))
    ux =  (np.abs(X-npx/2.)) / ((npx/2.)*px_size)
    uy =  (np.abs(Y-npy/2.)) / ((npy/2.)*px_size)
    
    ux_cp=cp.asarray(ux)
    uy_cp=cp.asarray(uy)
    
    propagator=cp.exp( -prop* cp.sqrt(1-l**2*(ux_cp**2+uy_cp**2)))
    
    
    for i in range(n_layers-1):
        #field enter ith slab and gets modified by material
        field_cube_cp[:,:,i] *= cp.exp(prop*(material_cp[:,:,i]))
        #field is propagated to next slab
        field_cube_cp[:,:,i+1]  = cp.fft.ifftshift(cp.fft.fft2(cp.fft.fftshift(field_cube_cp[:,:,i])))
        field_cube_cp[:,:,i+1] *= propagator
        field_cube_cp[:,:,i+1]  = cp.fft.ifftshift(cp.fft.ifft2(cp.fft.fftshift(field_cube_cp[:,:,i+1])))
        
    field_cube=cp.asnumpy(field_cube_cp)
    return field_cube


#############################################################
#       add_hole
#############################################################
def add_hole(field=np.zeros((1000,1000), dtype=complex), npx=None, hole_field=np.ones((100,100),dtype=complex), x_coor=0, y_coor=0, px_hole=1e-9, px_size=13.5e-6, l=1.59e-9, z=20e-2):
    '''add one OH/refH calculated from multislice to a matrix
    OUTPUT: composed image'''
    if npx==None:
        npx=field.shape[0]
        
    lpx = l*z/(px_size*npx)
    
    #let's redimension the holes to be the same number of pixels they have too be

    # hole square diameter, in px
    hole_px = math.ceil(px_hole*hole_field.shape[0]/lpx)
    y_coor_px= math.ceil(y_coor/lpx)
    x_coor_px= math.ceil(x_coor/lpx)
    
    #rescale entrance field pattern so that is the same number of pixels as the material
    hole_field_rescaled = resize(np.real(hole_field),(hole_px,hole_px)) + 1j * resize(np.imag(hole_field), (hole_px,hole_px))
    
    '''
    crop=math.floor((hole_field.shape[0]-hole_px)//2)
    #we have to rescale the hole field we have to be hole_px large. We do it by doing the FT, cropping it and FT-1
    hole_field_FT= np.fft.ifftshift(np.fft.fft2(np.fft.fftshift(hole_field)))
    hole_field_FT2= np.fft.ifftshift(np.fft.fft2(np.fft.fftshift(hole_field)))
    hole_field_rescaled=hole_field.copy()
    
    if crop>0:
        hole_field_rescaled=np.fft.fftshift(np.fft.ifft2(np.fft.ifftshift(hole_field_FT[crop:-crop,crop:-crop])))
    elif crop<0:
        crop=-crop
        hole_field_rescaled=np.fft.fftshift(np.fft.ifft2(np.fft.ifftshift(np.pad(hole_field_FT, crop, mode='constant'))))
    '''
   
    
    #hole_px+=2
    
    startx=field.shape[0]//2+x_coor_px-math.ceil(hole_px/2)
    starty=field.shape[0]//2+y_coor_px-math.ceil(hole_px/2)
    endx= startx + hole_field_rescaled.shape[1]
    endy= starty + hole_field_rescaled.shape[0]
    if ( startx > npx or starty > npx or endx > npx or endy > npx):
        print('We believe your CCD camera is too close, your FTH reconstruction is getting out of the camera')
        

    field[starty:endy, startx:endx] += hole_field_rescaled[:,:]
 
    
    return field
    
    
#############################################################
#       Holo simulation
#############################################################

def simulHolo2(im_p_focused,im_n_focused, Cntrs_level=1, readout_noise_average=0,
               readout_noise_sigma=3, sigma_h_px=0,
             max_counts_per_image=64000, counts_per_photon=50, number_frames=100, bs_size=0, sigma = 3):
    
    '''
    Given two starting real space images (for the two helicities), the function simulates the holograms introducing drift,
    contrast levels, coherence effects and Poisson noise
    INPUT:
            im_p_focused,im_n_focused: positive and negative helicity images
            Cntrs_level: contrast of the image. Can make the contrast lower than the original for values <1
            readout_noise_average, readout_noise_sigma: readout noise of the camera
            sigma_h_px: sigma of drift in pixles. Takes into account also the spatial incoherence, so that has to be be taken ito accout too
            max_counts_per_image: max number of counts the camera can take in one image
            counts_per_photon:
            number_of_frames: number of acquired frames. The more, the lower the noise
            
    ----------
    Author: RB_2020
    '''
    
    sample_pos=Cntrs_level*(im_p_focused-im_n_focused)/2+(im_p_focused+im_n_focused)/2
    sample_neg=Cntrs_level*(im_n_focused-im_p_focused)/2+(im_p_focused+im_n_focused)/2

    npx,npy=sample_pos.shape

    # 1. the coherent hologram is given by |FFT(object+reference)|^2
    pos = np.abs(np.fft.ifftshift(np.fft.fft2(np.fft.fftshift(sample_pos))))**2
    neg = np.abs(np.fft.ifftshift(np.fft.fft2(np.fft.fftshift(sample_neg))))**2
    
    # 4. simulate sample drift / vibrations AND SPATIAL INCOHERENCE
    if sigma_h_px > 0:
        kernel = np.outer(signal.gaussian(npx, sigma_h_px), signal.gaussian(npx, sigma_h_px))
        if kernel.sum() > 0:
            kernel /= kernel.sum()
            pos = signal.fftconvolve(pos,kernel,mode='same')
            neg = signal.fftconvolve(neg,kernel,mode='same')
            
    #mask beamstop
    pos=fth.mask_beamstop(pos, bs_size, sigma, center = None)
    neg=fth.mask_beamstop(neg, bs_size, sigma, center = None)

    # 6. adjust the maximum count to 64000 for a single image
    pos *= (max_counts_per_image)/pos.max()
    neg *= (max_counts_per_image)/neg.max()
    
    # consider you will have more than one frame
    pos *= number_frames
    neg *= number_frames
    
    
    # 7. add Poisson noise to number of photons (sqrt(counts/counts_per_photon))
    pos /= counts_per_photon
    neg /= counts_per_photon


    pos = np.random.poisson(pos).astype(np.float64)
    neg = np.random.poisson(neg).astype(np.float64)
    


    # 8. round to integer number of photons and convert back to counts
    pos = np.round(pos,0)
    neg = np.round(neg,0)
    pos *= counts_per_photon
    neg *= counts_per_photon
    
    # 9. add gaussian readout noise
    if (readout_noise_average > 0 or readout_noise_sigma > 0):
        pos += np.random.normal(readout_noise_average*number_frames,readout_noise_sigma*np.sqrt(number_frames),pos.shape)
        neg += np.random.normal(readout_noise_average*number_frames,readout_noise_sigma*np.sqrt(number_frames),neg.shape)
        
    pos/= number_frames
    neg/= number_frames
        
    pos=np.maximum(pos,np.zeros(pos.shape))
    neg=np.maximum(neg,np.zeros(pos.shape))
    #saturation
    pos=np.minimum(pos,np.ones(pos.shape)*max_counts_per_image)
    neg=np.minimum(neg,np.ones(pos.shape)*max_counts_per_image)

    #pos=pos.astype(complex)
    #neg=neg.astype(complex)
    
    return pos,neg


def simulHolo(im_p_focused,im_n_focused,mask,o_mask,r_mask,
              Cntrs_level=1, coherence=1, readout_noise_average=0, readout_noise_sigma=3, sigma_h_px=0,
             max_counts_per_image=64000, counts_per_photon=50, number_frames=100):
    
    '''Given two starting real space images (for the two helicities), the function simulates the holograms introducing drift,
    contrast levels, coherence effects and Poisson noise
    r_mask contains the mask of each reference hole
    RB_2020'''
    
    im_p_focused2=Cntrs_level*(im_p_focused-im_n_focused)/2+(im_p_focused+im_n_focused)/2
    im_n_focused2=Cntrs_level*(im_n_focused-im_p_focused)/2+(im_p_focused+im_n_focused)/2

    r=np.zeros((r_mask.shape[0],r_mask.shape[1],r_mask.shape[2]))
    for i in range(r_mask.shape[2]):
        r[:,:,i]=(im_p_focused2+im_n_focused2)/2*r_mask[:,:,i]
        print(i)

    npx,npy=im_p_focused.shape

    sample_pos=im_p_focused2 
    sample_neg=im_n_focused2
    o_pos=im_p_focused2*o_mask
    o_neg=im_n_focused2*o_mask

    # 1. the coherent hologram is given by |FFT(object+reference)|^2
    pos_coherent = np.abs(np.fft.ifftshift(np.fft.fft2(np.fft.fftshift(sample_pos))))**2
    neg_coherent = np.abs(np.fft.ifftshift(np.fft.fft2(np.fft.fftshift(sample_neg))))**2
    # 2. the incoherent hologram is given by |FFT(object)|^2 + |FFT(reference)|^2
    # For the incoherent part, we assume that the beam is coherent on the
    # scale of the object hole, but not on the scale of the distance
    # between object hole and reference.
    pos_incoherent = np.abs(np.fft.ifftshift(np.fft.fft2(np.fft.fftshift(o_pos))))**2 + np.sum(np.abs(np.fft.ifftshift(np.fft.fft2(np.fft.fftshift(r))))**2,axis=2)
    
    neg_incoherent = np.abs(np.fft.ifftshift(np.fft.fft2(np.fft.fftshift(o_neg))))**2 + np.sum(np.abs(np.fft.ifftshift(np.fft.fft2(np.fft.fftshift(r))))**2,axis=2)

    # 3. add coherent and incoherent holograms, weighted by the coherence factor
    pos = pos_coherent * coherence + pos_incoherent * (1-coherence)
    neg = neg_coherent * coherence + neg_incoherent * (1-coherence)

    # 4. simulate sample drift / vibrations
    if sigma_h_px > 0:
        kernel = np.outer(signal.gaussian(npx, sigma_h_px), signal.gaussian(npx, sigma_h_px))
        if kernel.sum() > 0:
            kernel /= kernel.sum()
            pos = signal.fftconvolve(pos,kernel,mode='same')
            neg = signal.fftconvolve(neg,kernel,mode='same')

    # 6. adjust the maximum count to 64000
    pos *= (max_counts_per_image*number_frames)/pos.max()
    neg *= (max_counts_per_image*number_frames)/neg.max()
    
    # 7. add Poisson noise to number of photons (sqrt(counts/counts_per_photon))
    pos /= counts_per_photon
    neg /= counts_per_photon
    #pos = np.random.poisson(pos).astype(np.float64)
    #neg = np.random.poisson(neg).astype(np.float64)
    #pos_coherent += np.random.normal(scale=np.sqrt(pos_coherent))
    #neg_coherent += np.random.normal(scale=np.sqrt(neg_coherent))
    
    # 6a. Due to numerical artifacts, some pixel values close to zero may
    #     now be negative. Therefore, we add 1e-3, which does not change the
    #     data in any measurable way, but ensures that all values are positive.
    pos += 1e-3
    neg += 1e-3

    # 8. round to integer number of photons and convert back to counts
    pos = np.round(pos,0)
    neg = np.round(neg,0)
    pos *= counts_per_photon
    neg *= counts_per_photon
    # 9. add gaussian readout noise
    if (readout_noise_average > 0 or readout_noise_sigma > 0):
        pos += np.random.normal(readout_noise_average*number_frames,readout_noise_sigma*np.sqrt(number_frames),pos.shape)
        neg += np.random.normal(readout_noise_average*number_frames,readout_noise_sigma*np.sqrt(number_frames),neg.shape)
    
    return pos,neg



###########################################



from scipy.optimize import least_squares

def autofactor(reco_pos, reco_neg):
    '''find a factor such that reco_pos - factor * reco_neg has minimum contrast'''
    factor = least_squares(lambda factor : (reco_pos-factor*reco_neg).std(),
                           1)
    return factor.x[0]


def autophase(reco,guess=0):
    '''find phase such that reco.imag has minimum contrast. Searches around guess +/- 0.5'''
    f = lambda phase : phaserot(reco,phase).imag.std()/phaserot(reco,phase).real.std()
    phase_guess = np.linspace(guess-3,guess+3,60)
    error = np.array([f(phase) for phase in phase_guess])
    try:
        phase = least_squares(f, phase_guess[error.argmin()])
        return phase.x[0]
    except:
        return phase_guess[error.argmin()]
    
def phaserot(image, angle):
    '''rotate complex angle of reconstruction by constant angle to move
    information from imaginary to real part.'''
    image = image.copy()
    j = np.complex(0, 1)
    image *= np.exp(j * angle)
    return image

def autophase_similar(image,image_original,o_mask,guess=0):
    '''find phase such that reco and reco_original are as similar as possible. Searches around guess +/- 0.5'''
    
    reco=np.abs(np.fft.fftshift(np.fft.fft2(np.fft.ifftshift(image))))**2
    reco_original=np.abs(np.fft.fftshift(np.fft.fft2(np.fft.ifftshift(image_original))))**2
    
    f = lambda phase :np.sum(np.real((fth.reconstruct(phaserot(reco,phase))-fth.reconstruct(reco_original))*o_mask)**2)
    phase_guess = np.linspace(guess-3.14,guess+3.14,40)
    error = np.array([f(phase) for phase in phase_guess])
    try:
        phase = least_squares(f, phase_guess[error.argmin()])
        return phase.x[0]
    except:
        return phase_guess[error.argmin()]
    
def autophase_similar2(reco,reco_original,o_mask,guess=0):
    '''find phase such that reco and reco_original are as similar as possible. Searches around guess +/- 0.5'''
    #f = lambda phase : phaserot(reco,phase).imag.std()/phaserot(reco,phase).real.std()
    f = lambda phase :np.sum(np.real((fth.reconstruct(phaserot(reco,phase))-fth.reconstruct(reco_original))*o_mask)**2)
    phase_guess = np.linspace(guess-3.14,guess+3.14,40)
    error = np.array([f(phase) for phase in phase_guess])
    try:
        phase = least_squares(f, phase_guess[error.argmin()])
        return phase.x[0]
    except:
        return phase_guess[error.argmin()]
    
def mask_gauss(sigma, mask):
    gauss_mask = gaussian_filter(mask, sigma, mode='constant', cval=1)
    gauss_mask_inv = gaussian_filter(np.logical_not(mask).astype(np.float64), sigma, mode='constant', cval=1)
    return (gauss_mask, gauss_mask_inv)