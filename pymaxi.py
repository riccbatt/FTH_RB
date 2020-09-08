"""
Python Dictionary for getting the data out of the hdf files recorded with MAXI

2020
@authors:   KG: Kathinka Gerlinger (kathinka.gerlinger@mbi-berlin.de)
            RB: Riccardo Battistelli (riccardo.battistelli@helmholtz-berlin.de)
            MS: Michael Schneider (michaelschneider@mbi-berlin.de)
"""
import h5py
import numpy as np
import time


def entry_exists(fname, entry_number):
    '''Returns true if an hdf entry exists in file.'''
    with h5py.File(fname, 'r') as h5:
        return f'entry{entry_number}' in h5
    

def wait_for_entry(fname, entry_number):
    '''Periodically checks whether an entry exists in file.
    Blocks execution until entry found or KeyboardInterrupt.'''
    t0 = time.time()
    while not entry_exists(fname, entry_number):
        t = time.time() - t0
        print(f'waiting for entry {entry_number} in {fname} ({t:.0f}s)',
              '\t\t', end='\r')
        time.sleep(1)
    print(f'entry {entry_number} found.')
    return


def measurement_info(fname, entry_number, wait=True):
    '''
    Prints all the keys in the measurement.
    INPUT:  fname: path and name of the hdf file
            entry_number: number of the entry you want to check
    OUTPUT: None
    -----
    author: KG 2020
    '''
    with h5py.File(fname, 'r') as f:
        print(list(f[f'entry{entry_number:d}/measurement'].keys()))
    return


def diode_scan(fname, entry_number, motor):
    '''
    Function to evaluate a diode scan.
    INPUT:  fname: string, path and name of the hdf file
            entry_number: integer, number of the entry you want to check
            motor: string, name of the motor used in the diode scan. You can check it with measurement_info()
    OUTPUT: diode: list of the values measured by the diode
            motor_val: list of the motorpositions
    -----
    author: KG 2020
    '''
    with h5py.File(fname, 'r') as f:
        diode = f[f'entry{entry_number:d}/measurement/diodeA'][()]
        motor_val = f[f'entry{entry_number:d}/measurement/{motor:s}'][()]
    return (diode, motor_val)


def get_measurement(fname, entry_number, name):
    '''
    Function to get data of a specific entry and name, saved under measurement.
    INPUT:  fname: string, path and name of the hdf file
            entry_number: integer, number of the entry you want to check
            name: string, name of the key you want to read. You can check it with measurement_info()
    OUTPUT: list of the measured values
    -----
    author: KG 2020
    '''
    with h5py.File(fname, 'r') as f:
        return f[f'entry{entry_number:d}/measurement/{name:s}'][()]


def get_mte(fname, entry_number):
    '''
    Function to get data of a the camera of a specific entry.
    INPUT:  fname: string, path and name of the hdf file
            entry_number: integer, number of the entry you want to check
    OUTPUT: array of the image data
    -----
    author: KG 2020
    '''
    with h5py.File(fname, 'r') as f:
        return f[f'entry{entry_number:d}/measurement/mte'][()][0]

###############################################################################
#       Beamshape


def load_meshscan(fname, entry):
    '''
    Function to get the 2D beamintensity map recorded with the diode in the MAXI chamber, as well as the 2D arrays of the z- and y- meshgrid
    INPUT:  fname: string, path and name of the hdf file
            entry: integer, number of the entry you want to check
    OUTPUT: sz and sy meshgrid and intensity arrays
    -----
    author: dscran 2020
    '''
    with h5py.File(fname, 'r') as h5:
        sz = h5[f'entry{entry}/measurement/sz'][:]
        sy = h5[f'entry{entry}/measurement/sy'][:]
        intens = h5[f'entry{entry}/measurement/diodeA'][:]
        t = h5[f'entry{entry}/title'][()].split()
        size_z = int(t[4]) + 1
        size_y = int(t[8]) + 1
        sz, sy, intens = [np.reshape(a, (size_z, size_y)) for a in [sz, sy, intens]]
    return sz, sy, intens

def twoD_Gaussian(data, amplitude, xo, yo, sigma_x, sigma_y, theta, offset):
    '''
    2D Gaussian Function, used as input function for the beamshape fit.
    INPUT:  data: 2D data array
            amplitude: amplitude of the Gaussian (factor in front of the exponential function)
            xo, yo: offset in x, y coordinate
            sigma_x, sigma_y: sigma for the Gaussion
            theta: incline angle of the Gaussian
            offset: global offset of the Gaussian
    OUTPUT: Gaussian of data, flattened array (i.e. column after column in a vector). scipy fitting function can only fit 1D arrays.
    '''
    xo = float(xo)
    yo = float(yo)    
    a = (np.cos(theta)**2)/(2*sigma_x**2) + (np.sin(theta)**2)/(2*sigma_y**2)
    b = -(np.sin(2*theta))/(4*sigma_x**2) + (np.sin(2*theta))/(4*sigma_y**2)
    c = (np.sin(theta)**2)/(2*sigma_x**2) + (np.cos(theta)**2)/(2*sigma_y**2)
    g = offset + amplitude*np.exp( - (a*((data[0]-xo)**2) + 2*b*(data[0]-xo)*(data[1]-yo) + c*((data[1]-yo)**2)))
    return g.ravel()

def integer(n):
    '''return the rounded integer (if you cast a number as int, it will floor the number)'''
    return np.int(np.round(n))