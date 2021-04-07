"""
Python Dictionary for loading camera data (greateyes and princeton instruments)

2016/2019
@author: dscran & KG
"""

import numpy as np
import xmltodict as xml2d
import h5py


##################################
#         GREATEYES              #
##################################

def load_greateyes(filename, size=[2052,2046]):
    #input filename and optional size (default is the output of the MAXY greateyes camera)
    image = np.fromfile(filename, dtype=np.int32)
    image = np.reshape(image,(size[0],size[1]))
    print("Loaded image " + filename)
    return image


##################################
#         PRINCETON              #
##################################

def load_spe(fname, header_only=False, return_header = False):
    """
    Reads a Roper scientific SPE-file and returns the contents as an array. Header info is stored in the 'info' dictionary.
    
    Parameters
    ==========
    fname :		file path
        
    Returns
    =======
    image : 	CCD image as numpy array
    infodict :	meta info dictionary
    """
    with open(fname, 'rb') as fid:
        info = _load_header(fid)
        Nx = np.int(info['xdim'])
        Ny = info['ydim']
        Nf = info['frames']
        if return_header:
            if info['SPE_ver'] >= 3:
                fid.seek(4100 + Nx * Ny * Nf * info['dtype']().nbytes)  # seek beyond image file
                footer_xml = fid.read().decode('utf-8')  # rest is XML footer (SPE3.0)
                info['footer'] = xml2d.parse(footer_xml)
                info['timestamp'] = info['footer']['SpeFormat']['DataHistories']['DataHistory']['Origin']['@created']
            info['filename'] = fname
            if header_only:
                return info
        img = _read_at(fid, 4100, Nx * Ny * Nf, info['dtype'])
        if Nf > 1:
            img = img.reshape(Nx, Ny, Nf).astype(np.float64)
        else:
            img = img.reshape(Nx, Ny).astype(np.float64)
        # print("loaded %s (%dx%d pixel, %d frame(s))" % (fname, Nx, Ny, Nf))
        if return_header:
            return img, info
        return img

# functions needed in loadspe
def _read_at(fid, pos, size, ntype):
    fid.seek(pos)
    return np.fromfile(fid, ntype, size)

def _geometry(geom):
    '''Return image orientation options (rotate, reverse, flip) from header
    option value.
    '''
    flags = [int(i) for i in bin(geom)[2:]]
    values = ['rotate', 'reverse', 'flip']
    active = []
    for f, v in zip(flags, values):
        if f:
            active.append(v)
    return active


def _load_header(fid):
    info = {}
    dtypes = [np.float32, np.int32, np.int16, np.uint16, None, np.float64,
              np.uint8, None, np.uint32]
    try:
        info['exposure'] = _read_at(fid, 10, 1, np.float32)[0]
        info['temperature'] = _read_at(fid, 36, 1, np.float32)[0]
        info['ydim'] = _read_at(fid, 42, 1, np.uint16)[0]
        dtype_id = _read_at(fid, 108, 1, np.int16)[0]
#        print('dtype id', dtype_id)
        info['dtype'] = dtypes[dtype_id]
        info['geometry'] = _geometry(_read_at(fid, 600, 1, np.uint16)[0])
        info['xdim'] = _read_at(fid, 656, 1, np.uint16)[0]
        info['accumulations'] = _read_at(fid, 668, 1, np.uint32)[0]
        info['XML offset'] = _read_at(fid, 678, 1, np.uint64)[0]
        info['frames'] = _read_at(fid, 1446, 1, np.int32)[0]
        info['ROI_ystart'] = _read_at(fid, 1512, 1, np.uint16)[0]
        info['ROI_yend'] = _read_at(fid, 1514, 1, np.uint16)[0]
        info['ROI_ybin'] = _read_at(fid, 1516, 1, np.uint16)[0]
        info['ROI_xstart'] = _read_at(fid, 1518, 1, np.uint16)[0]
        info['ROI_xend'] = _read_at(fid, 1520, 1, np.uint16)[0]
        info['ROI_xbin'] = _read_at(fid, 1522, 1, np.uint16)[0]
        info['SPE_ver'] = _read_at(fid, 1992, 1, np.float32)[0]
    except IndexError:
        print("error reading header")
    return info

#seperate functions
def dump_folder(spefolder, pngfolder=None):
    '''Convert all SPE files in <spefolder> to png in <pngfolder> (if specified).
    Use <spefolder> if unspecified. Default contrast setting 1% / 99%.'''
    import tqdm
    from glob import glob
    from os.path import join, split
    from imageio import imwrite
    
    filelist = glob(join(spefolder, '**spe'), recursive=True)
    errors_on = []
    for spefile in tqdm.tqdm(filelist):
        try:
            im = loadspe(spefile)[0]
            im = np.clip(im, *np.percentile(im, [1, 99]))
            pngname = split(spefile)[1][:-3] + 'png'
            if pngfolder is None:
                pngfolder = spefolder
            imwrite(join(pngfolder, pngname), im)
        except:
            errors_on.append(spefile)
    num_total = len(filelist)
    num_error = len(errors_on)
    print()
    print('%d/%d files converted.' % (num_total - num_error, num_total))
    if num_error > 0:
        print('Errors occured on:')
        for f in errors_on:
            print(f)
    return