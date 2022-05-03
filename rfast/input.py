import numpy as np

import ruamel.yaml as yaml
import warnings
import os

def read_yaml_input(scr_file):
    fil = open(scr_file, 'r')
    data = yaml.load(fil, Loader=yaml.Loader)
    fil.close()

    # Check for required keys, and check for types
    keys = list(data.keys())
    for key in REQUIRED_KEYS:
        if key not in keys:
            raise Exception('Missing key "' + key +
                            '" in input file "' + scr_file + '"')
        else:
            if type(data[key]) not in REQUIRED_KEYS[key]:
                raise Exception('Key "' + key + '" in input file "' + scr_file + '"' +
                                ' the wrong type. Should be ' + str(REQUIRED_KEYS[key]))

    # check options are valid
    src_options = ['diff', 'thrm', 'scnd', 'cmbn', 'trns', 'phas']
    if data['src'] not in src_options:
        raise Exception(
            '"src" must be one of the following: ' + str(src_options))

    if data['Nlev'] < 1:
        raise Exception('"Nlev" must be >= 1')

    if data['pmin'] < 0:
        raise Exception('"pmin" must be > 0')

    if data['pmax'] < 0:
        raise Exception('"pmax" must be > 0')

    if data['pmin'] >= data['pmax']:
        raise Exception('"pmin" must be less than "pmax"')

    bg_options = ['ar', 'ch4', 'co2', 'h2', 'he', 'n2', 'o2']
    if data['bg'] not in bg_options:
        raise Exception(
            '"bg" must be one of the following: ' + str(bg_options))

    species_r_options = ['ar', 'n2', 'o2', 'h2o',
                         'o3', 'co2', 'ch4', 'h2', 'he', 'n2o', 'co']
    for sp in data['species_r']:
        if sp not in species_r_options:
            raise Exception('"' + sp + '" is not an option for "species_r". ' +
                            'Must be one of the following: ' + str(species_r_options))
    data['species_r'] = np.array(data['species_r'], str)

    data['f0'] = np.array(data['f0'], np.float64)

    data['colr'] = np.array(data['colr'], np.int32)

    species_l_options = ['ch4', 'co2', 'h2o', 'o2', 'o3', 'n2o', 'co']
    for sp in data['species_l']:
        if sp not in species_l_options:
            raise Exception('"' + sp + '" is not an option for "species_l". ' +
                            'Must be one of the following: ' + str(species_l_options))
    data['species_l'] = np.array(data['species_l'], str)

    species_c_options = ['co2', 'h2', 'n2', 'o2']
    for sp in data['species_c']:
        if sp not in species_c_options:
            raise Exception('"' + sp + '" is not an option for "species_c". ' +
                            'Must be one of the following: ' + str(species_c_options))
    data['species_c'] = np.array(data['species_c'], str)

    for i in range(len(data['lams']) - 1):
        if data['lams'][i + 1] != data['laml'][i]:
            raise Exception('"lams" and "laml" do not perfectly meet')

    data['lams'] = np.array(data['lams'], np.float64)
    data['laml'] = np.array(data['laml'], np.float64)
    data['res'] = to_1d_ndarray(data['res'], np.float64)

    for mode in data['modes']:
        if mode not in [0, 1]:
            raise Exception('"modes" must be 0 or 1.')

    data['modes'] = np.array(data['modes'], np.int32)
    data['smpl'] = to_1d_ndarray(data['smpl'], np.float64)

    data['snr0'] = to_1d_ndarray(data['snr0'], np.float64)
    data['lam0'] = to_1d_ndarray(data['lam0'], np.float64)

    if data['nprocess'] == "max":
        data['nprocess'] = ""
    else:
        data['nprocess'] = int(data['nprocess'])

    ################################
    ### Ty's Check on the inputs ###
    ################################

    # set pf to -1 if user does not want iso-pressure opacities
    if (not data['fixp']):
        data['pf'] = -1

    # set tf to -1 if user does not want iso-temperature opacities
    if (not data['fixt']):
        data['tf'] = -1

    # check for consistency between wavelength grid and resolution grid
    if (data['lams'].shape[0] > 1 and data['lams'].shape[0] != data['res'].shape[0]):
        raise Exception(
            "rfast warning | major | smpl length inconsistent with wavelength grid")

    # check for consistency between resolution grid and over-sample factor
    if (data['smpl'].shape[0] > 1 and data['smpl'].shape[0] != data['res'].shape[0]):
        raise Exception(
            "rfast warning | major | smpl length inconsistent with resolution grid")

    # check for consistency between resolution grid and snr0 parameter
    if (data['snr0'].shape[0] > 1 and data['snr0'].shape[0] != data['res'].shape[0]):
        raise Exception(
            "rfast warning | major | snr0 length inconsistent with wavelength grid")

    # check for consistency between resolution grid and lam0 parameter
    if (data['lam0'].shape[0] > 1 and data['lam0'].shape[0] != data['res'].shape[0]):
        raise Exception(
            "rfast warning | major | lam0 length inconsistent with wavelength grid")

    # check that snr0 is within applicable wavelength range
    if (data['lam0'].shape[0] > 1):
        for i in range(data['lam0'].shape[0]):
            if (data['lam0'][i] < min(data['lams']) or data['lam0'][i] > max(data['laml'])):
                raise Exception(
                    "rfast warning | major | lam0 outside wavelength grid")
    else:
        if (data['lam0'][0] < min(data['lams']) or data['lam0'][0] > max(data['laml'])):
            raise Exception(
                "rfast warning | major | lam0 outside wavelength grid")

    # complete directory path if '/' is omitted
    if (len(data['opdir']) == 0):
        data['opdir'] = './hires_opacities/'
    elif (len(data['opdir']) > 0 and data['opdir'][-1] != '/'):
        data['opdir'] = data['opdir'] + '/'

    # check if opacities directory exists
    # if (not os.path.isdir(data['opdir'])):
    #     raise Exception(
    #         "rfast warning | major | opacities directory does not exist")

    # check if output directory exists
    if (len(data['dirout']) > 0 and data['dirout'][-1] != '/'):
        data['dirout'] = data['dirout'] + '/'
    if (len(data['dirout']) > 0 and not os.path.isdir(data['dirout'])):
        raise Exception('directory "dirout" does not exist')

    # check for mixing ratio issues
    if (np.sum(data['f0']) - 1 > 1.e-6 and not data['rdgas']):
        if (np.sum(data['f0']) - 1 < 1.e-3):
            warning.warn(
                "rfast warning | minor | input gas mixing ratios sum to slightly above unity")
        else:
            raise Exception(
                "rfast warning | major | input gas mixing ratios sum to much above unity")

    if 'gp' not in data and 'Mp' not in data:
        raise Exception(
            "Planet mass ('Mp') or gravity ('gp') must be specified.")
    elif 'gp' in data and 'Mp' in data:
        raise Exception(
            "rfast warning | major | cannot independently set planet mass and gravity in inputs")

    # set gravity if gp not set
    if 'gp' not in data:
        data['gp'] = 9.798 * data['Mp'] / data['Rp']**2

    # set planet mass if not set
    if "Mp" not in data:
        data['Mp'] = (data['gp'] / 9.798) * data['Rp']**2

    # cloud base cannot be below bottom of atmosphere
    if (data['pt'] + data['dpc'] > data['pmax']):
        raise Exception(
            "rfast warning | major | cloud base below bottom of atmosphere")

    # transit radius pressure cannot be larger than max pressure
    if (data['p10'] > data['pmax']):
        raise Exception(
            "rfast warning | major | transit radius pressure below bottom of atmosphere")

    return data

# helper function to convert float, int or list
# to a 1-D np.ndarray
def to_1d_ndarray(a, dtype):
    if isinstance(a, int) or isinstance(a, float):
        b = np.array([a], dtype=dtype)
    elif isinstance(a, list):
        b = np.array(a, dtype=dtype)
    else:
        raise ValueError('"a" must be a float, int or a list')
    return b


# map from the names of the keys
# to the allowable types
REQUIRED_KEYS = \
    {
        "fns": [str],
        "fnn": [str],
        "fnr": [str],
        "dirout": [str],
        "src": [str],
        "Nlev": [int],
        "pmin": [float, int],
        "pmax": [float, int],
        "bg": [str],
        "species_r": [list],
        "f0": [list],
        "rdgas": [bool],
        "fnatm": [str],
        "skpatm": [int],
        "colr": [list],
        "colpr": [int],
        "psclr": [float, int],
        "imix": [int],
        "t0": [float, int],
        "rdtmp": [bool],
        "fntmp": [str],
        "skptmp": [int],
        "colt": [int],
        "colpt": [int],
        "psclt": [float, int],
        "species_l": [list],
        "species_c": [list],
        "lams": [list],
        "laml": [list],
        "res": [list],
        "modes": [list],
        "regrid": [bool],
        "smpl": [float, int],
        "opdir": [str],
        "Rp": [float, int],
        "a": [float, int],
        "As": [float, int],
        "em": [float, int],
        "grey": [bool],
        "phfc": [int],
        "w": [float, int],
        "g1": [float, int],
        "g2": [float, int],
        "g3": [float, int],
        "pt": [float, int],
        "dpc": [float, int],
        "tauc0": [float, int],
        "lamc0": [float, int],
        "fc": [float, int],
        "ray": [bool],
        "cld": [bool],
        "fixp": [bool],
        "pf": [float, int],
        "fixt": [bool],
        "tf": [float, int],
        "p10": [float, int],
        "fp10": [bool],
        "ref": [bool],
        "sct": [bool],
        "alpha": [float, int],
        "ntg": [int],
        "Ts": [float, int],
        "Rs": [float, int],
        "ntype": [str],
        "snr0": [float, int],
        "lam0": [float, int],
        "rnd": [bool],
        "nwalkers": [int],
        "nstep": [int],
        "nburn": [int],
        "nprocess": [str, int],
        "thin": [int],
        "clr": [bool],
        "fmin": [float],
        "mmr": [bool],
        "progress": [bool],
        "restart": [bool],
    }

# Utility function for writing output
def src_to_names(src, is_noise=False):
    if (src == 'diff' or src == 'cmbn'):
        names = ['wavelength (um)', 'd wavelength (um)',
                 'albedo', 'flux ratio']
    elif (src == 'thrm'):
        names = ['wavelength (um)', 'd wavelength (um)',
                 'Tb (K)', 'flux (W/m**2/um)']
    elif (src == 'scnd'):
        names = ['wavelength (um)', 'd wavelength (um)',
                 'Tb (K)', 'flux ratio']
    elif (src == 'trns'):
        names = ['wavelength (um)', 'd wavelength (um)',
                 'zeff (m)', 'transit depth']
    elif (src == 'phas'):
        names = ['wavelength (um)', 'd wavelength (um)',
                 'reflect', 'flux ratio']

    if is_noise:
        names = names + ['data', 'uncertainty']

    return names
