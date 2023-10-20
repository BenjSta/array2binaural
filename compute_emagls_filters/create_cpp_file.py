# this script creates a cpp file containing the filter values for usage in a VST plugin

import numpy as np
import tqdm

xyz = np.load('compute_emagls_filters/xyz.npy')
arr = np.load('compute_emagls_filters/filters.npy')
roll = np.load('compute_emagls_filters/roll.npy')


with open('EASYCOM12356_TO_BINAURAL.cpp', 'w') as f:
    f.write('#include "MIC_ARRAY_CONSTANTS.h"\n')
    f.write('const float FILTERS[%d][%d][%d][%d][%d] = '%(
        arr.shape[0],arr.shape[1],arr.shape[2],arr.shape[3],arr.shape[4],
    ))
    f.write('{')
    for i in tqdm.tqdm(range(arr.shape[0])):
        f.write('{')
        for j in range(arr.shape[1]):
            f.write('{')
            for k in range(arr.shape[2]):
                f.write('{')
                mystr = ''
                for l in range(arr.shape[3]):
                    mystr += '{'
                    for m in range(arr.shape[4]):
                        mystr += np.format_float_scientific(arr[i, j, k, l, m], unique=True) + 'f,'
                    mystr += '},\n'
                mystr += '},\n' 
                f.write(mystr) 
            f.write('},\n') 
        f.write('},\n')
    f.write('};\n')

    f.write('const float XYZ[%d][3] = '%(
        arr.shape[0],
    ))
    
    f.write('{')
    mystr = ''
    for i in tqdm.tqdm(range(arr.shape[0])):
        mystr += '{'
        for j in range(3):
            mystr += np.format_float_scientific(xyz[i, j], unique=True) + 'f,'
        mystr += '},\n' 
    f.write(mystr) 
    f.write('};\n')

    f.write('const float ANGLES[%d] = '%( 
        arr.shape[1],
    ))
    f.write('{')
    mystr = ''
    for i in tqdm.tqdm(range(arr.shape[1])):
        mystr += np.format_float_scientific(roll[i], unique=True) + 'f,'
    mystr += '};\n' 
    f.write(mystr) 
