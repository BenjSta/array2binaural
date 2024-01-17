#%%
import json
import matplotlib.pyplot as plt
import numpy as np
import os
import pandas as pd
from statsmodels.stats.anova import AnovaRM


reslist = os.listdir('mushra/results/')
results_all = []
names_all = []
for r in reslist:
    p = os.path.join('mushra/results/', r)
    with open(p) as f:
        ratings = []
        names = []
        r = json.load(f)
        for part in range(2):
            ratings_all_part = []
            names_all_part = []
            for trial in r['Results']['Parts'][part]['Trials']:
                ratings_all_part.append(np.array(trial['Ratings'][1:]))
                names_all_part.append(np.array(trial['TrialName']))
            ratings_all_part = np.stack(ratings_all_part)
            names_all_part = np.stack(names_all_part)
            ratings.append(ratings_all_part)
            names.append(names_all_part)
        ratings = np.stack(ratings)
        names = np.stack(names)
        results_all.append(ratings)
        names_all.append(names)
results_all = np.stack(results_all)
names_all = np.stack(names_all)

df = pd.DataFrame(columns=['Subject', 'Stimulus', 'Reverb', 'Rendering', 'Rotation', 'Spatial', 'Timbral'])


SUBJ_LEVS = np.arange(16)
ROT_LEVS = [0, 1]
REV_LEVS = [0, 1]
STIM_LEVS = ['speech_opposite', 'speech_front', 'string_quartet']
REND_LEVS = ['Ref.', 'FOA', 'BFBR', 'MLS', 'BF+MLS*', ' BF+MLS', 'NBP', 'Mono']


for subj in SUBJ_LEVS:
    for rotation in ROT_LEVS:
        for reverb in REV_LEVS:
            for istim, stimulus in enumerate(STIM_LEVS):
                trial_ind = 2 * istim + (1 - reverb) + 6 * (1 - rotation)
                if subj == 0:
                    assert names_all[subj, 0, trial_ind] == names_all[subj, 1, trial_ind]
                    print(names_all[subj, 0, trial_ind], stimulus, reverb, rotation)   
                
                for irend, rendering in enumerate(REND_LEVS):
                    subj_rendering_results = results_all[subj, :, trial_ind, irend]
                    
                    df = pd.concat([df, pd.DataFrame({'Subject': [subj], 'Stimulus': [stimulus],
                               'Reverb': [reverb], 'Rendering': [rendering],
                               'Rotation': [rotation],
                               'Spatial': [subj_rendering_results[0]], 'Timbral': [subj_rendering_results[1]]})], ignore_index=True)

#%%
df.to_csv('my_data.csv')