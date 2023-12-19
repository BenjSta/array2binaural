import json
import matplotlib.pyplot as plt
import numpy as np
import os
from statsmodels.multivariate.manova import MANOVA
from statsmodels.stats.anova import AnovaRM
from statsmodels.stats.anova import anova_lm
from statsmodels.formula.api import ols
import pandas as pd

reslist = os.listdir('mushra/results/')
results_all = []
for r in reslist:
    p = os.path.join('mushra/results/', r)
    with open(p) as f:
        ratings = []
        r = json.load(f)
        for part in range(2):
            ratings_all_part = []
            for trial in r['Results']['Parts'][part]['Trials']:
                ratings_all_part.append(np.array(trial['Ratings'][1:]))

            ratings_all_part = np.stack(ratings_all_part)
            ratings.append(ratings_all_part)
        ratings = np.stack(ratings)
        results_all.append(ratings)
results_all = np.stack(results_all)

df = pd.DataFrame(columns=['Subject', 'Stimulus', 'Reverb', 'Rendering', 'Rotation', 'Spatial', 'Timbral'])

for subj in range(16):
    for rotation in [0, 1]:
        for reverb in [0, 1]:
            for istim, stimulus in enumerate(['speech_opposite', 'speech_front', 'string_quartet']):
                for irend, rendering in enumerate(['Ref.', 'FOA', 'BFBR', 'MLS', 'BF+MLS*', ' BF+MLS', 'NBP', 'Mono']):
                    trial_ind = 2 * istim + reverb + 6 * (1 - rotation)
                    
                    
                    subj_rendering_results = results_all[subj, :, trial_ind, irend]
                    
                    df = pd.concat([df, pd.DataFrame({'Subject': [subj], 'Stimulus': [stimulus],
                               'Reverb': [reverb], 'Rendering': [rendering],
                               'Rotation': [rotation],
                               'Spatial': [subj_rendering_results[0]], 'Timbral': [subj_rendering_results[1]]})], ignore_index=True)



model = MANOVA.from_formula('Spatial + Timbral ~ C(Subject) + C(Stimulus) + C(Rotation)*C(Reverb)*C(Rendering)', data=df).mv_test()


spatmodel = ols('Spatial ~ C(Subject) + C(Stimulus) + C(Rotation)*C(Reverb)*C(Rendering)', data=df).fit()
print(anova_lm(spatmodel))

results = AnovaRM(df, depvar='Spatial', subject='Subject', within=['Stimulus', 'Rotation', 'Reverb', 'Rendering', ]).fit()
print(results)

#timbmodel = MANOVA.from_formula('Timbral ~ C(Subject) + C(Stimulus) + C(Rotation)*C(Reverb)*C(Rendering)', data=df).mv_test()

print(model)
print(anova_lm(spatmodel))
print(spatmodel)
#print(timbmodel)

a = 1