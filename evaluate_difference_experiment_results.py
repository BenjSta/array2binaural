import json
import matplotlib.pyplot as plt
import numpy as np
import os
import scipy.stats as stats

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

# COMPUTE DIFFERENCE GRADES ON RAW DATA
foa_minus_mls = np.mean(results_all[..., 1] - results_all[..., 3], axis=-1)
bfbr_minus_mls = np.mean(results_all[..., 2] - results_all[..., 3], axis=-1)
compact_minus_mls = np.mean(results_all[..., 6] - results_all[..., 3], axis=-1)

mls_minus_bfmls_comp = np.mean(
    results_all[..., 3] - results_all[..., 5], axis=-1)
mls_minus_bfmls_incomp = np.mean(
    results_all[..., 3] - results_all[..., 4], axis=-1)
bfmls_incomp_minus_bfmls_comp = np.mean(
    results_all[..., 4] - results_all[..., 5], axis=-1)

diffgrades = np.array([foa_minus_mls, bfbr_minus_mls,
                       compact_minus_mls, mls_minus_bfmls_incomp, mls_minus_bfmls_comp, bfmls_incomp_minus_bfmls_comp]).transpose(1, 2, 0)

diffgrades_spatial = diffgrades[:, 0, :]
diffgrades_timbral = diffgrades[:, 1, :]

# PLOT MEDIAN / MEAN DIFFERENCE GRADES
size = 2
fig, axs = plt.subplots(nrows=2, ncols=1, sharex=True,
                        figsize=(3*size, 2*size))

axis = 0
h1 = axs[0].errorbar(np.array([0, 1, 2, 3, 4, 5])-0.125,
                     np.median(
    diffgrades_spatial, axis=0),  # , 1)),
    yerr=(np.median(diffgrades_spatial, axis=axis) -
          np.quantile(
        diffgrades_spatial, 0.25, axis=axis),
    np.quantile(diffgrades_spatial, 0.75, axis=axis) -
    np.median(diffgrades_spatial, axis=axis)), ls='', marker='o', markersize=8, capsize=7, markerfacecolor='tab:blue', color='k')
h2 = axs[0].errorbar(np.array([0, 1, 2, 3, 4, 5])+0.125,
                     np.median(
    diffgrades_timbral, axis=0),  # , 1)),
    yerr=(np.median(diffgrades_timbral, axis=axis) -
          np.quantile(
        diffgrades_timbral, 0.25, axis=axis),
    np.quantile(diffgrades_timbral, 0.75, axis=axis) -
    np.median(diffgrades_timbral, axis=axis)), ls='', marker='o', markersize=8, capsize=7, markerfacecolor='tab:red', color='k')

axs[0].set_xticks([0, 1, 2, 3, 4, 5],  ['FOA vs.\nMLS', 'BFBR vs.\nMLS', 'NBP vs.\nMLS',
                                        'MLS vs.\nBF+MLS' + r'$*$', 'MLS vs.\nBF+MLS', 'BF+MLS' + r'$*$' + 'vs. \n BF+MLS',])
# axs[0].legend(
#    [h1[0], h2[0]], ['spatial', 'timbral'], loc=(0.725, 0.01), framealpha=0.5, ncols=1, borderpad=0.1)
axs[0].legend(
    [h1[0], h2[0]], ['spatial', 'timbral'], loc='lower right', framealpha=1)

axs[0].set_ylabel(
    'difference')
axs[0].grid('on')

# # COHEN's D
cohensd_foa_mls = np.mean(foa_minus_mls, axis=0) / \
    np.std(foa_minus_mls, axis=0)

cohensd_bfbr_mls = np.mean(bfbr_minus_mls, axis=0) / \
    np.std(bfbr_minus_mls, axis=0)

cohensd_compact_mls = np.mean(compact_minus_mls, axis=0) / \
    np.std(compact_minus_mls, axis=0)

cohensd_mls_bfmls_comp = np.mean(mls_minus_bfmls_comp, axis=0) / \
    np.std(mls_minus_bfmls_comp, axis=0)

cohensd_mls_bfmls_incomp = np.mean(mls_minus_bfmls_incomp, axis=0) / \
    np.std(mls_minus_bfmls_incomp, axis=0)

cohensd_bfmls_incomp_bfmls_comp = np.mean(bfmls_incomp_minus_bfmls_comp, axis=0) / \
    np.std(bfmls_incomp_minus_bfmls_comp, axis=0)

cohensd = np.array([cohensd_foa_mls, cohensd_bfbr_mls,
                    cohensd_compact_mls, cohensd_mls_bfmls_incomp, cohensd_mls_bfmls_comp, cohensd_bfmls_incomp_bfmls_comp])
cohensd_spatial = cohensd[:, 0]
cohensd_timbral = cohensd[:, 1]

alpha = 0.05
pm_conf_int = stats.t.ppf(
    1 - alpha/2, results_all.shape[0]-1) / np.sqrt(results_all.shape[0])


h1 = axs[1].errorbar(np.array([0, 1, 2, 3, 4, 5])-0.125,
                     cohensd_spatial,
                     yerr=pm_conf_int, ls='', marker='o', markersize=8, capsize=7, markerfacecolor='tab:blue', color='k')
h2 = axs[1].errorbar(np.array([0, 1, 2, 3, 4, 5])+0.125,
                     cohensd_timbral,
                     yerr=pm_conf_int, ls='', marker='o', markersize=8, capsize=7, markerfacecolor='tab:red', color='k')

axs[1].set_xticks([0, 1, 2, 3, 4, 5], ['FOA vs.\nMLS', 'BFBR vs.\nMLS', 'NBP vs.\nMLS',
                                       'MLS vs.\nBF+MLS' + r'$*$', 'MLS vs.\nBF+MLS', 'BF+MLS' + r'$*$' + 'vs. \n BF+MLS',])

axs[1].set_ylabel(
    'Cohen\'s d')
axs[1].grid('on')

plt.savefig('figures/diffgrades.pdf', bbox_inches='tight')
