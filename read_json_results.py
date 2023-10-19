import json
import matplotlib.pyplot as plt
import numpy as np
import os

reslist = os.listdir('mushra_results/')
results_all = []
for r in reslist:
    p = os.path.join('mushra_results/', r)
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


TRIALS_DYNAMIC_ANECH = [1, 3, 5]  # Dynamic anech conditions
DYNAMIC_ANECH_TWOSPEAK_OPP = 1
DYNAMIC_ANECH_TWOSPEAK_CLOSE = 3
DYNAMIC_ANECH_STRING_QUARTET = 5

TRIALS_DYNAMIC_STRONGREV = [0, 2, 4]  # Dynamic strongrev conditions
DYNAMIC_STRONGREV_TWOSPEAK_OPP = 0
DYNAMIC_STRONGREV_TWOSPEAK_CLOSE = 2
DYNAMIC_STRONGREV_STRING_QUARTET = 4

TRIALS_STATIC_ANECH = [7, 9, 11]  # Static anech conditions
STATIC_ANECH_TWOSPEAK_OPP = 7
STATIC_ANECH_TWOSPEAK_CLOSE = 9
STATIC_ANECH_STRING_QUARTET = 11

TRIALS_STATIC_STRONGREV = [6, 8, 10]  # Static strongrev conditions
STATIC_STRONGREV_TWOSPEAK_OPP = 6
STATIC_STRONGREV_TWOSPEAK_CLOSE = 8
STATIC_STRONGREV_STRING_QUARTET = 10


TRIALS_TWOSPEAK_OPP = [[STATIC_ANECH_TWOSPEAK_OPP, STATIC_STRONGREV_TWOSPEAK_OPP], [
    DYNAMIC_ANECH_TWOSPEAK_OPP, DYNAMIC_STRONGREV_TWOSPEAK_OPP]]
TRIALS_TWOSPEAK_CLOSE = [[STATIC_ANECH_TWOSPEAK_CLOSE, STATIC_STRONGREV_TWOSPEAK_CLOSE], [
    DYNAMIC_ANECH_TWOSPEAK_CLOSE, DYNAMIC_STRONGREV_TWOSPEAK_CLOSE]]
TRIALS_STRING_QUARTET = [[STATIC_ANECH_STRING_QUARTET, STATIC_STRONGREV_STRING_QUARTET], [
    DYNAMIC_ANECH_STRING_QUARTET, DYNAMIC_STRONGREV_STRING_QUARTET]]


# ---- EVALUATION PER AUDIO MATERIAL ----
TRIALS = [TRIALS_TWOSPEAK_OPP, TRIALS_TWOSPEAK_CLOSE, TRIALS_STRING_QUARTET]

audio_names = ['twospeaker_opp', 'twospeaker_close', 'string_quartet']
title_names = ['speech_opposite', 'speech_front', 'string_quartet']
axis = 0  # axis to compute median
ALPHA = 0.2
fontsz = 14
xlbl_fontsz = 12
xlbls = ['Ref.', 'FOA', 'BFBR', 'MLS', 'BF+MLS' +
         r'$*$' + ' ', ' BF+MLS', 'NBP', 'Mono']
for audio in range(3):
    fig, axs = plt.subplots(ncols=2, nrows=2, sharex=True,
                            sharey=True, gridspec_kw={'hspace': 0.1, 'wspace': 0.02}, figsize=(8*2, 3*2))
    for row in range(2):
        for col in range(2):
            trials = TRIALS[audio][row][col]
            results_spatial = results_all[:, 0, trials, :]
            results_timbral = results_all[:, 1, trials, :]
            h1 = axs[row, col].errorbar(np.array([0, 1, 2, 3, 4, 5, 6, 7])-0.125,
                                        np.median(
                results_spatial, axis=0),  # , 1)),
                yerr=(np.median(results_spatial, axis=axis) -
                      np.quantile(
                    results_spatial, 0.25, axis=axis),
                np.quantile(results_spatial, 0.75, axis=axis) -
                np.median(results_spatial, axis=axis)), ls='', marker='o', markersize=8, capsize=7, markerfacecolor='tab:blue', color='k')
            axs[row, col].scatter(np.random.rand(np.prod(results_spatial.shape)) * 0.1 - 0.05 - 0.125 +
                                  (np.tile(np.array([0, 1, 2, 3, 4, 5, 6, 7])[None, None, :],
                                           (results_all.shape[0], 1, 1))).flatten(),
                                  (results_spatial).flatten(), s=5, alpha=ALPHA, color='tab:blue')
            h2 = axs[row, col].errorbar(np.array([0, 1, 2, 3, 4, 5, 6, 7])+0.125,
                                        np.median(
                                            results_timbral, axis=axis),
                                        yerr=(np.median(results_timbral, axis=axis) -
                                              np.quantile(
                                            results_timbral, 0.25, axis=axis),
                np.quantile(results_timbral, 0.75, axis=axis) -
                np.median(results_timbral, axis=axis)), ls='', marker='o', markersize=8, capsize=7, markerfacecolor='tab:red', color='k')
            axs[row, col].scatter(np.random.rand(np.prod(results_timbral.shape)) * 0.1 - 0.05 + 0.125 +
                                  (np.tile(np.array([0, 1, 2, 3, 4, 5, 6, 7])[None, None, :],
                                           (results_all.shape[0], 1, 1))).flatten(),
                                  (results_timbral).flatten(), s=5, alpha=ALPHA, color='tab:red')

            axs[row, col].set_xticks(
                [0, 1, 2, 3, 4, 5, 6, 7], xlbls, fontsize=xlbl_fontsz)
            axs[row, col].legend(
                [h1[0], h2[0]], ['spatial', 'timbral'], loc='upper right', fontsize=fontsz, framealpha=1.0)

            if row == 0:
                if col == 0:
                    axs[row, col].set_title(
                        'anechoic conditions' + ': ' + title_names[audio], fontsize=fontsz)
                    axs[row, col].set_ylabel(
                        'perceived similarity', fontsize=xlbl_fontsz)
                if col == 1:
                    axs[row, col].set_title(
                        'reverberant conditions' + ': ' + title_names[audio], fontsize=fontsz)
                    axs[row, col].yaxis.set_label_position("right")
                    axs[row, col].yaxis.tick_left()
                    axs[row, col].set_ylabel(
                        'static array', fontsize=fontsz)
            if row == 1:
                if col == 0:
                    axs[row, col].set_ylabel(
                        'perceived similarity', fontsize=xlbl_fontsz)
                if col == 1:
                    axs[row, col].yaxis.set_label_position("right")
                    axs[row, col].yaxis.tick_left()
                    axs[row, col].set_ylabel(
                        'rotating array', fontsize=fontsz)
            axs[row, col].grid('on')

    plt.savefig('figures/exp_results_' +
                audio_names[audio] + '.pdf', bbox_inches='tight')

# ---- Evaluation for pooled audio material (MEAN POOLING) ----
TRIALS = [[TRIALS_STATIC_ANECH, TRIALS_STATIC_STRONGREV],
          [TRIALS_DYNAMIC_ANECH, TRIALS_DYNAMIC_STRONGREV]]
axis = 0  # axis to compute median
fig, axs = plt.subplots(ncols=2, nrows=2, sharex=True,
                        sharey=True, gridspec_kw={'hspace': 0.1, 'wspace': 0.02}, figsize=(8*2, 3*2))
for row in range(2):
    for col in range(2):
        trials = TRIALS[row][col]
        results_spatial = np.mean(
            results_all[:, 0, trials, :], axis=1)  # mean pooling
        results_timbral = np.mean(
            results_all[:, 1, trials, :], axis=1)  # mean pooling
        h1 = axs[row, col].errorbar(np.array([0, 1, 2, 3, 4, 5, 6, 7])-0.125,
                                    np.median(
            results_spatial, axis=0),  # , 1)),
            yerr=(np.median(results_spatial, axis=axis) -
                  np.quantile(
                results_spatial, 0.25, axis=axis),
            np.quantile(results_spatial, 0.75, axis=axis) -
            np.median(results_spatial, axis=axis)), ls='', marker='o', markersize=8, capsize=5, markerfacecolor='tab:blue', color='k')
        axs[row, col].scatter(np.random.rand(np.prod(results_spatial.shape)) * 0.1 - 0.05 - 0.125 +
                              (np.tile(np.array([0, 1, 2, 3, 4, 5, 6, 7])[None, None, :],
                                       (results_all.shape[0], 1, 1))).flatten(),
                              (results_spatial).flatten(), s=5, alpha=ALPHA, color='tab:blue')
        h2 = axs[row, col].errorbar(np.array([0, 1, 2, 3, 4, 5, 6, 7])+0.125,
                                    np.median(
                                        results_timbral, axis=axis),
                                    yerr=(np.median(results_timbral, axis=axis) -
                                          np.quantile(
                                        results_timbral, 0.25, axis=axis),
            np.quantile(results_timbral, 0.75, axis=axis) -
            np.median(results_timbral, axis=axis)), ls='', marker='o', markersize=8, capsize=5, markerfacecolor='tab:red', color='k')
        axs[row, col].scatter(np.random.rand(np.prod(results_timbral.shape)) * 0.1 - 0.05 + 0.125 +
                              (np.tile(np.array([0, 1, 2, 3, 4, 5, 6, 7])[None, None, :],
                                       (results_all.shape[0], 1, 1))).flatten(),
                              (results_timbral).flatten(), s=5, alpha=ALPHA, color='tab:red')

        axs[row, col].set_xticks(
            [0, 1, 2, 3, 4, 5, 6, 7], xlbls, fontsize=xlbl_fontsz)
        axs[row, col].legend(
            [h1[0], h2[0]], ['spatial', 'timbral'], loc='upper right', fontsize=fontsz, framealpha=1.0)

        if row == 0:
            if col == 0:
                axs[row, col].set_title(
                    'anechoic conditions', fontsize=fontsz)
                axs[row, col].set_ylabel(
                    'perceived similarity', fontsize=xlbl_fontsz)
            if col == 1:
                axs[row, col].set_title(
                    'reverberant conditions', fontsize=fontsz)
                axs[row, col].yaxis.set_label_position("right")
                axs[row, col].yaxis.tick_left()
                axs[row, col].set_ylabel(
                    'static array', fontsize=fontsz)
        if row == 1:
            if col == 0:
                axs[row, col].set_ylabel(
                    'perceived similarity', fontsize=xlbl_fontsz)
            if col == 1:
                axs[row, col].yaxis.set_label_position("right")
                axs[row, col].yaxis.tick_left()
                axs[row, col].set_ylabel(
                    'rotating array', fontsize=fontsz)
        axs[row, col].grid('on')

plt.savefig('figures/exp_results_audio_mean_pooled.pdf',
            bbox_inches='tight')
