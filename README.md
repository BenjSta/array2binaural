# Array2Binaural

This repository contains the code to reproduce the signals and figures in the paper

```
@inproceedings{stahl2023perceptual2,
    title = {Perceptual Comparison of Dynamic Binaural Reproduction Methods for Head-Mounted Microphone Arrays},
    author = {Stahl, Benjamin and Riedel, Stefan},
    booktitle = {Proc. 155th Audio Engineering Society Convention},
    year = {2023}}
```

A demo of the listening experiment stimuli is available here: [https://array2binaural_demo.iem.sh/](https://array2binaural_demo.iem.sh/).

## Instructions for Stimulus Generation / Filter Computation / Instrumental Evaluation
### Environment
- Install the packages in `requirements.txt`.

### Download third-party data
- Download the Easycom array impulse responses data from [https://spear2022data.blob.core.windows.net/spear-data/Device_ATFs.h5](https://spear2022data.blob.core.windows.net/spear-data/Device_ATFs.h5) into `origin_array_tf_data/`.
- Download the boundary element method (BEM)-simulated array transfer functions by McCormack et al. from [https://zenodo.org/records/6401603/files/HMD_SensorArrayResponses.mat](https://zenodo.org/records/6401603/files/HMD_SensorArrayResponses.mat) into `origin_array_tf_data/`
- Download the EBU-SQAM snippets by running `simulate_scenarios_and_mic_signals/utils/download_and_cut_ebu_sqam.py` (without changing the working directory, i.e., from `./`).


### Convert the array transfer functions to SH domain
- Run the script `encode_array_into_sh.py` to encode the Easycom array transfer functions into the spherical harmonics domain. This will create the file `Easycom_array_32000Hz_o25_22samps_delay.npy`. 

### Simulate the scenarios and microphone signals
- Run the script `simulate_scenarios_and_mic_signals/generate_stimuli.py` from `./`. This will re-create the 6 Ambisonic `wav` files in `simulate_scenarios_and_mic_signals/audio_o25/`, however, with different seed for the diffuse part of the impulse response.

- Run the script `simulate_scenarios_and_mic_signals/create_mic_signals.py` from `./`. This will create simulated microphone signals in the `simulate_scenarios_and_mic_signals/rendered_mic` folder.

### Compute the magnitude-least-squares (MLS) filters
- Run `compute_emagls_filters/compute_emagls2_for_rotations.py` from `./` to compute MLS filters for a fine grid of rotations. This uses CUDA for speed-up, change the device to `cpu`, if you don't have a GPU/CUDA.
- (Optional:) Run `compute_emagls_filters/create_cpp_file.py` from `./` to create a `cpp` file with the filter coefficients. This can be used in a real-time application on the given array.  

### Compute 1st/5th-order Ambisonic signals for FOA encoding/decoding, BFBR, and DOA-informed BF+residual rendering.
- Run the script `encode.py` in order to apply FOA encoding and beamformers+5th-order encoding and write the output files into `encoded/`. These can be used for binaural rendering using Ambisonic binaural decoders. (For BFBR and  DOA-informed BF+residual rendering, this is a convenience step we used in our experiment, as we used the SceneRotator and 5th-order BinauralDecoder of the IEM-Plugin Suite for the rendering, so we did not have to write a real-time processor that selects and convolves HRIRs according to listener orientations.)

### Instrumental Evaluation
- In order to compute the resulting ILDs and ITDs and render the corresponding figures into `figures/`, run the scripts `ild_itd_analysis/compute_filters.py` and `ild_itd_analysis/evaluate_filters.py`. from `./`.
- In order to carry out the MUSIC simulation study investigating the inherent localization robustness of different arrays, run the script `simulation_study_music.py`. This will also create a figure in `figures/`.

## Evaluation of the listening experiment results
- Run `evaluate_raw_experiment_results.py` and `evaluate_difference_experiment_results.py` in order to obtain the listening experiment response data visualizations displayed in the paper.