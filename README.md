# CALNet
CNN Attention LSTM Net, a hybrid deep learning model designed to identify
**eclipsing binaries** from TESS (Transiting Exoplanet Survey Satellite)
light curves.

[![arXiv](https://img.shields.io/badge/arXiv-2504.15875-b31b1b.svg)](https://arxiv.org/abs/2504.15875)

## Overview
Eclipsing binaries (EBs) are binary star systems exibiting periodic dimming as
one star passes in front of the other. These systems enable precise measurements
of fundamental stellar parameters, such as mass, radius, and temperature,
providing key insights into the stellar evolution and some other astrophysical
processes.

The Transiting Exoplanet Survey Satellite (TESS) is a space telescope designed
to search for exoplanets using the transit method.
TESS has scanned over 85% of the entire sky and released 2-minute cadence light
curves for over 500,000 stars, making it a powerful tool for detecting eclipsing
binaries.

This model combines **Convolutional Neural Networks (CNNs)**,
**Long Short-Term Memory (LSTM) networks**, and an **Attention Mechanism
(CBAM)** to achieve high-precision classification of eclipsing binaries. 

## Key Features
- **Novel Data Fusion**: Integrates light curve (LC) data and generalized
  Lomb-Scargle periodograms (GLS) for enhanced feature extraction.
- **Robust Architecture**: CALNet leverages CNNs for spatial features, LSTMs for
  temporal dependencies, and CBAM to focus on critical patterns.
- **High Performance**: Achieves **99.1% recall** on test data, significantly
  reducing false negatives.

## Methodology
1. **Data Preprocessing**: Light curves are standardized via spline interpolation,
   outlier removal, and normalization. GLS periodograms capture periodic signals.

2. **Model Architecture**: 
   - **LC Branch**: Processes LC data through CNN-CBAM blocks and LSTM layers.
   - **GLS Branch**: Extracts spectral features from periodograms.
   - **Feature Fusion**: Combines outputs from both branches for final
     classification.

  ![image](https://github.com/wangleon/CALNet/blob/main/figures/CALNet_architecture.png)

  *Architecture of CALNet. The structure of CAP modules is shown in the right.*

  ![image](https://github.com/wangleon/CALNet/blob/main/figures/CBAM_architecture.png)

  *Structure of CBAM (Convolutional Block Attention Module), which combines CAM
  (Channel Attention Module) and SAM (Spatial Attention Module)*

3. **Training**: Utilizes cross-entropy loss and Adam optimizer.

## Results

- **Recall**: 99.1% (4,187/4,225 known EBs correctly identified).

  ![image](https://github.com/wangleon/CALNet/blob/main/figures/CALNet_preformance.png)

  *Accuracy-loss curves of CALNet on training set and test set (left), and
  confusion matrix of CALNet (right)*

- **Discoveries**: 10,533 eclipsing binaries validated from Sectors 1--88 of
  TESS 2-minute cadence data through manual inspection. The catalog is
  `output/newecl.dat` with Astropy `ascii.fixed_width_two_line` format,
  including TIC IDs, coordinates, and *V* and *G* magnitudes from
  [TESS Input Catalog v8.2](https://vizier.cds.unistra.fr/viz-bin/VizieR-3?-source=IV/39/tic82).

  Below figures shows the sky position, Tmag (TESS magnitude) histogram, and HRD
  of the eclipsing binaries identified from this work (blue) and
  [Prša et al. 2022](https://ui.adsabs.harvard.edu/abs/2022ApJS..258...16P/abstract)
  (yellow).

  ![image](https://github.com/wangleon/CALNet/blob/main/figures/skymap.png)

  ![image](https://github.com/wangleon/CALNet/blob/main/figures/tmag_hist.png)

  ![image](https://github.com/wangleon/CALNet/blob/main/figures/tmag_hist.png)


## Usage 

### Data Preparation
Before using this machine learning model, users can download all the 2-minute cadence light curves from the
[MAST archive](https://archive.stsci.edu/tess/bulk_downloads/bulk_downloads_ffi-tp-lc-dv.html), and we assume
all downloaded files are stored in the following path

        tess/lc/s{DDD}/{SSSSS}.fits

where `DDD` is the zero-padded 3-digit sector number (e.g., `s015` for Sector 15), and `SSSSS.fits` is the
filename of the light curve data in FITS format. This path can be changed in `paths.json`.

### Training
The list of TICs as input of training data can be found in `data/training_samples.dat`.

### Model File
A trained Keras model file can be downloaded from
[this link](https://calnet.s3.cn-north-1.amazonaws.com.cn/CALNet.keras).

## See Also
More details can be found in [arXiv:2504.15875](https://arxiv.org/abs/2504.15875).
