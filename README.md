# CALNet
CNN Attention LSTM Net, a hybrid deep learning model to identify eclipsing binaries from TESS light curves.

## About TESS
The Transiting Exoplanet Survey Satellite (TESS) is a space telescope designed to search for exoplanets using
the transit method. As of Apirl 2024, TESS has scanned over 85% of the entire sky, and released 2-minute cadence
light curves for about 530,000 stars. Besides the light curves, TESS also releases the Full-Frame Images (FFIs)
with the sampling of 30 minutes.

## Data Preparation
Before using this machine learning model, users can download all the 2-minute cadence light curves from the
[MAST archive](https://archive.stsci.edu/tess/bulk_downloads/bulk_downloads_ffi-tp-lc-dv.html), and we assume
all downloaded files are stored in the following path

        tess/lc/s{DDD}/{SSSSS}.fits

where `DDD` is the zero-padded 3-digit sector number (e.g., `s015` for Sector 15), and `SSSSS.fits` is the
filename of the light curve data in FITS format. This path can be changed in `paths.json`.

## Training Data
The list of TICs as input of training data can be found in `data/training_samples.dat`.

## Model Architecture

![image](https://github.com/wangleon/CALNet/blob/main/figures/CALNet_architecture.png)

*Architecture of CALNet. The structure of CAP modules is shown in the right.*

![image](https://github.com/wangleon/CALNet/blob/main/figures/CBAM_architecture.png)

*Structure of CBAM (Convolutional Block Attention Module), which combines
CAM (Channel Attention Module) and SAM (Spatial Attention Module)*

## Performance

![image](https://github.com/wangleon/CALNet/blob/main/figures/CALNet_preformance.png)

*Accuracy-loss curves of CALNet on training set and test set (left),
and confusion matrix of CALNet (right)*

## Results
Using CALNet, we totally identified 10,533 eclipsing binaries from Sectors 1-88 of TESS
2-min cadence light curves. The catalog is `output/newecl.dat` with Astropy
`ascii.fixed_width_two_line` format, including TIC IDs, coordinates, and *V* and *G*
magnitudes from [TESS Input Catalog v8.2](https://vizier.cds.unistra.fr/viz-bin/VizieR-3?-source=IV/39/tic82).

## See Also
More details can be found in [arXiv:2504.15875](https://arxiv.org/abs/2504.15875).
