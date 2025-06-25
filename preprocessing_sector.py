import os
import json
import numpy as np
import preprocessing as prep
import periodogram #GLS


def getProcessedData(lc_pre_loc, lc_pre_save_loc, gls_pre_save_loc,
        sector, gap_width, outlier_cut, flux_len, period_lst):
    '''
    get preprocessed lc and gls data and save them in the file.
    Args:
        lc_pre_loc: The file path prefix where the raw lc stored.
        lc_pre_save_loc: The file path prefix where the processed lc will be stored.
        gls_pre_save_loc: The file path prefix where the processed gls will be stored.
        sector: The sector number of the TESS data.
        gap_with: when the gap reaches the gap_width, we split the light curve.
        outlier_cut:The spline residuals before the data point with the maximum standard deviation from the median are considered outliers.
        flux_len: the length of the light curve we set.
        period_lst: The x-axis of the GLS data represents the frequency.
    '''
    # The file location where the data is stored.
    lc_csv_file = os.path.join(lc_pre_save_loc, 's{:03d}.csv'.format(sector))
    GLS_csv_file = os.path.join(gls_pre_save_loc, 's{:03d}.csv'.format(sector))
    
    # get lc files
    path = os.path.join(lc_pre_loc, 's{:03d}'.format(sector))
    filepath = os.listdir(path)
    for i in range(len(filepath)):
        print(i)
        fname = filepath[i]
        filename = os.path.join(path, fname)
        print(filename)
        # split process
        out_time,out_flux = prep.lc_split(filename, gap_width)
        for t,f in zip(out_time,out_flux):
            # we abandon the flux whose length < 2000
            if len(t) >= 2000:
                # dropMax process
                t,f = prep.dropMax(t,f,99)
                # preprocessing of lc
                final_flux = prep.process_lc(t,f,outlier_cut,flux_len)
                save_flux = [fname]+list(final_flux)
                # save into file
                prep.writeData(save_flux,lc_csv_file)
                
                # GLS operation, get power
                pdm = periodogram.GLS(t, f)
                power, winpower = pdm.get_power(period=period_lst)
                # normalization
                p_max = max(power)
                p_min = min(power)
                p = (power - p_min)/(p_max - p_min)
                save_power = [fname]+list(p)
                # save
                prep.writeData(save_power,GLS_csv_file)

gap_width = 0.75 
outlier_cut = 3
flux_len = 4000
period_lst = np.logspace(-3, 1, 1125)[125:]


f = open('../../Predict_Data/paths.json')
PATHS = json.load(f)
f.close()

sector_lst = [91]

for sector in sector_lst:
    getProcessedData(
        PATHS['lc_pre_loc'],
        PATHS['lc_pre_save_loc'],
        PATHS['gls_pre_save_loc'],
        sector, gap_width, outlier_cut, flux_len, period_lst)
