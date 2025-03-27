import preprocessing
import periodogram #GLS


def getProcessedData(lc_pre_loc, lc_pre_save_loc, gls_pre_save_loc, sec, gap_width, outlier_cut, flux_len, period_lst):
    '''
    get preprocessed lc and gls data and save them in the file.
    Args:
        lc_pre_loc: The file path prefix where the raw lc stored.
        lc_pre_save_loc: The file path prefix where the processed lc will be stored.
        gls_pre_save_loc: The file path prefix where the processed gls will be stored.
        sec:sector name
        gap_with: when the gap reaches the gap_width, we split the light curve.
        outlier_cut:The spline residuals before the data point with the maximum standard deviation from the median are considered outliers.
        flux_len: the length of the light curve we set.
        period_lst: The x-axis of the GLS data represents the frequency.
    '''
    # The file location where the data will be stored.
    lc_csv_file = lc_pre_save_loc + sec + '.csv'
    GLS_csv_file = gls_pre_save_loc + sec + '.csv'
    
    # get lc files
    dir = lc_pre_loc + sec + '/'
    fileLst = os.listdir(dir)
    for file in fileLst:
        filename = dir + file
        # split process
        out_time,out_flux = lc_split(filename,gap_width)
        for t,f in zip(out_time,out_flux):
            # we abandon the flux whose length < 2000
            if len(t) >= 2000:
                # dropMax process
                t,f = dropMax(t,f,99)
                # preprocessing of lc
                final_flux = process_lc(t,f,outlier_cut,flux_len)
                save_flux = [file]+list(final_flux)
                # save into file
                writeData(save_flux,lc_csv_file)
                
                # GLS operation, get power
                pdm = periodogram.GLS(t, f)
                power, winpower = pdm.get_power(period=period_lst)
                # normalization
                p_max = max(power)
                p_min = min(power)
                p = (power - p_min)/(p_max - p_min)
                save_power = [file]+list(p)
                # save
                writeData(save_power,GLS_csv_file)

gap_width = 0.75 
outlier_cut = 3
flux_len = 4000
period_lst = np.logspace(-3, 1, 1125)[125:]

lc_pre_loc = "F:/tess/lc/"
lc_pre_save_loc = 'F:/tess/processedData/lc/'
gls_pre_save_loc = 'F:/tess/processedData/GLS/'

sectorlst = ['s088']

for sec in sectorlst:
    getProcessedData(lc_pre_loc, lc_pre_save_loc, gls_pre_save_loc, sec, gap_width, outlier_cut, flux_len, period_lst)