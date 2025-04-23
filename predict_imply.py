import os
import json
import matplotlib.pyplot as plt
from astropy.table import Table 

import predict

#get EB data we already known
EBData = Table.read('data/newclass_samples.dat',
        format='ascii.fixed_width_two_line')
allTic = set(EBData['TIC'])

# read model cache -- CALNet
model = tf.keras.saving.load_model("models/model/CALNet.keras")


f = open('paths.json')
PATHS = json.load(f)
f.close()

threshold = 0.9
eb_images_pre_loc = 'outputs/images/'


sector_lst = [87, 88]

for sector in sector_lst:
    pred,true,cross = crossValidate(sector, threshold,
                PATHS['pred_score_pre_loc'],
                PATHS['lc_pre_save_loc'],
                PATHS['gls_pre_save_loc'],
                )
    print("--------------" + sec + "----------------")
    print("the number of EBs we predict:{}".format(len(pred)))
    print("the number of EBs we already known: {}".format(len(true)))
    print("the number of EBs we recall:{}".format(len(cross)))

    new = list(pred - true - cross) # new EBs we predict

    # plot EB lc figures and save
    for tic in new:
        fname = get_lc_file(sector, tic)
        filename = os.path.join(PATHS['lc_pre_loc'],
                    's{:03d}'.format(sector), fname)
        t,f = get_time_flux_from_filepath(filename)

        # file path that lc images will be stores
        filepath = os.path.join(eb_images_pre_loc, 's{:03d}'.format(sector))
        if not os.path.exists(filepath):
            os.makedirs(filepath)

        figname = os.path.join(filepath, 'tic{:011d}.png'.format(tic))
        if os.path.exists(figname):
            continue
        
        # plot and save
        fig = plt.figure(figsize = (40,10), dpi=100)
        ax = fig.gca()
        ax.plot(t,f,'o',color = 'C0',ms=3, alpha=1)
        fig.savefig(figname)
        plt.close(fig)
