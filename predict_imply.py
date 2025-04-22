import matplotlib.pyplot as plt
from astropy.table import Table 
import predict

#get EB data we already known
EBData = Table.read('data/rawData/newclass_samples.dat', format='ascii.fixed_width_two_line')
allTic = set(EBData['TIC'])

# read model cache -- CALNet
model = tf.keras.saving.load_model("models/model/CALNet.keras")

lc_pre_save_loc = 'F:/tess/processedData/lc/'
gls_pre_save_loc = 'F:/tess/processedData/GLS/'
pred_score_pre_loc = 'F:/tess/predict/predScore/'
threshold = 0.9
lc_pre_loc = 'F:/tess/lc/'
eb_images_pre_loc = 'outputs/images/'


sectorLst = ['s087','s088']

for sec in sectorLst:
    pred,true,cross = crossValidate(sec, threshold, pred_score_pre_loc, lc_pre_save_loc, gls_pre_save_loc)
    print("--------------" + sec + "----------------")
    print("the number of EBs we predict:{}".format(len(pred)))
    print("the number of EBs we already known: {}".format(len(true)))
    print("the number of EBs we recall:{}".format(len(cross)))

    new = list(pred - true - cross) # new EBs we predict

    # plot EB lc figures and save
    for tic in new:
        name = get_lc_file(int(sec.strip('s')),tic)
        filename = lc_pre_loc + sec + '/' + name
        t,f = get_time_flux_from_filepath(filename)

        filepath = eb_images_pre_loc + sec # file path that lc images will be stores
        if not os.path.exists(filepath):
            os.makedirs(filepath)
        figName = filepath + '/' + str(tic) + '.png'
        if os.path.exists(figName):
            continue
        
        # plot and save
        fig = plt.figure(figsize = (40,10),dpi = 100)
        plt.ion()
        plt.plot(t,f,'o',color = 'C0',ms = 3,alpha = 1)
        plt.ioff()
        plt.savefig(filepath + '/' + str(tic) + '.png')
        plt.close()