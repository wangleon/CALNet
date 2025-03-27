# import csv
import os
import numpy as np
import pandas as pd
from astropy.io import fits
import tensorflow as tf


filekey_lst = {
    1: (2018206045859, 120), 2: (2018234235059, 121),
    3: (2018263035959, 123), 4: (2018292075959, 124),
    5: (2018319095959, 125), 6: (2018349182500, 126),
    7: (2019006130736, 131), 8: (2019032160000, 136),
    9: (2019058134432, 139), 10: (2019085135100, 140),
    11: (2019112060037, 143), 12: (2019140104343, 144),
    13: (2019169103026, 146), 14: (2019198215352, 150),
    15: (2019226182529, 151), 16: (2019253231442, 152),
    17: (2019279210107, 161), 18: (2019306063752, 162),
    19: (2019331140908, 164), 20: (2019357164649, 165),
    21: (2020020091053, 167), 22: (2020049080258, 174),
    23: (2020078014623, 177), 24: (2020106103520, 180),
    25: (2020133194932, 182), 26: (2020160202036, 188),
    27: (2020186164531, 189), 28: (2020212050318, 190),
    29: (2020238165205, 193), 30: (2020266004630, 195),
    31: (2020294194027, 198), 32: (2020324010417, 200),
    33: (2020351194500, 203), 34: (2021014023720, 204),
    35: (2021039152502, 205), 36: (2021065132309, 207),
    37: (2021091135823, 208), 38: (2021118034608, 209),
    39: (2021146024351, 210), 40: (2021175071901, 211),
    41: (2021204101404, 212), 42: (2021232031932, 213),
    43: (2021258175143, 214), 44: (2021284114741, 215),
    45: (2021310001228, 216), 46: (2021336043614, 217),
    47: (2021364111932, 218), 48: (2022027120115, 219),
    49: (2022057073128, 221), 50: (2022085151738, 222),
    51: (2022112184951, 223), 52: (2022138205153, 224),
    53: (2022164095748, 226), 54: (2022190063128, 227),
    55: (2022217014003, 242), 56: (2022244194134, 243),
    57: (2022273165103, 245), 58: (2022302161335, 247),
    59: (2022330142927, 248), 60: (2022357055054, 249),
    61: (2023018032328, 250), 62: (2023043185947, 254),
    63: (2023069172124, 255), 64: (2023096110322, 257),
    65: (2023124020739, 259), 66: (2023153011303, 260),
    67: (2023181235917, 261), 68: (2023209231226, 262),
    69: (2023237165326, 264), 70: (2023263165758, 265),
    71: (2023289093419, 266), 72: (2023315124025, 267),
    73: (2023341045131, 268), 74: (2024003055635, 269),
    75: (2024030031500, 270), 76: (2024058030222, 271),
    77: (2024085201119, 272), 78: (2024114025118, 273),
    79: (2024142205832, 274), 80: (2024170053053, 275),
    81: (2024196212429, 276), 82: (2024223182411, 278),
    83: (2024249191853, 280), 84: (2024274222008, 281),
    85: (2024300212641, 282), 86: (2024326142117, 283),
    87: (2024353092137, 284), 88: (2025014115807, 285)
}

def get_lc_file(sector, tic):
    # get lc file name using its sector and tic
    timestamp, scid = filekey_lst[sector]
    return 'tess{:13d}-s{:04d}-{:016d}-{:04d}-s_lc.fits'.format(timestamp, sector, tic, scid)

def get_time_flux_from_filepath(filePath):
    '''
    get time and flux from file path of lc file
    Args:
        filePath: full path of lc file.
    returns:
        t: time series.
        f: flux series.
    '''
    table = fits.getdata(filePath)
    time = table['TIME']
    flux = table['PDCSAP_FLUX']
    q_lst = table['QUALITY']
    m = q_lst == 0
    time = time[m]
    flux = flux[m]
    m2 = ~np.isnan(flux)
    t = time[m2]
    f = flux[m2]
    return t,f



def predict(sector, lc_pre_save_loc, gls_pre_save_loc):
      '''
      Load preprocessed data and input it into the model for prediction.

      Args:
            sector: Sector name (string format)
            lc_pre_save_loc: The file path prefix where the processed lc  stored.
            gls_pre_save_loc: The file path prefix where the processed gls  stored.
      Output:
            predData: Predicted results (2D vector format)
            ticLst: TIC name list of the sector data
      '''
      # read processed data of TESS
      lc = pd.read_csv(lc_pre_save_loc + sector + '.csv',header=None)
      GLS = pd.read_csv(gls_pre_save_loc + sector + '.csv',header=None)
      # The first column of the data stores the names of lc.
      ticLst = lc.loc[:,0]
      # Excluding the first column are data points
      lcData = lc.loc[:,1:]
      GLSData = GLS.loc[:,1:]
      
      # Convert the format to facilitate input into the model.
      lcData = list(np.array(lcData))
      lcData = tf.convert_to_tensor(lcData)
      lcData = tf.expand_dims(lcData, 2)
      GLSData = list(np.array(GLSData))
      GLSData = tf.convert_to_tensor(GLSData)
      GLSData = tf.expand_dims(GLSData, 2)

      # predict
      predArray = model.predict([lcData,GLSData])
      #predArray = model.predict(lcData)

      #Save the predicted scores of each class
      predData = pd.DataFrame(predArray)

      return predData,ticLst

def getLabel(predData,threshold):
    '''
    Map the model's predicted results to their respective categories.

    Args:
        predData: The result data obtained after model prediction.
        threshold: The threshold for determining the category.
    Output:
        predLabel: Predicted category.
        scoreLst: Scores after prediction
    '''
    predLabel = []
    for i in range(len(predData)):
        maxone = np.argmax(predData[i])
        score = predData[i][maxone]
        if score >= threshold:
            label = label_lst[maxone]
        if score < threshold:
            label = 'NOTSURE'
        predLabel.append(label)
    return predLabel

def crossValidate(sec, th, pred_score_pre_loc, lc_pre_save_loc, gls_pre_save_loc):
    '''
    Cross-validation
    Args:
        sec: sector name.
        th: threshold for determining the category.
        pre_score_pre_loc: The file location prefix where the predScore stored.
        lc_pre_save_loc: The file path prefix where the processed lc  stored.
        gls_pre_save_loc: The file path prefix where the processed gls  stored.
    return:
        predEBTic: The set of TICs predicted as EB by the model.
        trueEBTic: The set of TICs of EBs we already known.
        crossEBTic: The intersection set of the predEBTic and trueEBTic.
    '''
    # If no prediction has been made of this sector, perform the prediction and obtain the prediction scores.
    if(not os.path.exists(pred_score_pre_loc + sec + '.csv')):
        predData, nameLst = predict(sec, lc_pre_save_loc, gls_pre_save_loc)
        ticLst = [int(name.split("-")[2]) for name in nameLst]
        predScore = pd.concat([pd.DataFrame(ticLst),predData],axis= 1)
        predScore.columns=["tic","EBscore","OTHERSscore"]
        predScore = predScore.drop_duplicates(subset='tic',keep='first').reset_index(drop=True)
        predScore.to_csv(pred_score_pre_loc + sec + '.csv',sep=',',index=False)

    # If prediction has already been made, directly read the saved prediction score table.
    else:
        predScore = pd.read_csv(pred_score_pre_loc + sec + '.csv')

    # Perform category prediction based on the threshold.
    predEBTic = set(predScore[predScore['EBscore'] >= th]['tic'])
    trueEBTic = set(EBData[EBData["sector"] == int(sec.strip('s'))]['TIC'])
    crossEBTic = predEBTic & allTic
    
    return predEBTic,trueEBTic,crossEBTic