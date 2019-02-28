#! python3
# coding: utf-8
import os.path
from time import *


def download_mit_tec_data(y1, m1, d1, y2, m2, d2, path):
    import madrigalWeb.madrigalWeb

    # constants
    user_fullname = 'Shichuang He'
    user_email = 'ch.xiaohe@pku.edu.cn'
    user_affiliation = 'ISPAT of SESS of Peking University'
    madrigalUrl = 'http://madrigal.haystack.mit.edu/madrigal'
    # madrigalUrl = "http://cedar.openmadrigal.org/"
    code = 8000                   # code=<instrument code>,8000:Millstone, World-wide GPS Receiver

    # outpath = path + '{}\\raw_data\\'.format(y1)
    outpath = path + '{}\\data\\'.format(y1)
    print(outpath)
    if not os.path.exists(outpath):
        os.makedirs(outpath)
        # os.mkdir(path)
    Data = madrigalWeb.madrigalWeb.MadrigalData(madrigalUrl)
    print('get access')
    print()

    expList = Data.getExperiments(code,
                                  y1, m1, d1, 0, 0, 0,
                                  y2, m2, d2, 0, 0, 0)
    print(len(expList))
    for exp in expList:
        fileList = Data.getExperimentFiles(exp.id)
        for thisFile in fileList:
            if thisFile.category == 1:
                thisFilename = thisFile.name
                filename = thisFilename.split('/')[-1] + '.hdf5'

                if not os.path.exists(outpath + filename):
                    print(filename, end='  ')
                    starttime = time()
                    Data.downloadFile(thisFilename, outpath + filename, user_fullname,
                                      user_email, user_affiliation, 'hdf5')
                    print('-->download:', round(time() - starttime, 2), 'sec')
                    print()
                else:
                    print(filename, 'got it')


# out_path = 'E:\\master\\DATA\\GPS_MIT\\'
out_path = "C:\\DATA\\GPS_MIT\\millstone\\"
# out_path = "G:\\research\\DATA\\GPS_MIT\\"
y0, m0, d0 = 2016, 1, 1
y, m, d = 2016, 1, 10
download_mit_tec_data(y0, m0, d0, y, m, d, out_path)

"""
globalDownload.py --verbose --url=http://cedar.openmadrigal.org --outputDir=/tmp --user_fullname="Shichuang+He" 
--user_email=ch.xiaohe@pku.edu.cn --user_affiliation="ISPAT+of+SESS+of+Peking+University" --format="hdf5" 
--startDate="01/01/2014" --endDate="12/31/2014" --inst=8000 --kindat=3500 --expName="cedar"
"""
