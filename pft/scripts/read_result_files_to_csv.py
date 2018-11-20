import os
import numpy as np
array = np.array
dirFiles = os.listdir('/home/sci/ricbl/Documents/projects/temp_radiology/radiology-project/pft/log')
dirFiles.sort()
foundStart = False
for filename in dirFiles:
    if filename == 'log20180926-091332-6366.txt':
        foundStart = True
    if filename == 'save':
        foundStart = False
    if foundStart:
        timestamp = filename[3:-4]
        with open ('/home/sci/ricbl/Documents/projects/temp_radiology/radiology-project/pft/log/' + filename,"r" ) as fileHandle:
            lineList = fileHandle.readlines()
        for i in range(len(lineList)):
            try:
                param_name = lineList[i].split(':')[-2]
                if  param_name == 'model_to_load':
                    line_model_to_load = i
                if param_name == 'max_date_diff_to_use_for_test':
                    line_dd_test = i
            except IndexError:
                pass
        timestamp = lineList[line_model_to_load].split(':')[-1][1:-1]
        #dd_train = 180
        #{'2915':2,'7273':2,'1714':2,'6025':2,'1573':2,
        #            '7824':10,'6055':10,'5579':10,'1558':10,'1217':10,
        #            '5473':180,'5560':180,'3890':180,'1764':180,'1660':180}[timestamp[-4:]]
        dd_train = {'7848':2,'9895':2,'7256':2,'5297':2,'3377':2,
                    '9825':10,'4255':10,'8446':10,'6772':10,'2271':10,
                    '1256':180,'9432':180,'7222':180,'9156':180,'1413':180}[timestamp[-4:]]
        dd_test = int(lineList[line_dd_test].split(':')[-1][1:-1])
        dict2str = ''
        for i in range(0,6):
            line = lineList[len(lineList)-6+i]
            if i == 0:
                dict1str = line[line.find('{'):]
            elif i == 1:
                dict2str += line[line.find('{'):]
            else:
                dict2str += line
        dict1 = eval(dict1str)
        dict2 = eval(dict2str)

        def recur_dicts(dicta, pft_value):
            for measure1, value1 in dicta.iteritems():
                if type(value1) == dict:
                    recur_dicts(value1,pft_value)
                elif type(value1)==np.ndarray:
                    for i in range(value1.shape[0]):
                        for j in range(value1.shape[1]):
                            print(str(dd_train) + ',' + str(dd_test) + ',' + timestamp + ',' + pft_value + ',' + measure1 + ',' + str(value1[i][j]) + ',' + str(i) + ',' + str(j))
                else:
                    print(str(dd_train) + ',' + str(dd_test) + ',' +timestamp + ',' + pft_value + ',' + measure1 + ',' + str(value1))
                    
        for pft_value, innerdict in dict1.iteritems():
            recur_dicts(innerdict, pft_value)

        recur_dicts(dict2, '')