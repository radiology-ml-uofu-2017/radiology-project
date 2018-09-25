import pickle
import pandas as pd
import numpy as np

file_root = '/usr/sci/projects/DeepLearning/Tolga_Lab/Projects/Project_JoyceSchroeder/data/data_PFT/MetaData/MetaData_'
file_end = {'2012-2016':'2012-2016/Chest_Xray_2012-2016_TJ_clean_ResearchID_PFTValuesAndInfo_No_PHI.csv', '2017':'2017/Chest_Xray_20180316_Clem_clean_ResearchID_PFTValuesAndInfo_noPHI.csv'}
all_labels = pd.concat([pd.read_csv(file_root + file_end[dataset]).assign(dataset=dataset) for dataset in ['2017','2012-2016']])
subjectids = np.array(all_labels['Subject_Global_ID'].unique())
with open('./validationsubjectids.pkl') as f:
    valids = pickle.load(f)
total_subjects = len(subjectids)
print(len(subjectids))
subjectids = np.setdiff1d(subjectids,np.array(list(valids)))
print(len(subjectids))
prng = np.random.RandomState()
queue = prng.permutation(subjectids.shape[0])
testids = set(subjectids[queue[:int(total_subjects*0.2)]])
print(len(testids))
print(len(valids))
pickle.dump( testids, open( "./testsubjectids.pkl", "wb" ) )