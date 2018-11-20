import csv
try:
    import cPickle as pickle
except:
    import _pickle as pickle 
                
try:
    with open('./testsubjectids.pkl') as f: 
        testids = pickle.load(f)
except TypeError:
    with open('./testsubjectids.pkl', 'rb') as f:
        testids = pickle.load(f, encoding='latin1')
try:
    with open('./validationsubjectids.pkl') as f:
        valids = pickle.load(f)
except TypeError:
    with open('./validationsubjectids.pkl', 'rb') as f:
        valids = pickle.load(f, encoding='latin1')

with open("validationsubjects.csv", "w") as output:
    cw = csv.writer(output)
    for a in list(valids):
        cw.writerow([a])

with open("testsubjects.csv", "w") as output:
    cw = csv.writer(output)
    for a in list(testids):
        cw.writerow([a])
