import arff
import string
import numpy as np
from skmultilearn.model_selection import IterativeStratification
from sklearn import preprocessing

class StandardScaler:
    def __init__(self):
        self.mean = None
        self.std = None

    def fit(self, X):
        self.mean = np.mean(X, axis=0)
        self.std = np.std(X, axis=0)
        # To avoid division by zero, replace zeros with ones.
        self.std[self.std == 0] = 1

    def transform(self, X):
        return (X - self.mean) / self.std

    def fit_transform(self, X):
        self.fit(X)
        return self.transform(X)

class Data():
    def __init__(self, X,y,X2,Y2):
        self.X,self.y,selfX2,self.Y2 = X,y,X2,Y2

    def Data(self):
        return self.X, self.y

def get_splits():
    dirs = ['bibtex', 'bookmarks','CAL500', \
            'corel5k', 'delicious', 'emotions', \
            'enron', 'flags', 'genbase','mediamill', 'medical', \
            'scene', 'yeast']
    num_labels = [159, 208,174, \
                  374, 983, 6, \
                  53, 7, 27, 101, 45, \
                  6, 14]
    a = []
    a.extend(dirs[0:3])
    a.extend(['Corel5k'])
    a.extend(dirs[4:])
    arffs = []

    nominals = [False] * 13
    nominals[8] = True
    with open('dataset_dump.txt', 'w') as dp:
        for i, j, k, q in zip(dirs, a, nominals, num_labels):
            dp.write("### " + i + " ###\n")
            with open('datasets/{}/{}.arff'.format(i, j), 'r') as fp:
                arffs.append(arff.load(fp, encode_nominal=k))
            with open('stratified/{}_xtrain.txt'.format(i), 'w') as xtr, \
                    open('stratified/{}_ytrain.txt'.format(i), 'w') as ytr, \
                    open('stratified/{}_xtest.txt'.format(i), 'w') as xte, \
                    open('stratified/{}_ytest.txt'.format(i), 'w') as yte:
                for p in arffs[-1]:
                    if p != 'data':
                        dp.write("{}\n".format(p))
                        if p == 'attributes':
                            for pp, pi in zip(arffs[-1][p], range(len(arffs[-1][p]))):
                                dp.write("{}".format(pi) + "\t" + str(pp) + "\n")
                        else:
                            dp.write(str(arffs[-1][p]) + "\n")
                X = np.array([arffs[-1]['data'][iii][:-q] for iii in range(len(arffs[-1]['data']))]).astype(float)
                Y = np.array([arffs[-1]['data'][iii][-q:] for iii in range(len(arffs[-1]['data']))]).astype(float)
                k_fold = IterativeStratification(n_splits=2, order=1, sample_distribution_per_fold=[0.7, 0.3])
                for train, test in k_fold.split(X, Y):
                    X_train, X_test, Y_train, Y_test = X[train], X[test], Y[train], Y[test]
                for set, labels, d1, d2 in zip([X_train, X_test], [Y_train, Y_test], [xtr, xte], [ytr, yte]):
                    for inst in set:
                        for variable in inst[:-1]:
                            d1.write(str(variable) + ",")
                        d1.write(str(inst[-1]) + "\n")
                    for lbl in labels:
                        for clss in lbl[:-1]:
                            d2.write(str(clss) + ",")
                        d2.write(str(lbl[-1]) + "\n")
                print("{} features: {}, numl: {}\ntrain instances: {}\ntest instances: {}" \
                      .format(i, len(X_train[0]), len(Y_train[0]), len(X_train), len(X_test)))
    return

if __name__ == "__main__":
    get_splits()

def get_data(dataset):
    folder = "stratified/"
    prefix_x = "{}_x".format(dataset)
    prefix_y = "{}_y".format(dataset)
    X = []; Y = []; Xte = []; Yte = []
    with open(folder+prefix_x+'train.txt','r') as fp:
        for l1 in fp:
            X.append([float(ddd) for ddd in l1.replace("\n","").split(",")])
    with open(folder+prefix_y+'train.txt','r') as dp:
        for l2 in dp:
            Y.append([float(ddd) for ddd in l2.replace("\n","").split(",")])
    with open(folder+prefix_x+'test.txt','r') as fp, open(folder+prefix_y+'test.txt','r') as dp:
        for l1,l2 in zip(fp,dp):
            Xte.append([float(ddd) for ddd in l1.replace("\n","").split(",")])
            Yte.append([float(ddd) for ddd in l2.replace("\n","").split(",")])
    scaler = StandardScaler()
    X_train_scaled = scaler.fit_transform(X)
    X_test_scaled = scaler.transform(Xte)
    return X_train_scaled, np.array(Y), X_test_scaled, np.array(Yte)

def parser(file):
    matrix = []
    with open(file,'r') as fp:
        for line in fp:
            ln = np.array(line.replace("\n","").split(',')).astype(float)
            matrix.append(ln)
    return np.array(matrix)

def get_dataset(name):
    path = "stratified/"+name+'_{}.txt'
    sets = ['xtrain','ytrain','xtest','ytest']
    x1,y1,x2,y2 = get_data(name)
    return [x1,y1,x2,y2]