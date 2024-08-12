import main
datasets = ["emotions", "CAL500", "corel5k", "enron", "flags", "genbase", "scene", "medical", "yeast"]
subspaces = 20 #recommended
regularisation = 0.0 #not used
layers = 1 #recommended
maxepoch = 1500

for d in datasets:
    for taskID in range(1):
        main.main(layers, subspaces, d, maxepoch, regularisation, taskID)