see example.py on example usage for all datasets

For example:
    python main.py 1 20 emotions 500 0.0 1

Arguments: layers, subspaces, dataset, maxgen, regularisation_strength, taskID
Parsed as: main(int(sys.argv[1]), int(sys.argv[2]), sys.argv[3], int(sys.argv[4]), float(sys.argv[5]),
                float(sys.argv[6]))
