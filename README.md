# Consistent-Lebesgue-Measure-based-Multi-label-Learner

see example.py on example usage for all datasets

For example:
    python main.py 1 20 emotions 500 0.0 1

Arguments: layers, subspaces, dataset, maxgen, regularisation_strength, taskID
Parsed as: main(int(sys.argv[1]), int(sys.argv[2]), sys.argv[3], int(sys.argv[4]), float(sys.argv[5]),
                float(sys.argv[6]))

# Requirements
liac-arff 2.5.0

cma 3.3.0

scikit-learn 0.24.2

scikit-multilearn 0.2.0

scipy 1.5.4

numpy 1.19.5

pymoo 0.6.0.1

#Stratified IMDB-F
Stratified files for IMDB-F can be sent upon request.
