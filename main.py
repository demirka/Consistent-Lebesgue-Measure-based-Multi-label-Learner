from model import FeedForwardModel
import numpy as np
import datasets
import sys
import problem
from pymoo.indicators.hv.monte_carlo import ApproximateMonteCarloHypervolume
import utility
import time
import cmaes_fit
import cma

def main_loop(maxgen, x0, sigma0, optimisation, model, regularisation_strength, printable_params, X2, Y2):
    """
    :param maxgen: Maximum number of epoch.
    :param x0: Initial weights of f.
    :param sigma0: Initial step-size.
    :param optimisation: Optimisation object (problem object with helper functions).
    :param model: Model f object.
    :param regularisation_strength: Sparsity regularisation, unused currently.
    :param printable_params: For recording files.
    :param X2: Test X.
    :param Y2: Test Y.
    :return: Weights of f after \mathcal{O} epoch and execution time.
    """
    xnext = x0.copy()
    s = time.time()
    standard_loop(xnext, sigma0, maxgen, optimisation, model, regularisation_strength, printable_params, X2, Y2)
    r = time.time()-s
    return xnext, r

def standard_loop(x0, sigma0, maxgen, optimisation, model, regularisation_strength, printable_params, X2, Y2):
    """
    :param maxgen: Maximum number of epoch.
    :param x0: Initial weights of f.
    :param sigma0: Initial step-size.
    :param optimisation: Optimisation object (problem object with helper functions).
    :param model: Model f object.
    :param regularisation_strength: Sparsity regularisation, unused currently.
    :param printable_params: For recording files.
    :param X2: Test X.
    :param Y2: Test Y.
    :return: Hypervolumes over time (optional).
    """
    HVSS = []
    ref_point = [1] * 3
    ind = ApproximateMonteCarloHypervolume(ref_point=ref_point)
    es = cma.CMAEvolutionStrategy(x0, sigma0, {'verbose': 1, 'verb_disp': 1, 'verb_append': 1, 'verb_log': 0})
    progress = 1
    times = []
    bayes_risks = [np.inf,np.inf,np.inf,np.inf]
    conv_counter = 0
    old_res = [None,None,None]
    while not es.stop() and progress <= maxgen:
        sttt = time.time()
        print("Iteration: {}/{}".format(progress,maxgen))
        stt = time.time()
        print("\tAsking...")
        solutions = es.ask()
        print("\t{:.3f} minutes. Evaluating...".format((time.time()-stt)/60))
        stt = time.time()
        objectives = np.array(
            [optimisation.objective_function_helper(x, model) for x in
             solutions])
        classification_objectives = np.array([o[:-2] for o in objectives])
        sparsity = [regularisation_strength * o[-1] for o in objectives]
        print("\t{:.3f} minutes. Hypervolume contributions...".format((time.time()-stt)/60))
        stt = time.time()
        HV, HVC = ind._calc(ref_point, classification_objectives)
        HV_minimisation = np.array([1-hi for hi in HVC])
        fitness_values = np.array([hi for hi, s in zip(HV_minimisation, sparsity)])
        print("\t{:.3f} minutes. Telling...".format((time.time()-stt)/60))
        stt = time.time()
        es.tell(solutions, fitness_values)
        print("\t{:.3f} minutes. Logging...".format((time.time() - stt)/60))
        stt = time.time()
        res = es.result
        HVSS.append(HV)
        """
        Exploration vs. exploitation. Trajectory follows the incumbent mean then the incumbent solution.
        """
        xroll = np.random.randint(0,1)
        if xroll <= progress/maxgen:
            best = res.xbest
        else:
            best = res.xfavorite
        if None in old_res:
            old_res = utility.get_objectives(best, optimisation, model)
        else:
            new_res = utility.get_objectives(best, optimisation, model)
            if utility.is_converged(old_res,new_res):
                print("Convergence Warning, increasing from {} to {}".format(conv_counter,conv_counter+1))
                conv_counter+=1
                old_res = new_res
                if conv_counter == 3:
                    print("Converged, setting to incumbent mean")
                    best=res.xfavorite+np.random.normal(0,.001,len(res.xfavorite))
                    conv_counter = 0
            else:
                old_res = new_res
        utility.stat_logger(best, optimisation, model, printable_params, HVSS, progress, regularisation_strength)
        utility.record_weights(best, printable_params)
        utility.bayes_recorder(best, optimisation, model, printable_params, HVSS, progress, regularisation_strength,bayes_risks, X2, Y2)
        print("\t{:.3f} minutes. Done.".format((time.time() - stt)/ 60))
        times.append((time.time()-sttt)/60)
        print("Est. {:.3f} minutes remaining for {} iterations left out of {}.".format(np.mean(times)*(maxgen-progress), maxgen-progress, maxgen))
        progress += 1
    return HVSS

def main(layers, subspaces, dataset, maxgen, regularisation_strength, taskID):
    """
    :param layers: Number of hidden layers for f. Recommended setting = 1.
    :param subspaces: Number of subspaces C.
    :param dataset: Dataset name (parsed).
    :param maxgen: Maximum number of epoch.
    :param regularisation_strength: Optional.
    :param taskID: ID for repeated runs.
    :return: None.
    """
    X, Y, X2, Y2 = datasets.get_data(dataset)
    print("{} {}".format(X.shape, Y.shape))
    input = X.copy()
    features, labels, instances, test_instances = input.shape[1], Y.shape[1], input.shape[0], X2.shape[0]
    n_variables = subspaces*(features+labels+1)+labels+features+layers*(subspaces**2+subspaces)
    model = FeedForwardModel(layers, subspaces, features, labels)
    optimisation = problem.MLCAttentionProblem(input, Y, dataset, n_variables, features, labels, instances,
                                               subspaces, layers, X2, Y2)
    optimisation.stat_logger()
    std_dev = np.sqrt(1. / n_variables)
    x0 = np.random.randn(n_variables) * std_dev
    model.print()
    sigma0 = 1  # initial step-sizes
    printable_params = "{}_{}_{}_{}_{}_{}".format(dataset,layers,subspaces,regularisation_strength,maxgen,taskID)
    end_solution, time = main_loop(maxgen, x0, sigma0, optimisation, model, regularisation_strength, printable_params, X2, Y2)
    utility.time_recorder(time,printable_params)
    return


if __name__ == "__main__":
    """
    layers, subspaces, dataset, maxgen, regularisation_strength, taskID
    """
    main(int(sys.argv[1]), int(sys.argv[2]), sys.argv[3], int(sys.argv[4]), float(sys.argv[5]),
         float(sys.argv[6]))
