import cma
import numpy as np

def hansen_cmaes_fit(subpopX, subpopid, idxX, decomposed_x, indices_X, sigma0, ind, ref_point, gen, maxgen, optimisation,
               ccframework, model, regularisation_strength):
    """
    :param subpopX:
    :param subpopid:
    :param idxX:
    :param decomposed_x:
    :param indices_X:
    :param sigma0:
    :param ind:
    :param ref_point:
    :param gen:
    :param maxgen:
    :param optimisation:
    :param ccframework:
    :param model:
    :param regularisation_strength:
    :return:
    """
    es = cma.CMAEvolutionStrategy(subpopX, sigma0, {'verbose':0, 'verb_disp': 0, 'verb_append':1, 'verb_log': 0})
    progress = gen / maxgen
    maxsubgen = int(maxgen / 10 * 2)
    HVS = []
    while not es.stop() and maxsubgen != 0:
        solutions = es.ask()
        objectives = np.array(
            [optimisation.objective_function(x, subpopid, idxX, decomposed_x, indices_X, ccframework, model) for x in
             solutions])
        classification_objectives = np.array([o[:-1] for o in objectives])
        sparsity = [regularisation_strength*o[-1] for o in objectives]
        HV, HVC = ind._calc(ref_point, classification_objectives)
        HVS.append(HV)
        HV_minimisation = np.array([1-hi for hi in HVC])
        fitness_values = np.array([hi+s for hi,s in zip(HV_minimisation,sparsity)]) #include the sparsity in here
        es.tell(solutions, fitness_values)
        maxsubgen -= 1
    res = es.result
    xroll = np.random.uniform(0, 1)
    if xroll <= progress:
        return res.xbest, HVS
    else:
        return res.xfavorite, HVS
    return

def fitness_helper(x, subpopX, subpopid, idxX, decomposed_x, indices_X, sigma0, ind, ref_point, gen, maxgen, optimisation,
               ccframework, model, regularisation_strength, HVS):
    objectives = np.array(
        [optimisation.objective_function(xi, subpopid, idxX, decomposed_x, indices_X, ccframework, model) for xi in
         x])
    classification_objectives = np.array([o[:-1] for o in objectives])
    sparsity = [regularisation_strength * o[-1] for o in objectives]
    HV, HVC = ind._calc(ref_point, classification_objectives)
    HV_minimisation = np.array([1 - hi for hi in HVC])
    HVS.append(HV)
    fitness_values = np.array([hi + s for hi, s in zip(HV_minimisation, sparsity)])  # include the sparsity in here
    return tf.convert_to_tensor(fitness_values)

def tensorflow_cmaes_fit(subpopX, subpopid, idxX, decomposed_x, indices_X, sigma0, ind, ref_point, gen, maxgen, optimisation,
               ccframework, model, regularisation_strength):

    HVS = []
    cma = CMA(
        initial_solution=subpopX,
        initial_step_size=sigma0,
        fitness_function=lambda x: fitness_helper(x, subpopX, subpopid, idxX, decomposed_x, indices_X, sigma0, ind, ref_point, gen, maxgen, optimisation,
               ccframework, model, regularisation_strength, HVS),
    )
    print("Subpop: {}".format(subpopid[0]))
    best_solution, best_fitness = cma.search(max_generations=int(maxgen / 10 * 2))
    print("Best fitness: {}".format(best_fitness))
    print("HV mu: {}".format(np.mean(HVS)))
    return best_solution, HVS