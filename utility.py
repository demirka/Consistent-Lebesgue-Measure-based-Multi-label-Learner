import numpy as np
import os

def weight_extractor(problem, recomposed_representation):
    """
    :param problem: Problem object.
    :param recomposed_representation: Full vector representation of f.
    :return: Weights of each component of f in vector form.
    """
    """
    Extracting E and D
    """
    encoder_size = problem.features * problem.subspaces
    transformer_size = encoder_size + problem.labels * problem.subspaces
    E = recomposed_representation[:encoder_size]
    D = recomposed_representation[encoder_size:transformer_size]
    """
    Extracting L and B_l
    """
    E_a, B_l = [], []
    layer_encoder_size = problem.subspaces ** 2
    layer_encoder_start = transformer_size
    for _ in range(problem.layers):
        E_a.append(recomposed_representation[ \
                   layer_encoder_start: \
                   layer_encoder_start + layer_encoder_size])
        layer_encoder_start += layer_encoder_size
    layer_bias_start = layer_encoder_start
    for _ in range(problem.layers):
        B_l.append(recomposed_representation[layer_bias_start:layer_bias_start + problem.subspaces])
        layer_bias_start += problem.subspaces
    encoder_bias_start = layer_bias_start
    """
    Extracting B_E and B_D
    """
    B_E = recomposed_representation[encoder_bias_start:encoder_bias_start + problem.subspaces]
    decoder_bias_start = encoder_bias_start + problem.subspaces
    B_D = recomposed_representation[decoder_bias_start:decoder_bias_start + problem.labels]
    return E, D, np.array(E_a), np.array(B_l), B_E, B_D

def record_weights(xsolution, printable_params):
    file_path = "dump/" + printable_params + "_weights.data"
    with open(file_path, 'w') as fp:
        for var in xsolution:
            fp.write("{:.6f}\n".format(var))
    return

def time_recorder(time, printable_params):
    file_path = "dump/" + printable_params + "_time.data"
    with open(file_path, 'a') as fp:
        fp.write("{:.6f}\n".format(time))
    return

def record_final_solution(xsolution, optimisation, model, printable_params, Xtest, Ytest):
    sparsity = optimisation.l1_norm(xsolution)
    E, D, E_a, B_l, B_E, B_D = weight_extractor(optimisation, xsolution)
    model.set_weights(E, D, E_a, B_l, B_E, B_D)
    hl, mif1, avp, log_loss = optimisation.validate(model)
    hl_te, mif1_te, avp_te, log_loss_te = optimisation.evaluate_test(model, Xtest, Ytest)
    file_path = "dump/" + printable_params + "_final.data"
    if not os.path.exists(file_path):
        with open(file_path, 'a') as fp:
            fp.write("hl, mif1, avp, log_loss, hl_te, mif1_te, avp_te, log_loss_te, sparsity, weight_mu, weight_std\n")
    with open(file_path, 'a') as fp:
        fp.write("{:.6f}, {:.6f}, {:.6f}, {:.6f}, {:.6f}, {:.6f}, {:.6f}, {:.6f}, {:.6f}, {:.6f}, {:.6f}\n".format(\
            hl, mif1, avp, log_loss, hl_te, mif1_te, avp_te, log_loss_te, sparsity, np.mean(xsolution), \
            np.std(xsolution)))
    return

def stat_logger(xsolution, optimisation, model, printable_params, HVSS, gen, regularisation_strength):
    sparsity = optimisation.l1_norm(xsolution)
    E, D, E_a, B_l, B_E, B_D = weight_extractor(optimisation, xsolution)
    model.set_weights(E, D, E_a, B_l, B_E, B_D)
    hl, mif1, avp, log_loss, hl_val, mif1_val, avp_val, log_loss_val = optimisation.validate(model)
    file_path = "dump/"+printable_params+".data"
    if not os.path.exists(file_path):
        with open(file_path, 'a') as fp:
            fp.write("iter, hl, mif1, avp, log_loss, hl_val, mif1_val, avp_val, log_loss_val, sparsity, HV_mu, HV_max, HV_min, HV_std, weight_mu, weight_std\n")
    with open(file_path, 'a') as fp:
        fp.write(
            "{}, {:.6f}, {:.6f}, {:.6f}, {:.6f}, {:.6f}, {:.6f}, {:.6f}, {:.6f}, {:.6f}, {:.6f}, {:.6f}, {:.6f}, {:.6f}, {:.6f}, {:.6f}\n".format( \
                gen, hl, mif1, avp, log_loss, hl_val, mif1_val, avp_val,
                log_loss_val, sparsity, np.mean(HVSS), np.max(HVSS), np.min(HVSS), np.std(HVSS), np.mean(xsolution), \
                np.std(xsolution)))
    return

def bayes_risk(obtained,bayes):
    return True if obtained<bayes else False

def get_objectives(xsolution, optimisation, model):
    sparsity = optimisation.l1_norm(xsolution)
    E, D, E_a, B_l, B_E, B_D = weight_extractor(optimisation, xsolution)
    model.set_weights(E, D, E_a, B_l, B_E, B_D)
    return optimisation.validate(model)

def is_converged(obj1, obj2):
    if any(abs(np.array(obj1)-np.array(obj2))<=1e-4):
        return True
    return False

def bayes_helper(hl,mif1,avp,log_loss, bayes_recorder,printable_params,xsolution,hl_te, mif1_te, avp_te, log_loss_te, sparsity, gen, hl_val, mif1_val, avp_val, log_loss_val):
    for obtained,actual,indx,flag in zip([hl_val, mif1_val, avp_val, log_loss_val],bayes_recorder,enumerate(bayes_recorder),["hl", "mif1", "avp", "logloss"]):
        if bayes_risk(obtained,actual):
            bayes_recorder[indx[0]] = obtained
            file_path = "dump/" + printable_params + "_{}_bayes_risk.data".format(flag)
            record_weights(xsolution,printable_params+"_"+flag)
            with open(file_path, 'w') as fp:
                fp.write("iter, hl, mif1, avp, log_loss, hl_te, mif1_te, avp_te, log_loss_te, hl_val, mif1_val, avp_val, log_loss_val, sparsity, weight_mu, weight_std\n")
                fp.write("{},{:.6f}, {:.6f}, {:.6f}, {:.6f}, {:.6f}, {:.6f}, {:.6f}, {:.6f}, {:.6f}, {:.6f}, {:.6f}, {:.6f}, {:.6f}, {:.6f}, {:.6f}\n".format( \
                            gen, hl, mif1, avp, log_loss, hl_te, mif1_te, avp_te, log_loss_te, hl_val, mif1_val, avp_val, log_loss_val, sparsity, np.mean(xsolution), \
                            np.std(xsolution)))
    return
def bayes_recorder(xsolution, optimisation, model, printable_params, HVSS, gen, regularisation_strength, bayes_recorder, Xtest, Ytest):
    sparsity = optimisation.l1_norm(xsolution)
    E, D, E_a, B_l, B_E, B_D = weight_extractor(optimisation, xsolution)
    model.set_weights(E, D, E_a, B_l, B_E, B_D)
    hl, mif1, avp, log_loss, hl_val, mif1_val, avp_val, log_loss_val = optimisation.validate(model)
    hl_te, mif1_te, avp_te, log_loss_te = optimisation.evaluate_test(model, Xtest, Ytest)
    bayes_helper(hl,mif1,avp,log_loss, bayes_recorder,printable_params,xsolution,hl_te, mif1_te, avp_te, log_loss_te, sparsity, gen, hl_val, mif1_val, avp_val, log_loss_val)
    return