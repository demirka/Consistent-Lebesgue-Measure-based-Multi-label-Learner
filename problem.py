import numpy as np
from sklearn.metrics import label_ranking_average_precision_score, f1_score, hamming_loss, log_loss
from sklearn.model_selection import ShuffleSplit, train_test_split
import warnings
import utility
warnings.simplefilter(action='ignore', category=FutureWarning)


class MLCAttentionProblem:

    def __init__(self, X, Y, dataset, n_variables, features, labels, instances, subspaces, layers, X2, Y2):
        self.dataset, self.n_variables, self.features, self.labels, self.instances, self.subspaces, self.layers, self.X, self.Y, self.X2, self.Y2, self.size = \
            dataset, n_variables, features, labels, instances, subspaces, layers, X, Y, X2, Y2, 0.2
        return

    def stat_logger(self):
        print(
            "Dataset: {}\nn_variables: {}\nfeatures: {}\nlabels: {}\ninstances: {}\nsubspaces: {}\nlayers: {}" \
                .format(self.dataset, self.n_variables, self.features, self.labels,
                        self.instances, self.subspaces, self.layers))
        return

    def l1_norm(self, solution):
        return np.linalg.norm(solution, ord=1)

    def objective_function(self, x, subpopid, idxX, decomposed_x, indices_X, ccframework, model):
        self.n_variables
        decomposed_copy = decomposed_x.copy()
        decomposed_copy[subpopid[0]] = x

        recomposed_representation = ccframework.recompose(decomposed_copy, indices_X)
        return self.objective_function_helper(recomposed_representation, model)

    def objective_function_helper(self, recomposed_representation, model):
        sparsity = self.l1_norm(recomposed_representation)

        E, D, E_a, B_l, B_E, B_D = utility.weight_extractor(self, recomposed_representation)
        model.set_weights(E, D, E_a, B_l, B_E, B_D)

        hl, mif1, avp, logloss = self.evaluate(model)

        return hl, mif1, avp, logloss, sparsity

    def evaluate(self, model):
        Xtr, Xval, Ytr, Yval = train_test_split(self.X,self.Y, test_size=self.size, random_state=2024)
        out_hl, out_mif1, out_avp, out_logloss = self.validation_helper(Xtr, Ytr, model)
        return out_hl, out_mif1, out_avp, out_logloss

    def validation_helper(self, X, Y, model):
        predY = model.inference(X)
        m = [np.inf,np.inf,np.inf,np.inf]
        for thresh in [0.5]:
            Yhat = predY.copy()
            Yhat[predY > thresh] = 1
            Yhat[predY <= thresh] = 0
            out_hl = hamming_loss(Yhat, Y)
            out_mif1 = 1 - f1_score(Y, Yhat, average='micro')
            out_avp = 1 - label_ranking_average_precision_score(Y, predY)
            out_loss = log_loss(Y, predY)
            m2 = [out_hl,out_mif1,out_avp,out_loss]
            for met in enumerate(m):
                if m2[met[0]]<m[met[0]]:
                    m[met[0]] = m2[met[0]]
        return m[0],m[1],m[2],m[3]

    def validate(self, model):
        Xtr, Xval, Ytr, Yval = train_test_split(self.X, self.Y, test_size=self.size, random_state=2024)
        out_hl, out_mif1, out_avp, out_loss = self.validation_helper(Xtr,Ytr,model)
        out_hl_val, out_mif1_val, out_avp_val, out_loss_val = self.validation_helper(self.X,self.Y,model)
        return out_hl, out_mif1, out_avp, out_loss, out_hl_val, out_mif1_val, out_avp_val, out_loss_val

    def evaluate_test(self, model, Xtest, Y):
        predY = model.test_inference(Xtest)
        m = [np.inf, np.inf, np.inf, np.inf]
        for thresh in [0.5]:
            Yhat = predY.copy()
            Yhat[predY > thresh] = 1
            Yhat[predY <= thresh] = 0
            out_hl = hamming_loss(Yhat, Y)
            out_mif1 = 1 - f1_score(Y, Yhat, average='micro')
            out_avp = 1 - label_ranking_average_precision_score(Y, predY)
            out_loss = log_loss(Y, predY)
            m2 = [out_hl, out_mif1, out_avp, out_loss]
            for met in enumerate(m):
                if m2[met[0]] < m[met[0]]:
                    m[met[0]] = m2[met[0]]
        return m[0], m[1], m[2], m[3]