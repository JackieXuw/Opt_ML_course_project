"""
Implement Bayesian optimizer for tuning hyperparameters.
"""
import numpy as np
from scipy.stats import norm
import GPy
import itertools


class TuneBO:

    def __init__(self, obj_fun, hyper_params_dict, init_hyper_param):
        self.obj_fun = obj_fun
        self.hyper_param_names = hyper_params_dict.keys()
        self.candidates = list(itertools.product(*list(hyper_params_dict[key]
                                                       for key in
                                                       self.hyper_param_names))
                               )
        self.kernel_var = 0.1
        self.noise_level = 0.1
        self.num_eps = 1e-6
        self.set_kernel()
        self.init_hyper_param = init_hyper_param
        self.evaluation_history = []   # each item: (point, performance_metric)
        self.setup_optimizer()

    def get_obj(self, hyper_param_point_dict):
        train_error, test_error, exec_time = \
            self.obj_fun(**hyper_param_point_dict)
        self.evaluation_history.append((hyper_param_point_dict,
                                        [train_error, test_error, exec_time]))
        train_error = np.expand_dims(np.array([train_error]), axis=1)
        test_error = np.expand_dims(np.array([test_error]), axis=1)
        exec_time = np.expand_dims(np.array([exec_time]), axis=1)
        return train_error, test_error, exec_time

    def set_kernel(self):
        self.train_error_kernel = GPy.kern.RBF(input_dim=
                                               len(self.candidates[0]),
                                               variance=self.kernel_var,
                                               lengthscale=1.0,
                                               ARD=True)

        self.test_error_kernel = GPy.kern.RBF(input_dim=
                                              len(self.candidates[0]),
                                              variance=self.kernel_var,
                                              lengthscale=1.0,
                                              ARD=True)

        self.exec_time_kernel = GPy.kern.RBF(input_dim=len(self.candidates[0]),
                                             variance=self.kernel_var,
                                             lengthscale=1.0,
                                             ARD=True)

    def setup_optimizer(self):
        # The statistical model of our objective function
        init_param_point = dict(zip(self.hyper_param_names,
                                    self.init_hyper_param))
        init_train_error, init_test_error, init_exec_time = \
            self.get_obj(init_param_point)
        init_X = np.expand_dims(np.array(self.init_hyper_param), axis=0)
        self.best_obj = init_test_error
        self.train_erro_gp = GPy.models.GPRegression(init_X,
                                                     init_train_error,
                                                     self.train_error_kernel,
                                                     noise_var=
                                                     self.noise_level ** 2)
        self.train_erro_gp.optimize()

        self.test_erro_gp = GPy.models.GPRegression(init_X,
                                                    init_test_error,
                                                    self.test_error_kernel,
                                                    noise_var=
                                                    self.noise_level ** 2)
        self.test_erro_gp.optimize()

        self.exec_time_gp = GPy.models.GPRegression(init_X,
                                                    init_exec_time,
                                                    self.exec_time_kernel,
                                                    noise_var=
                                                    self.noise_level ** 2)
        self.exec_time_gp.optimize()

    def get_acquisition(self, type='EI'):
        obj_mean, obj_var = self.test_erro_gp.predict(np.array(self.candidates))
        obj_mean = obj_mean.squeeze()
        obj_var = obj_var.squeeze()

        if type == 'EI':
            # calculate EI
            f_min = self.best_obj
            z = (f_min - obj_mean)/np.maximum(np.sqrt(obj_var), self.num_eps)
            EI = (f_min - obj_mean) * norm.cdf(z) + \
                np.sqrt(obj_var) * norm.pdf(z)
            return EI

    def make_step(self):
        acq = self.get_acquisition()
        maximizer = self.candidates[np.argmax(acq)]
        next_point = dict(zip(self.hyper_param_names,
                              maximizer))
        maximizer = np.expand_dims(maximizer, axis=0)
        # evaluate the next point's function
        train_error, test_error, exec_time = self.get_obj(next_point)
        self.best_obj = min(test_error[0, 0], self.best_obj)
        self.train_erro_gp.set_XY(np.vstack([self.train_erro_gp.X, maximizer]),
                                  np.vstack([self.train_erro_gp.Y, train_error])
                                  )
        self.train_erro_gp.optimize()

        self.test_erro_gp.set_XY(np.vstack([self.test_erro_gp.X, maximizer]),
                                  np.vstack([self.test_erro_gp.Y, test_error])
                                  )
        self.test_erro_gp.optimize()

        self.exec_time_gp.set_XY(np.vstack([self.exec_time_gp.X, maximizer]),
                                  np.vstack([self.exec_time_gp.Y, exec_time])
                                  )
        self.exec_time_gp.optimize()

        return train_error[0, 0], test_error[0, 0], exec_time[0, 0]
