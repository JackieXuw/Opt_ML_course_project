"""
Implement Bayesian optimizer for tuning hyperparameters.
"""
import numpy as np
from scipy.stats import norm
import GPy
import itertools
import math
from model import *
from torch import optim


def bayesian_run(num_hidden, num_layers, lr, momentum, mini_batch_size, \
                 num_epochs):
    train_model = Net(num_hidden=num_hidden, num_layers=num_layers)
    sgd = optim.SGD(train_model.parameters(), lr=lr, momentum=momentum)
    train_error, test_error, exec_time = run(train_model, sgd,
                                             mini_batch_size=mini_batch_size,
                                             num_epochs=num_epochs
                                             )
    return train_error, test_error, exec_time

class TuneBO:

    def __init__(self, obj_fun=bayesian_run, hyper_params_dict=None, \
                 init_hyper_param=None, parameters_range=None):
        self.parameters_range = parameters_range
        if parameters_range is not None:
            hyper_params_dict = self.transform_parameters()
        else:
            self.hyper_param_names = hyper_params_dict.keys()
            self.init_hyper_param = init_hyper_param
            space_type_dict = dict()
            for name in self.hyper_param_names:
                space_type_dict[name] = ('linspace',)
            self.space_type_dict = space_type_dict

        self.obj_fun = obj_fun
        self.candidates = list(itertools.product(*list(hyper_params_dict[key]
                                                       for key in
                                                       self.hyper_param_names))
                               )
        self.kernel_var = 0.1
        self.noise_level = 0.1
        self.num_eps = 1e-6
        self.set_kernel()
        self.evaluation_history = []   # each item: (point, performance_metric)
        self.setup_optimizer()

    def transform_parameters(self):
        parameters_range = self.parameters_range
        hyper_params_dict = dict()
        space_type_dict = dict()
        for name, (start, end, grid_size, space) in parameters_range.items():
            if space[:17] == 'discrete_linspace':
                hyper_params_dict[name] = np.linspace(start, end, grid_size,
                                                      dtype=int)
                space_type_dict[name] = ('discrete_linspace', )
            elif space[:8] == 'linspace':
                hyper_params_dict[name] = np.linspace(start, end, grid_size)
                space_type_dict[name] = ('linspace', )
            elif space[:17] == 'discrete_logspace':
                base = int(space[18:])
                start = math.log(start, base)
                end = math.log(end, base)
                hyper_params_dict[name] = np.linspace(start, end, grid_size)
                space_type_dict[name] = ('discrete_logspace', base)
            elif space[:8] == 'logspace':
                base = int(space[9:])
                start = math.log(start, base)
                end = math.log(end, base)
                hyper_params_dict[name] = np.linspace(start, end, grid_size)
                space_type_dict[name] = ('logspace', base)
            elif space == 'fixed' and grid_size == 1 and start == end:
                hyper_params_dict[name] = [start]
                space_type_dict[name] = ('fixed', )
            else:
                raise ValueError
        self.hyper_param_names = list(hyper_params_dict.keys())
        self.space_type_dict = space_type_dict
        init_hyper_param = []
        for name in self.hyper_param_names:
            val = np.random.choice(hyper_params_dict[name])
            init_hyper_param.append(val)
        self.init_hyper_param = init_hyper_param
        return hyper_params_dict

    def get_obj(self, hyper_param_point_dict):
        for name in self.hyper_param_names:
            if 'log' in self.space_type_dict[name][0]:
                base = self.space_type_dict[name][1]
                hyper_param_point_dict[name] = base ** \
                    hyper_param_point_dict[name]
                if 'discrete' in self.space_type_dict[name][0]:
                    hyper_param_point_dict[name] = int(
                        hyper_param_point_dict[name])
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

        time_mean, time_var = self.exec_time_gp.predict(np.array(
            self.candidates))
        time_mean = time_mean.squeeze()
        time_var = time_var.squeeze()

        if type == 'EI':
            # calculate EI
            f_min = self.best_obj
            z = (f_min - obj_mean)/np.maximum(np.sqrt(obj_var), self.num_eps)
            EI = (f_min - obj_mean) * norm.cdf(z) + \
                np.sqrt(obj_var) * norm.pdf(z)
            return EI

        if type == 'EIpC':
            # calculate EI per unit cost (the cost is exec time)
            f_min = self.best_obj
            z = (f_min - obj_mean)/np.maximum(np.sqrt(obj_var), self.num_eps)
            EI = (f_min - obj_mean) * norm.cdf(z) + \
                np.sqrt(obj_var) * norm.pdf(z)
            Cost = np.maximum(time_mean, 3) ** 0.5
            EIpC = EI/Cost
            return EIpC




    def make_step(self):
        acq = self.get_acquisition(type='EIpC')
        maximizer = self.candidates[np.argmax(acq)]
        next_point = dict(zip(self.hyper_param_names,
                              maximizer))
        maximizer = np.expand_dims(maximizer, axis=0)
        # evaluate the next point's function
        train_error, test_error, exec_time = self.get_obj(next_point)
        self.best_obj = min(test_error[0, 0], self.best_obj)

        try:
            X = self.train_erro_gp.X
            Y = self.train_erro_gp.Y
            self.train_erro_gp.set_XY(np.vstack(
                [self.train_erro_gp.X, maximizer]
            ),
                                  np.vstack(
                                      [self.train_erro_gp.Y, train_error])
                                  )
            self.train_erro_gp.optimize()
        except Exception as e:
            print(e.message, e.args)
            self.train_erro_gp.set_XY(X, Y)
            self.train_erro_gp.optimize()

        try:
            X = self.test_erro_gp.X
            Y = self.test_erro_gp.Y
            self.test_erro_gp.set_XY(np.vstack(
                [self.test_erro_gp.X, maximizer]),
                                  np.vstack(
                                      [self.test_erro_gp.Y, test_error])
                                  )
            self.test_erro_gp.optimize()
        except Exception as e:
            print(e.message, e.args)
            self.test_erro_gp.set_XY(X, Y)
            self.test_erro_gp.optimize()

        try:
            X = self.exec_time_gp.X
            Y = self.exec_time_gp.Y
            self.exec_time_gp.set_XY(np.vstack([self.exec_time_gp.X, maximizer]),
                                  np.vstack([self.exec_time_gp.Y, exec_time])
                                  )
            self.exec_time_gp.optimize()
        except Exception as e:
            print(e.message, e.args)
            self.exec_time_gp.set_XY(X, Y)
            self.optimize()

        return train_error[0, 0], test_error[0, 0], exec_time[0, 0]

    def run(self, num_evals=20):
        for _ in range(num_evals):
            self.make_step()
