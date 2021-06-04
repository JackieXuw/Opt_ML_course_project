"""
Implement Bayesian optimizer for tuning hyperparameters.
"""
import numpy as np
import GPy
import itertools
import

class TuneBO:

    def __init__(self, obj_fun, hyper_params_dict, init_hyper_param):
        self.obj_fun = obj_fun
        self.hyper_param_names = hyper_params_dict.keys()
        self.candidates = list(itertools.product(*list(hyper_params_dict[key]
                                                       for key in
                                                       self.hyper_param_names))
                               )
        self.kernel_var = 0.1
        self.set_kernel()
        self.setup_optimizer()
        self.init_hyper_param = init_hyper_param

    def get_obj(self, hyper_param):





    def set_kernel(self):
        self.obj_kernel = GPy.kern.RBF(input_dim=len(self.candidates[0]),
                                    variance=self.kernel_var,
                                lengthscale=1.0,
                                ARD=True)

    def setup_optimizer(self):
        # The statistical model of our objective function
        self.gp_obj = GPy.models.GPRegression(self.x0_arr,
                                              init_obj_val_arr,
                                              self.kernel_list[0],
                                              noise_var=self.noise_level ** 2)

        self.gp_constr_list = []
        for i in range(self.opt_problem.num_constrs):
            self.gp_constr_list.append(
                GPy.models.GPRegression(self.x0_arr,
                                        np.expand_dims(
                                            init_constr_val_arr[:, i], axis=1),
                                        self.kernel_list[i+1],
                                        noise_var=self.noise_level ** 2))

        self.opt = safeopt.SafeOpt([self.gp_obj] + self.gp_constr_list,
                                   self.parameter_set,
                                   [-np.inf] + [0.] *
                                   self.opt_problem.num_constrs,
                                   lipschitz=None,
                                   threshold=0.1
                                   )
        self.cumu_vio_cost = np.zeros(self.opt_problem.num_constrs)

    def get_obj_constr_val(self, x_arr, noise=False):
        obj_val_arr, constr_val_arr = self.opt_problem.sample_point(x_arr)
        obj_val_arr = -1 * obj_val_arr
        constr_val_arr = -1 * constr_val_arr
        return obj_val_arr, constr_val_arr

    def plot(self):
        # Plot the GP
        self.opt.plot(100)
        # Plot the true function
        y, constr_val = self.get_obj_constr_val(self.parameter_set,
                                                noise=False)

    def make_step(self):
        x_next = self.opt.optimize()
        x_next = np.array([x_next])
        # Get a measurement from the real system
        y_obj, constr_vals = self.get_obj_constr_val(x_next)
        if np.all(constr_vals >= 0):
            self.best_obj = max(self.best_obj, y_obj[0, 0])
        y_meas = np.hstack((y_obj, constr_vals))
        violation_cost = self.opt_problem.get_total_violation_cost(-constr_vals)
        violation_total_cost = np.sum(violation_cost, axis=0)
        self.cumu_vio_cost = self.cumu_vio_cost + violation_total_cost
        # Add this to the GP model
        self.opt.add_new_data_point(x_next, y_meas)
        return y_obj, constr_vals
