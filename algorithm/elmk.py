# -*- coding: utf-8 -*-
import numpy as np


class ELMKernel():

    def __init__(self, params=[]):
        super(self.__class__, self).__init__()

        self.regressor_name = "elmk"

        self.available_kernel_functions = ["rbf", "linear", "poly"]

        self.default_param_kernel_function = "rbf"
        self.default_param_c = 9
        self.default_param_kernel_params = [-15]

        self.output_weight = []
        self.training_patterns = []

        # Initialized parameters values
        if not params:
            self.param_kernel_function = self.default_param_kernel_function
            self.param_c = self.default_param_c
            self.param_kernel_params = self.default_param_kernel_params
        else:
            self.param_kernel_function = params[0]
            self.param_c = params[1]
            self.param_kernel_params = params[2]

    # ########################
    # Private Methods
    # ########################

    def _kernel_matrix(self, training_patterns, kernel_type, kernel_param,
                        test_patterns=None):
        number_training_patterns = training_patterns.shape[0]

        if kernel_type == "rbf":
            if test_patterns is None:
                temp_omega = np.dot(
                    np.sum(training_patterns ** 2, axis=1).reshape(-1, 1),
                    np.ones((1, number_training_patterns)))

                temp_omega = temp_omega + temp_omega.conj().T

                omega = np.exp(
                    -(2 ** kernel_param[0]) * (temp_omega - 2 * (np.dot(
                        training_patterns, training_patterns.conj().T))))

            else:
                number_test_patterns = test_patterns.shape[0]

                temp1 = np.dot(
                    np.sum(training_patterns ** 2, axis=1).reshape(-1, 1),
                    np.ones((1, number_test_patterns)))
                temp2 = np.dot(
                    np.sum(test_patterns ** 2, axis=1).reshape(-1, 1),
                    np.ones((1, number_training_patterns)))
                temp_omega = temp1 + temp2.conj().T

                omega = \
                    np.exp(- (2 ** kernel_param[0]) *
                           (temp_omega - 2 * np.dot(training_patterns,
                                                    test_patterns.conj().T)))
        elif kernel_type == "linear":
            if test_patterns is None:
                omega = np.dot(training_patterns, training_patterns.conj().T)
            else:
                omega = np.dot(training_patterns, test_patterns.conj().T)

        elif kernel_type == "poly":
            # Power a**x is undefined when x is real and 'a' is negative,
            # so is necessary to force an integer value
            kernel_param[1] = round(kernel_param[1])

            if test_patterns is None:
                temp = np.dot(training_patterns, training_patterns.conj().T)+ kernel_param[0]

                omega = temp ** kernel_param[1]
            else:
                temp = np.dot(training_patterns, test_patterns.conj().T)+ kernel_param[0]
                omega = temp ** kernel_param[1]

        else:
            print("Error: Invalid or unavailable kernel function.")
            return

        return omega

    def _local_train(self, training_patterns, training_expected_targets,
                     params):

        # If params not provided, uses initialized parameters values
        if not params:
            pass
        else:
            self.param_kernel_function = params[0]
            self.param_c = params[1]
            self.param_kernel_params = params[2]

        # Need to save all training patterns to perform kernel calculation at
        # testing and prediction phase
        self.training_patterns = training_patterns

        number_training_patterns = self.training_patterns.shape[0]

        # Training phase

        omega_train = self._kernel_matrix(self.training_patterns,
                                           self.param_kernel_function,
                                           self.param_kernel_params)

        self.output_weight = np.linalg.solve(
            (omega_train + np.eye(number_training_patterns) /
             (2 ** self.param_c)),
            training_expected_targets)

        training_predicted_targets = np.dot(omega_train, self.output_weight)

        return training_predicted_targets

    def _local_test(self, testing_patterns, testing_expected_targets,
                    predicting):

        omega_test = self._kernel_matrix(self.training_patterns,
                                          self.param_kernel_function,
                                          self.param_kernel_params,
                                          testing_patterns)

        testing_predicted_targets = np.dot(omega_test.conj().T,
                                           self.output_weight)

        return testing_predicted_targets


