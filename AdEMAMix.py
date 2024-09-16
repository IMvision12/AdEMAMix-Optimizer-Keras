from keras.src import ops
from keras.src.optimizers import optimizer

class AdEMAMix(optimizer.Optimizer):
    """Optimizer that implements the AdEMAMix algorithm.

    AdEMAMix optimization is based on adaptive estimation of first-order,
    second-order moments, and an additional slow-moving average.

    Args:
        learning_rate: A float, a `keras.optimizers.schedules.LearningRateSchedule` instance,
            or a callable that takes no arguments and returns the actual value to use.
            The learning rate. Defaults to `0.001`.
        beta_1: The exponential decay rate for the 1st moment estimates. Defaults to `0.9`.
        beta_2: The exponential decay rate for the 2nd moment estimates. Defaults to `0.999`.
        beta_3: The exponential decay rate for the slow-moving average. Defaults to `0.9999`.
        epsilon: A small constant for numerical stability. Defaults to `1e-7`.
        alpha: Scaling factor for the slow-moving average. Defaults to `5.0`.
        T_alpha_beta3: Time step for the `alpha` and `beta_3` decay. Defaults to `None`.
        weight_decay: A float value for L2 weight regularization.
        {{base_optimizer_keyword_args}}
    """
    def __init__(self,
                 learning_rate=0.001,
                 beta_1=0.9,
                 beta_2=0.999,
                 beta_3=0.9999,
                 epsilon=1e-7,
                 alpha=5.0,
                 T_alpha_beta3=None,
                 weight_decay=None,
                 clipnorm=None,
                 clipvalue=None,
                 global_clipnorm=None,
                 use_ema=False,
                 ema_momentum=0.99,
                 ema_overwrite_frequency=None,
                 loss_scale_factor=None,
                 gradient_accumulation_steps=None,
                 name="AdEMAMix",
                 **kwargs):
        super().__init__(learning_rate=learning_rate,
                         name=name,
                         weight_decay=weight_decay,
                         clipnorm=clipnorm,
                         clipvalue=clipvalue,
                         global_clipnorm=global_clipnorm,
                         use_ema=use_ema,
                         ema_momentum=ema_momentum,
                         ema_overwrite_frequency=ema_overwrite_frequency,
                         loss_scale_factor=loss_scale_factor,
                         gradient_accumulation_steps=gradient_accumulation_steps,
                         **kwargs)
        self.beta_1 = beta_1
        self.beta_2 = beta_2
        self.beta_3 = beta_3
        self.epsilon = epsilon
        self.alpha = alpha
        self.T_alpha_beta3 = T_alpha_beta3

    def build(self, var_list):
        """Initialize optimizer variables."""
        if self.built:
            return
        super().build(var_list)
        self._momentums = []
        self._velocities = []
        self._exp_avg_slow = []
        for var in var_list:
            self._momentums.append(
                self.add_variable_from_reference(var, "momentum")
            )
            self._velocities.append(
                self.add_variable_from_reference(var, "velocity")
            )
            self._exp_avg_slow.append(
                self.add_variable_from_reference(var, "exp_avg_slow")
            )

    def update_step(self, gradient, variable, learning_rate):
        """Update step given gradient and the associated model variable."""
        lr = ops.cast(learning_rate, variable.dtype)
        gradient = ops.cast(gradient, variable.dtype)
        local_step = ops.cast(self.iterations + 1, variable.dtype)

        beta_1_power = ops.power(ops.cast(self.beta_1, variable.dtype), local_step)
        beta_2_power = ops.power(ops.cast(self.beta_2, variable.dtype), local_step)

        # Retrieve optimizer variables
        m = self._momentums[self._get_variable_index(variable)]
        v = self._velocities[self._get_variable_index(variable)]
        exp_avg_slow = self._exp_avg_slow[self._get_variable_index(variable)]

        alpha_t = self.alpha
        beta_3_t = self.beta_3
        if self.T_alpha_beta3 is not None:
            alpha_t = ops.minimum(local_step * alpha_t / self.T_alpha_beta3, alpha_t)
            beta_3_t = ops.minimum(ops.exp(ops.log(self.beta_1) * ops.log(self.beta_3) /
                                           ((1 - local_step / self.T_alpha_beta3) * ops.log(self.beta_3) +
                                            (local_step / self.T_alpha_beta3) * ops.log(self.beta_1))),
                                   self.beta_3)

        bias_correction1 = 1 - beta_1_power
        bias_correction2 = 1 - beta_2_power

        # Update first and second moments
        self.assign_add(m, ops.multiply(ops.subtract(gradient, m), 1 - self.beta_1))
        self.assign_add(v, ops.multiply(ops.subtract(ops.square(gradient), v), 1 - self.beta_2))
        self.assign_add(exp_avg_slow, ops.multiply(ops.subtract(gradient, exp_avg_slow), 1 - beta_3_t))

        # Compute the denominator (with bias correction)
        denom = ops.sqrt(v) / ops.sqrt(bias_correction2) + self.epsilon
        step_size = lr / bias_correction1

        # Apply weight decay
        if self.weight_decay is not None and self.weight_decay != 0:
            variable.assign_sub(lr * self.weight_decay * variable)

        # Update the variable
        update = m + alpha_t * exp_avg_slow
        variable.assign_sub(step_size * ops.divide(update, denom))

    def get_config(self):
        config = super().get_config()
        config.update({
            "beta_1": self.beta_1,
            "beta_2": self.beta_2,
            "beta_3": self.beta_3,
            "epsilon": self.epsilon,
            "alpha": self.alpha,
            "T_alpha_beta3": self.T_alpha_beta3,
        })
        return config
