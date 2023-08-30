import numpy as np
import matplotlib.pyplot as plt
import jax.numpy as jnp

class SineGaussian():

    def __init__(self,noise_std):
        """_summary_
        """
        self.times = jnp.linspace(0,10,150)
        self.noise_std = noise_std
        self.params_order = ["A", "f", "ph", "t0", "tau"]
        self.prior_ranges = {
            "A": (0.1,1),
            "ph": (0,0.0),
            "f": (0.1,1),
            "t0": (2,2),
            "tau": (0.2,0.2)
        }
        self.infer_names = ["A", "f"]
        self.batch_size = 2

    def __getitem__(self,idx):
        """_summary_

        Returns:
            _type_: _description_
        """
        norm_parameters, ps = self.sample_parameters(self.batch_size)
        model = np.zeros((self.batch_size, len(self.times)))
        for i in range(self.batch_size):
            model[i] = self.signal_model(*[parameters[key][i] for key in self.params_order])

        noise = self.make_noise(self.batch_size)

        return norm_parameters, model + noise

    def get_test_data(self, N):
        norm_parameters, parameters = self.sample_parameters(N)
        #model = self.sin_gaussian(parameters)
        model = np.zeros((N, len(self.times)))
        for i in range(N):
            model[i] = self.signal_model(*[parameters[key][i] for key in self.params_order])
        noise = self.make_noise(N)
        return parameters, model+noise, norm_parameters, model

    def make_noise(self, N):
        """_summary_

        Args:
            N (_type_): _description_

        Returns:
            _type_: _description_
        """
        samples = np.random.normal(0, self.noise_std, size = (N, len(self.times)))
        return samples

    def signal_model(self, A, f, ph, t0, tau):
        """_summary_

        Args:
            A (_type_): _description_
            f (_type_): _description_
            ph (_type_): _description_
            t0 (_type_): _description_
            tau (_type_): _description_

        Returns:
            _type_: _description_
        """
        sine = A*jnp.cos(2*jnp.pi*f * self.times + ph)
        #exp_window = jnp.exp(-(self.times - t0)**2/tau)
        return sine#*exp_window

    def sin_gaussian(self, params):
        """Sine gaussian model

        Args:
            params (_type_): _description_

        Returns:
            _type_: _description_
        """
        sine = jnp.expand_dims(params["A"], 1)*jnp.cos(2*np.pi*np.expand_dims(params["f"], 1) * jnp.expand_dims(self.times, 0) + jnp.expand_dims(params["ph"], 1))
        exp_window = jnp.exp(-(self.times - jnp.expand_dims(params["t0"], 1))**2/jnp.expand_dims(params["tau"], 1))

        return sine * exp_window

    def sin_gaussian_single(self, params):
        """Sine gaussian model

        Args:
            params (_type_): _description_

        Returns:
            _type_: _description_
        """
        sine = params["A"]*jnp.cos(2*np.pi*params["f"] * self.times + params["ph"])
        exp_window = jnp.exp(-(self.times - params["t0"])**2/params["tau"])

        return sine * exp_window

    def sample_parameters(self, N):
        """sample from the parameters (uniform in prior ranges)

        Args:
            N: number of parameters for each sample
        Returns:
            dict: dictionary of sampled parameters
        """
        params = {}
        norm_parameters = {}
        for key in self.prior_ranges:
            norm_parameters[key] = np.random.uniform(0,1, size=N)
            params[key] = norm_parameters[key] * (self.prior_ranges[key][1] - self.prior_ranges[key][0]) + self.prior_ranges[key][0]

        return norm_parameters, params

class Line():

    def __init__(self,noise_std):
        """_summary_
        """
        self.times = np.linspace(0,10,500)
        self.noise_std = noise_std
        self.prior_ranges = {
            "m": (0,2),
            "c": (0,1),
        }
        self.infer_names = ["m","c"]
        self.batch_size = 2

    def __getitem__(self,idx):
        """_summary_

        Returns:
            _type_: _description_
        """
        norm_parameters, parameters = self.sample_parameters(self.batch_size)
        model = self.sig_model(parameters)
        noise = self.make_noise(self.batch_size)

        return norm_parameters, model + noise

    def get_test_data(self, N):
        norm_parameters, parameters = self.sample_parameters(N)
        model = self.sig_model(parameters)
        noise = self.make_noise(N)
        return parameters, model+noise, norm_parameters, model

    def make_noise(self, N):
        """_summary_

        Args:
            N (_type_): _description_

        Returns:
            _type_: _description_
        """
        samples = np.random.normal(0, self.noise_std, size = (N, len(self.times)))
        return samples

    def sig_model(self, params):
        """Sine gaussian model

        Args:
            params (_type_): _description_

        Returns:
            _type_: _description_
        """
        line = np.expand_dims(params["m"], 1)*np.expand_dims(self.times, 0) + np.expand_dims(params["c"], 1)
        return line 

    def sample_parameters(self, N):
        """sample from the parameters (uniform in prior ranges)

        Args:
            N: number of parameters for each sample
        Returns:
            dict: dictionary of sampled parameters
        """
        params = {}
        norm_parameters = {}
        for key in self.prior_ranges:
            norm_parameters[key] = np.random.uniform(0,1, size=N)
            params[key] = norm_parameters[key] * (self.prior_ranges[key][1] - self.prior_ranges[key][0]) + self.prior_ranges[key][0]

        return norm_parameters, params

if __name__ == "__main__":

    mc = SineGaussian()

    data = mc[0]

    fig, ax = plt.subplots()
    ax.plot(data[1][0])

    fig.savefig("./data_test.png")