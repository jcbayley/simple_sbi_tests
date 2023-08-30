from signal import SIGRTMAX
from tarfile import LENGTH_LINK
import vitamin
import numpy as np
import torch
import os
import pickle
from torchsummary import summary
import matplotlib.pyplot as plt
import scipy.stats as st
from generate_data import SineGaussian, Line
import torch.nn as nn
import corner
import bilby
import pandas


def setup_model(num_params,length, device="cpu", continue_train = False, checkpoint_dir=""):
    #Set up the CVAE parmaeters and bounds
    #parameters to infer and the output distributions of the CVAE (default is Truncated Normal)
    inf_pars = {f"p{i}":"TruncatedNormal" for i in range(num_params)}
    # the bounds for each of the parameters, used to rescale parameters internally
    bounds = {f"p{i}_{bnd}":val for i in range(num_params) for bnd,val in zip(["min","max"], [0,1])}
    # layers shared across the three networks (all available layers are defined in Docs)
    # three individual networks designs to come after the shared network                                                                                                  
    #r1_network = ['Linear(64)', 'Linear(32)', 'Linear(16)']
    #r2_network = ['Linear(64)', 'Linear(32)', 'Linear(16)']
    #q_network = ['Linear(64)', 'Linear(32)', 'Linear(16)']

    shared_conv = nn.Sequential(
        nn.Conv2d(2, 32, (5,32), padding="same"),
        nn.ReLU(),
        nn.Conv2d(32, 32, (9,9), padding="same"),
        nn.MaxPool2d(2),
        nn.ReLU(),
        nn.Conv2d(32, 32, (5,5), padding="same"),
        nn.MaxPool2d(2),
        nn.ReLU(),
        nn.Conv2d(32, 8, (3,3), padding="same"),
        nn.MaxPool2d(2),
        nn.ReLU(),
    )

    r1_network = nn.Sequential(
        nn.LazyLinear(256),
        nn.ReLU(),
        nn.LazyLinear(128),
        nn.ReLU(),
        nn.LazyLinear(64),
    )

    q_network = nn.Sequential(
        nn.LazyLinear(256),
        nn.ReLU(),
        nn.LazyLinear(128),
        nn.ReLU(),
        nn.LazyLinear(64),
    )

    r2_network = nn.Sequential(
        nn.LazyLinear(256),
        nn.ReLU(),
        nn.LazyLinear(128),
        nn.ReLU(),
        nn.LazyLinear(64),
    )


    # initialise the model 
    model = vitamin.vitamin_model.CVAE(z_dim=8, # latent space size
                                    n_modes = 1, # number of modes in the latent space
                                    x_dim = num_params,  # number of parameters to infer
                                    inf_pars=inf_pars, # inference parameters
                                    bounds=bounds, # inference parameters bounds
                                    y_dim=length[2], # number of datapoints
                                    n_channels=length[0], # number of input channels
                                    shared_conv=shared_conv,
                                    r1_network=r1_network,
                                    r2_network=r2_network,
                                    q_network=q_network,
                                    logvarmin=True,
                                    device = device).to(device)

    if continue_train:
        checkpoint = torch.load(os.path.join(checkpoint_dir,"model.pt"), map_location=device)
        model.load_state_dict(checkpoint["model_state_dict"])

    model.forward(torch.ones((1, length[0], length[1], length[2])).to(device), torch.ones((1, model.x_dim)).to(device))
    summary(model, [(length[0], length[1], length[2]), (num_params, )])

    optimiser = torch.optim.AdamW(model.parameters(), lr=5e-4, weight_decay=1e-5)
    scheduler = torch.optim.lr_scheduler.LinearLR(optimiser, start_factor=1.0, end_factor=0.01, total_iters=200)

    if continue_train:
        optimiser.load_state_dict(checkpoint["optimiser_state_dict"])

    model.to(torch.float32)
    model.to(device)

    return model, optimiser, scheduler

def pp_plot(results, filename=None, save=True, confidence_interval=[0.68, 0.95, 0.997],
                 lines=None, legend_fontsize='x-small', keys=None, title=True,
                 confidence_interval_alpha=0.1, weight_list=None,
                 **kwargs):
    """
    Make a P-P plot for a set of runs with injected signals.

    Parameters
    ==========
    results: list
        A list of Result objects, each of these should have injected_parameters
    filename: str, optional
        The name of the file to save, the default is "outdir/pp.png"
    save: bool, optional
        Whether to save the file, default=True
    confidence_interval: (float, list), optional
        The confidence interval to be plotted, defaulting to 1-2-3 sigma
    lines: list
        If given, a list of matplotlib line formats to use, must be greater
        than the number of parameters.
    legend_fontsize: float
        The font size for the legend
    keys: list
        A list of keys to use, if None defaults to search_parameter_keys
    title: bool
        Whether to add the number of results and total p-value as a plot title
    confidence_interval_alpha: float, list, optional
        The transparency for the background condifence interval
    weight_list: list, optional
        List of the weight arrays for each set of posterior samples.
    kwargs:
        Additional kwargs to pass to matplotlib.pyplot.plot

    Returns
    =======
    fig, pvals:
        matplotlib figure and a NamedTuple with attributes `combined_pvalue`,
        `pvalues`, and `names`.
    """
    import matplotlib.pyplot as plt

    if keys is None:
        keys = results[0].search_parameter_keys

    if weight_list is None:
        weight_list = [None] * len(results)

    credible_levels = list()
    print(keys)
    for i, result in enumerate(results):
        cr_level = result.get_all_injection_credible_levels(keys, weights=weight_list[i])
        credible_levels.append(cr_level)
        
    credible_levels = pandas.DataFrame(credible_levels)

    print(credible_levels)

    if lines is None:
        colors = ["C{}".format(i) for i in range(8)]
        linestyles = ["-", "--", ":"]
        lines = ["{}{}".format(a, b) for a, b in product(linestyles, colors)]
    if len(lines) < len(credible_levels.keys()):
        raise ValueError("Larger number of parameters than unique linestyles")

    x_values = np.linspace(0, 1, 1001)

    N = len(credible_levels)
    fig, ax = plt.subplots()

    if isinstance(confidence_interval, float):
        confidence_interval = [confidence_interval]
    if isinstance(confidence_interval_alpha, float):
        confidence_interval_alpha = [confidence_interval_alpha] * len(confidence_interval)
    elif len(confidence_interval_alpha) != len(confidence_interval):
        raise ValueError(
            "confidence_interval_alpha must have the same length as confidence_interval")

    for ci, alpha in zip(confidence_interval, confidence_interval_alpha):
        edge_of_bound = (1. - ci) / 2.
        lower = scipy.stats.binom.ppf(1 - edge_of_bound, N, x_values) / N
        upper = scipy.stats.binom.ppf(edge_of_bound, N, x_values) / N
        # The binomial point percent function doesn't always return 0 @ 0,
        # so set those bounds explicitly to be sure
        lower[0] = 0
        upper[0] = 0
        ax.fill_between(x_values, lower, upper, alpha=alpha, color='k')

    pvalues = []
    logger.info("Key: KS-test p-value")
    for ii, key in enumerate(credible_levels):
        pp = np.array([sum(credible_levels[key].values < xx) /
                       len(credible_levels) for xx in x_values])
        pvalue = scipy.stats.kstest(credible_levels[key], 'uniform').pvalue
        pvalues.append(pvalue)
        logger.info("{}: {}".format(key, pvalue))

        try:
            name = results[0].priors[key].latex_label
        except AttributeError:
            name = key
        label = "{} ({:2.3f})".format(name, pvalue)
        plt.plot(x_values, pp, lines[ii], label=label, **kwargs)

    Pvals = namedtuple('pvals', ['combined_pvalue', 'pvalues', 'names'])
    pvals = Pvals(combined_pvalue=scipy.stats.combine_pvalues(pvalues)[1],
                  pvalues=pvalues,
                  names=list(credible_levels.keys()))
    logger.info(
        "Combined p-value: {}".format(pvals.combined_pvalue))

    if title:
        ax.set_title("N={}, p-value={:2.4f}".format(
            len(results), pvals.combined_pvalue))
    ax.set_xlabel("C.I.")
    ax.set_ylabel("Fraction of events in C.I.")
    ax.legend(handlelength=2, labelspacing=0.25, fontsize=legend_fontsize)
    ax.set_xlim(0, 1)
    ax.set_ylim(0, 1)
    fig.tight_layout()
    if save:
        if filename is None:
            filename = 'outdir/pp.png'
        safe_save_figure(fig=fig, filename=filename, dpi=500)

    return fig, pvals


def make_pp_plot(samples, truths, savepath, labels):

    results = []
    for sp in range(len(samples)):
        res = bilby.result.Result()
        post = pandas.DataFrame(data = samples[sp], columns = labels)
        res.posterior = post
        res.search_parameter_keys = labels
        res.injection_parameters = {labels[i]:truths[sp][i] for i in range(len(labels))}
        res.priors = {labels[i]:bilby.prior.Gaussian(0,1, name=labels[i]) for i in range(len(labels))}
        results.append(res)

    fig, pv = pp_plot(results, filename = savepath)


def custom_pp_plot(samples, truths, savepath, labels):

    cred_region = []
    for i,res in enumerate(samples):
        test_cred = []
        for j,par in enumerate(labels):
            cred_reg = sum(res[:,j] < truths[i,j])/len(res[:,j])
            test_cred.append(cred_reg)
        cred_region.append(test_cred)
    cred_region = np.array(cred_region)

    fig, ax = plt.subplots()

    x_values = np.linspace(0,1,100)
    N = len(cred_region)

    confidence_interval=[0.68, 0.95, 0.997]
    confidence_interval_alpha=[0.1] * len(confidence_interval)
    for ci, alpha in zip(confidence_interval, confidence_interval_alpha):
        edge_of_bound = (1. - ci) / 2.
        lower = st.binom.ppf(1 - edge_of_bound, N, x_values) / N
        upper = st.binom.ppf(edge_of_bound, N, x_values) / N
        # The binomial point percent function doesn't always return 0 @ 0,
        # so set those bounds explicitly to be sure
        lower[0] = 0
        upper[0] = 0
        ax.fill_between(x_values, lower, upper, alpha=alpha, color='k')

    
    ax.plot(x_values, x_values, "k")
    for i, par in enumerate(labels):
        pp = np.array([sum(cred_region[:,i] < x)/len(cred_region[:,i]) for x in x_values])

        ax.plot(x_values, pp, label=par)
    

    ax.legend()
    fig.savefig(savepath)

class TestCallback():
    def __init__(self, model, test_data_array, save_directory, test_interval = 100, make_pp_plot=True, device="cpu"):
        self.model=model
        self.test_data_array = test_data_array
        self.test_interval = test_interval
        self.save_directory = save_directory
        self.make_pp_plot = make_pp_plot
        self.device=device

    def on_epoch_end(self, epoch, logs={}):

        if epoch == "end" or epoch % self.test_interval == 0:
            tdat,ttruth = self.test_data_array
            self.model.device=self.device
            self.model.to(self.device)
            posts = self.model.test(tdat.to(self.device), 1000)
            if not os.path.isdir(os.path.join(self.save_directory, f"epoch_{epoch}")):
                os.makedirs(os.path.join(self.save_directory, f"epoch_{epoch}"))
            if self.make_pp_plot:
                custom_pp_plot(posts, ttruth.numpy(), os.path.join(self.save_directory, f"epoch_{epoch}", "ppplot.png"), ["A", "phase", "frequency", "t0", "tau"])

            for i,post in enumerate(posts):
                fig = corner.corner(post, truths=ttruth[i])
                fig.savefig(os.path.join(self.save_directory, f"epoch_{epoch}", f"test_posterior_{i}.png"))

                #fig, ax = plt.subplots()
                #hst = ax.hist(post[:,0])
                #ax.axvline(ttruth[i][0], color="C2")
                #fig.savefig(os.path.join(self.save_directory, f"epoch_{epoch}", f"single_test_posterior_{i}.png"))
            

def run_train_vitamin(base_dir = "./test"):
    length = 10000
    times = np.linspace(0,100,length)
    standard_deviation = 3
    n_train, n_val, n_test = int(16), int(5e2), int(1e2)
    #n_sinusoids = 0
    #seglen = 300
    #frequency_range = (0.1,200)
    #A_range = (0.01,2)
    #tau_range = (0.01,10)

    train_data = SineGaussian(
        noise_std=noise_std
        )
    
    validation_data = SineGaussian(
        noise_std=noise_std
        )

    test_data = SineGaussian(
        noise_std=noise_std
        )

    num_params = 5
    device = "cuda:0"
    n_epochs = 30000
    continue_train = False
    train_model = True
    test_model = True
    load_test_data = False
    plot_test = True

    checkpoint_dir = os.path.join(base_dir, "checkpoint")

    if not os.path.isdir(checkpoint_dir):
        os.makedirs(checkpoint_dir)

    if load_test_data:
        with open(os.path.join(base_dir, f"test_data_nsin{n_sinusoids}.pkl"),"rb") as f:
            test_data_array = pickle.load(f)
    else:
        test_data_array = test_data[0]
        with open(os.path.join(base_dir, f"test_data_nsin{n_sinusoids}.pkl"),"wb+") as f:
            pickle.dump(test_data_array, f)

    if plot_test:
        print(np.shape(test_data_array[0]))
        for i in range(len(test_data_array[0])):
            if i > 10:
                break
            fig, ax = plt.subplots()
            img = ax.imshow(test_data_array[0][i][0].T**2 + test_data_array[0][i][1].T**2, aspect="auto", origin="lower",interpolation="none")
            ax.set_xlabel("time")
            ax.set_ylabel("frequency")

            fig.savefig(os.path.join(base_dir,f"nsin{n_sinusoids}_test{i}.png"))


    #train_dat = torch.utils.data.DataLoader([data[int(i)] for i in range(n_train)], batch_size = 128, shuffle=True)
    #val_dat = torch.utils.data.DataLoader([data[int(i)] for i in range(n_val)], batch_size = 128, shuffle=True)
    #test_dat = torch.utils.data.DataLoader([test_data[int(i)] for i in range(n_test)], batch_size = 128, shuffle=False)

    n_channels, n_fft, n_samp = train_data[0][0][0].shape

    model, optimiser, scheduler = setup_model(
        num_params, 
        (n_channels, n_fft, n_samp), 
        device=device, 
        checkpoint_dir=checkpoint_dir, 
        continue_train = continue_train
        )

    callbacks = []
    callbacks.append(vitamin.callbacks.SaveModelCallback(
        model, 
        optimiser, 
        checkpoint_dir, 
        save_interval = 50))

    callbacks.append(vitamin.callbacks.LossPlotCallback(
        base_dir, 
        checkpoint_dir, 
        save_interval = 50))

    callbacks.append(vitamin.callbacks.AnnealCallback(
            model, 
            1500, 
            2000,
            1))

    test_callback = TestCallback(
        model, 
        test_data_array,
        base_dir,
        500
    )
    callbacks.append(test_callback)
    
    if train_model:
        vitamin.train.train_loop(
            model, 
            device, 
            optimiser, 
            n_epochs, 
            train_data, 
            validation_iterator = val_data, 
            callbacks = callbacks,
            verbose = False,
            continue_train=continue_train,
            checkpoint_dir=checkpoint_dir,
            )

    if test_model:

        test_callback.on_epoch_end("end")


if __name__ == "__main__":

    run_train_vitamin()
   