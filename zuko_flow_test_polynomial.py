#from signal import SIGRTMAX
#from tarfile import LENGTH_LINK
import numpy as np
import torch
import torch.nn as nn
from torch.utils.data import DataLoader, Dataset
import os
import pickle
import scipy.stats as st
import matplotlib.pyplot as plt
import zuko
from scipy.spatial.distance import jensenshannon

output_dir = "./outputs_2par/"
appended = ""

if not os.path.isdir(output_dir):
    os.makedirs(output_dir)


# a few conditions to choose which samplers to run
generate_test = False         # generate testing data
load_test = True              # load testing data 
train_network = True          # train the flow
test_network = True           # test the flow
run_mcmc_sampler = False      # run mcmc sampler on data
make_test_plots = True        # make plots for testing


num_params = 2                             # number of inference parameters
length = 50                                # number of data samples
n_transforms = 2
xdat = np.linspace(0,1,length)             # "time" or x position samples
sigma = 0.1                                # vairance of gaussian
# prior distributions
# use Gaussian priors here as we can get an analytic posterior
prior_mean = np.zeros(num_params) + 0.1    
prior_cov = np.identity(num_params)*0.01
n_epochs = 10

######################################
## Set up the network and optimiser
######################################

prior = torch.distributions.MultivariateNormal(torch.zeros(num_params), torch.eye(num_params))


model = zuko.flows.CNF(num_params, length, transforms=n_transforms).to(torch.float32)
optimiser = torch.optim.AdamW(model.parameters(), 1e-5, weight_decay=1e-5)


#################
## Data generation functions
##################

def data_model(x: np.array, ms: np.array) -> np.array:
    """Simple polynomial model

    Args:
        x (_type_): times
        ms (_type_): polynomial coefficients in order

    Returns:
        _type_: polynomial
    """
    y = 0
    for i, m in enumerate(ms):
        y += ms[i]*(x**i)
    return y


def get_dataset(xdat, num_data, prior_mean, prior_cov, length = 100, sigma=0.1, num_params = 2):
    """generates some polynomial signals with a gaussian prior on the parameters

    Args:
        xdat (_type_): _description_
        num_data (_type_): _description_
        prior_mean (_type_): _description_
        prior_cov (_type_): _description_
        length (int, optional): _description_. Defaults to 100.
        sigma (float, optional): _description_. Defaults to 0.1.
        num_params (int, optional): _description_. Defaults to 2.

    Returns:
        _type_: _description_
    """
    y = []
    x = []
    # Gaussian prior                                                                                                                     
    an_mvn = st.multivariate_normal(prior_mean, prior_cov)
    for i in range(num_data):
        # draw coefficients from prior
        ms = an_mvn.rvs(1)                                                                                         
        # make sure data has 3 dimensions (number_examples, number_datapoints, number_channels)     
        # compute signal and add noise                                                                  
        y.append(np.expand_dims(data_model(xdat,ms) + np.random.normal(0,sigma,size=len(xdat)),-1))
        x.append(ms)
    return np.array(x),np.swapaxes(np.array(y), 2, 1)

####################################
## Generate testing data
####################################

if generate_test:   
    #generate 100 examples of test_data
    test_dat = get_dataset(xdat, 100, prior_mean, prior_cov, num_params=num_params, length=length, sigma=sigma)
    with open(os.path.join(output_dir, "test_data.pkl"), "wb") as f:
        pickle.dump(test_dat, f)
    test_dataloader = DataLoader([[test_dat[1][i], test_dat[0][i]] for i in range(len(test_dat[0]))], batch_size = 1)

elif load_test:
    # load test data if already generated
    with open(os.path.join(output_dir, "test_data.pkl"), "rb") as f:
        test_dat = pickle.load(f)
    test_dataloader = DataLoader([[test_dat[1][i], test_dat[0][i]] for i in range(len(test_dat[0]))], batch_size = 1)

##########################################
## Training functions
############################################

def train_loop(model, dataset, optimiser):
    """train one epoch

    Args:
        model (_type_): zuko model
        dataset (_type_): _description_
        optimiser (_type_): _description_

    Returns:
        _type_: _description_
    """
    train_loss = 0
    for index, (data, label) in enumerate(dataset):
        # flatten data for input to flow
        input_data = data.to(torch.float32).flatten(start_dim=1)
        # pass input to flow and compute the mean log probability over batch
        loss = -model(
            input_data
            ).log_prob(
                label.to(torch.float32)
                ).mean()
        train_loss += loss.item()

        # zero gradients and update model
        optimiser.zero_grad()
        loss.backward()
        optimiser.step()

    avloss = train_loss / len(dataset.dataset)
    print(f"Train Loss: {avloss}")
    return avloss

def test_loop(model, dataset):
    """Test the model

    Args:
        model (_type_): zuko model
        dataset (_type_): _description_

    Returns:
        _type_: _description_
    """
    test_loss = 0
    with torch.no_grad():
        for data, label in dataset:
            loss = -model(
                data.flatten(start_dim=1).to(torch.float32)
                ).log_prob(
                    label.to(torch.float32)
                    ).mean()

            test_loss += loss.item()
    avloss =  test_loss / len(dataset.dataset)
    print(f"Val Loss: {avloss}")
    return avloss

def sample_data(model, dataset, nsamples):
    """Generate samples from the data

    Args:
        model (_type_): zuko model
        dataset (_type_): dataset with (data, label)
        nsamples (_type_): _description_

    Returns:
        _type_: _description_
    """
    model.eval()
    with torch.no_grad():
        outputs = []
        labels = []
        for data, label in dataset:
            with torch.no_grad():
                # pass input to flow and generate samples
                output = model(
                    data.flatten(start_dim=1).to(torch.float32)
                    ).sample((nsamples,))

            outputs.append(output.numpy())
            labels.append(label.numpy()[0])

    return outputs, labels

def compute_JS_div(estimated_samples, sampler_samples, Nsamp=5000, nstep=100, ntest=100):
    """compute JS estimate
        take the mean JS divergence of {nstep} random draws of {Nsamp} samples from each of the sets of samples
        samples should be of shape (nsampes, nparameters)
    """
    temp_JS = np.zeros((ntest, np.shape(estimated_samples)[1]))
    if np.shape(estimated_samples)[1] != np.shape(estimated_samples)[1]:
        raise Exception("Samples should have the same number of parameters")
    SMALL_CONST = 1e-162
    def my_kde_bandwidth(obj, fac=1.0):
        """We use Scott's Rule, multiplied by a constant factor."""
        return np.power(obj.n, -1./(obj.d+4)) * fac
        
    for n in range(ntest):
        # Get some random indicies of samples of size Nsamp
        idx1 = np.random.randint(0,estimated_samples.shape[0],Nsamp)
        idx2 = np.random.randint(0,sampler_samples.shape[0],Nsamp)
        for pr in range(np.shape(estimated_samples)[1]):
            # get the samples from each set
            kdsampp = estimated_samples[idx1, pr:pr+1][~np.isnan(estimated_samples[idx1, pr:pr+1])].flatten()
            kdsampq = sampler_samples[idx2, pr:pr+1][~np.isnan(sampler_samples[idx2, pr:pr+1])].flatten()
            # create points to evaluate the kde on
            eval_points = np.linspace(np.min([np.min(kdsampp), np.min(kdsampq)]), np.max([np.max(kdsampp), np.max(kdsampq)]), nstep)
            # create kde
            kde_p = st.gaussian_kde(kdsampp)(eval_points)
            kde_q = st.gaussian_kde(kdsampq)(eval_points)
            # compute JS divergence
            current_JS = np.power(jensenshannon(kde_p, kde_q),2)
            temp_JS[n][pr] = current_JS

    # cmompute the mean JS divergence over each above test
    temp_JS = np.mean(temp_JS, axis = 0)

    return temp_JS


########################################
#### Train the network
########################################

if train_network == True:

    # generate the training dataset and the validation dataset
    # generate the training dataset and the validation dataset
    tr_dataset = get_dataset(xdat, 100000, prior_mean, prior_cov, num_params=num_params, length=length, sigma=sigma)
    val_dataset = get_dataset(xdat, 1000, prior_mean, prior_cov, num_params=num_params, length=length, sigma=sigma)
    # rescale training parameters to be between 0 and 1 (from the range [-1,1])                                                                                       
    rescale_train_par = tr_dataset[0]#(tr_dataset[0] - minbound)/(maxbound-minbound)
    rescale_val_par = val_dataset[0]#(val_dataset[0] - minbound)/(maxbound-minbound)
    # put data into dataloader
    train_dataloader = torch.utils.data.DataLoader([[tr_dataset[1][i], rescale_train_par[i]] for i in range(len(tr_dataset[0]))], batch_size = 512, shuffle=True)
    val_dataloader = torch.utils.data.DataLoader([[val_dataset[1][i], rescale_val_par[i]] for i in range(len(val_dataset[0]))], batch_size = 128)   

    losses = []
    val_losses = []
    for i in range(n_epochs):
        print("Epoch: ", i)
        t_tr_loss = train_loop(model, train_dataloader, optimiser)
        losses.append(t_tr_loss)
        t_v_loss = test_loop(model, val_dataloader)
        val_losses.append(t_v_loss)

        fig, ax = plt.subplots()
        ax.plot(losses)
        ax.plot(val_losses)
        #ax.set_yscale("log")
        fig.savefig(os.path.join(output_dir, "losses.png"))

    # save outputs
    with open(os.path.join(output_dir, "loss.pkl"), "wb") as f:
        pickle.dump([losses,val_losses], f)

    

    # save the weights of the model
    torch.save(
        {"model_state_dict": model.state_dict(),
        "optimiser_state_dict": optimiser.state_dict()
        },
        os.path.join(output_dir,"model.pt"))

########################
# Test the network
#########################


if test_network:
    # load the weights of pretrained model
    if train_network == False:
        checkpoint = torch.load(os.path.join(output_dir,"model.pt"))
        model.load_state_dict(checkpoint["model_state_dict"])


    # generate some samples (Run each sample through individually with shape (1, datapoints, channels))
    samples, labels = sample_data(model, test_dataloader, 10000)

    with open(os.path.join(output_dir, "samples.pkl"), "wb") as f:
        pickle.dump(samples, f)

    # analytic posterior
    an_posts = []
    for td in range(len(test_dat[1])):
        # analytic posterior                                                                                                                                          
        phi = [np.ones((length, 1))]
        for i in range(num_params - 1):
            phi.append(np.expand_dims(np.power(xdat,i+1), -1))
        phi = np.concatenate(phi, axis = 1)
        prior_cov_inv = np.linalg.inv(prior_cov)
        posterior_cov = sigma**2 * np.linalg.inv(sigma**2 * prior_cov_inv + phi.T @ phi)
        posterior_mean = posterior_cov @ (prior_cov_inv @ prior_mean + phi.T @ np.array(test_dat[1][td]).flatten() / sigma**2)
        an_posts.append([posterior_mean, posterior_cov])

    with open(os.path.join(output_dir, f"samples{appended}.pkl"), "wb") as f:
        pickle.dump(samples, f)

    with open(os.path.join(output_dir, f"analytic_meancov{appended}.pkl"), "wb") as f:
        pickle.dump(an_posts, f)

##########################
## Run the MCMC sampler on test data
#########################

if run_mcmc_sampler:
    import pymc as pm
    # run mcmc on the same test data using pymc                                                                                                                       
    mcmc_samples = []
    # initialise the x data                                                                                                                                           
    #loop over all of the test data                                                                                                                                   
    for td in range(len(test_dat[1])):
        # setup pymc model                                                                                                                                            
        with pm.Model() as gauss_model:
            # uniform priors on each of the parameters as in the training data                                                                                        
            priors = [pm.Normal(f"p{i}",prior_mean[i],np.sqrt(np.diag(prior_cov)[i])) for i in range(num_params)]
            # Gaussian likelihood with fixed sigma as in training                                                                                                     
            lik = pm.Normal("lik", mu=data_model(xdat,priors), sigma=sigma, observed = np.squeeze(test_dat[1][td]))

            # setup sampler and generate samples                                                                                                                      
            mcmc_samples.append(pm.sample(2000, chains=5))

    with open(os.path.join(output_dir,"mcmc_samples.pkl"),"wb") as f:
        pickle.dump(mcmc_samples, f)

###################
## Make some test plots
##################

if make_test_plots:

    with open(os.path.join(output_dir,"mcmc_samples.pkl"),"rb") as f:
        mcmc_samples = pickle.load(f)

    mc_samps = []
    for ind in range(len(mcmc_samples)):
        mc_samps.append(np.array([np.concatenate(np.array(getattr(mcmc_samples[ind].posterior,f"p{pnum}"))) for pnum in range(num_params)]))

    with open(os.path.join(output_dir, f"samples{appended}.pkl"), "rb") as f:
        estimated_samples = pickle.load(f)

    with open(os.path.join(output_dir, f"analytic_meancov{appended}.pkl"), "rb") as f:
        meancov = pickle.load(f)

    an_samples = []
    for m,c in meancov:
        an_mvn = st.multivariate_normal(m.reshape(-1), c)
        an_samples.append(an_mvn.rvs(10000))

    kls_vit_mc = []
    kls_vit_an = []
    kls_an_mc = []

    for mc_samp, vit_samp, an_samp in zip(mc_samps, estimated_samples, an_samples):
        vit_samp = vit_samp.squeeze()
        kls_vit_mc.append(compute_JS_div(np.array(vit_samp), np.array(mc_samp).T, Nsamp=1000, ntest = 100, nstep = 100))
        kls_vit_an.append(compute_JS_div(np.array(vit_samp), np.array(an_samp), Nsamp=1000, ntest = 100, nstep = 100))
        kls_an_mc.append(compute_JS_div(np.array(an_samp), np.array(mc_samp).T, Nsamp=1000, ntest = 100, nstep = 100))

    with open(os.path.join(output_dir,f"kl_vit_mc_divs{appended}.pkl"),"wb") as f:
        pickle.dump(kls_vit_mc,f)

    with open(os.path.join(output_dir,f"kl_vit_an_divs{appended}.pkl"),"wb") as f:
        pickle.dump(kls_vit_an,f)

    with open(os.path.join(output_dir,f"kl_an_mc_divs{appended}.pkl"),"wb") as f:
        pickle.dump(kls_an_mc,f)
