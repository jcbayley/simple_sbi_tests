#from signal import SIGRTMAX
#from tarfile import LENGTH_LINK
import numpy as np
import torch
import torch.nn as nn
from torch.utils.data import DataLoader, Dataset
import os
import pickle
import pymc as pm
import scipy.stats as st
import matplotlib.pyplot as plt
from ffjord import Ffjord
import zuko
import vitamin

output_dir = "./outputs_2par/"
appended = ""
# a few conditions to choose which samplers to run
generate_test = False
load_test = True
train_network = True
test_network = True
run_mcmc_sampler = False
make_test_plots = True

num_params = 2
length = 50
xdat = np.linspace(0,1,length)
sigma = 0.1
prior_mean = np.zeros(num_params) + 0.1
prior_cov = np.identity(num_params)*0.01


if not os.path.isdir(output_dir):
    os.makedirs(output_dir)

def data_model(x, ms):
    #define a simple straight line model
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
    # this is to compare with the analytic case                                                                                                                       
    an_mvn = st.multivariate_normal(prior_mean, prior_cov)
    for i in range(num_data):
        ms = an_mvn.rvs(1)
        #ms = np.random.uniform(size = num_params)                                                                                                                    
        # make sure data has 3 dimensions (number_examples, number_datapoints, number_channels)                                                                       

        y.append(np.expand_dims(data_model(xdat,ms) + np.random.normal(0,sigma,size=len(xdat)),-1))
        x.append(ms)
    return np.array(x),np.swapaxes(np.array(y), 2, 1)

if generate_test:   
    #generate test_data
    test_dat = get_dataset(xdat, 100, prior_mean, prior_cov, num_params=num_params, length=length, sigma=sigma)
    with open(os.path.join(output_dir, "test_data.pkl"), "wb") as f:
        pickle.dump(test_dat, f)
    test_dataloader = DataLoader([[test_dat[1][i], test_dat[0][i]] for i in range(len(test_dat[0]))], batch_size = 1)

elif load_test:
    # load test data if already generated
    with open(os.path.join(output_dir, "test_data.pkl"), "rb") as f:
        test_dat = pickle.load(f)
    test_dataloader = DataLoader([[test_dat[1][i], test_dat[0][i]] for i in range(len(test_dat[0]))], batch_size = 1)



def train_loop(model, dataset, optimiser):
    train_loss = 0
    for index, (data, label) in enumerate(dataset):
        #print(data.size(), label.size())
        input_data = data.to(torch.float32).flatten(start_dim=1)
        #loss = -model.log_prob(label.to(torch.float32), conditional=input_data).mean()
        #z, loss = model.compute_loss(label.to(torch.float32), conditional=input_data)
        loss = -model(input_data).log_prob(label.to(torch.float32)).mean()
        train_loss += loss.item()

        optimiser.zero_grad()
        loss.backward()
        optimiser.step()
    avloss = train_loss / len(dataset.dataset)
    print(f"Train Loss: {avloss}")
    return avloss

def test_loop(model, dataset):
    test_loss = 0
    with torch.no_grad():
        for data, label in dataset:
            #loss = -model.log_prob(label.to(torch.float32), conditional=data.flatten(start_dim=1).to(torch.float32)).mean()
            #z, loss = model.compute_loss(label.to(torch.float32), conditional=data.flatten(start_dim=1).to(torch.float32))
            loss = -model(data.flatten(start_dim=1).to(torch.float32)).log_prob(label.to(torch.float32)).mean()

            test_loss += loss.item()
    avloss =  test_loss / len(dataset.dataset)
    print(f"Val Loss: {avloss}")
    return avloss

def sample_data(model, dataset, nsamples):
    model.eval()
    with torch.no_grad():
        outputs = []
        labels = []
        for data, label in dataset:
            with torch.no_grad():
                #samples = model.prior.sample((nsamples, 1)).view(nsamples, -1)

                #print(samples.size(), data.repeat((nsamples, 1)).size())
                #output = model.inverse(samples.to(torch.float32), data.to(torch.float32).flatten(start_dim=1).repeat((nsamples, 1)))

                output = model(data.flatten(start_dim=1).to(torch.float32)).sample((nsamples,))

            outputs.append(output.numpy())
            labels.append(label.numpy()[0])

    return outputs, labels


numpy_mask = np.random.randint(1, size = num_params)

prior = torch.distributions.MultivariateNormal(torch.zeros(num_params), torch.eye(num_params))

#model = RealNVP(input_dim, hidden_dims, output_dim, prior, mask)

class Args():
    def __init__(self,):
        pass

args = Args()
args.layers = 1
args.stepsize = 0.1
args.t0 = 0
args.t1 = 1
args.solver = "rk4"
args.backprop = "adjoint"
args.trace = "hutchinson"

#model = Ffjord((num_params,), cfg = args)
model = zuko.flows.CNF(num_params, length, transforms=3).to(torch.float32)
optimiser = torch.optim.AdamW(model.parameters(), 1e-5, weight_decay=1e-5)


if train_network == True:

    # generate the training dataset and the validation dataset
    # generate the training dataset and the validation dataset
    tr_dataset = get_dataset(xdat, 100000, prior_mean, prior_cov, num_params=num_params, length=length, sigma=sigma)
    val_dataset = get_dataset(xdat, 1000, prior_mean, prior_cov, num_params=num_params, length=length, sigma=sigma)
    # rescale training parameters to be between 0 and 1 (from the range [-1,1])                                                                                       
    rescale_train_par = tr_dataset[0]#(tr_dataset[0] - minbound)/(maxbound-minbound)
    rescale_val_par = val_dataset[0]#(val_dataset[0] - minbound)/(maxbound-minbound)
    train_dataloader = torch.utils.data.DataLoader([[tr_dataset[1][i], rescale_train_par[i]] for i in range(len(tr_dataset[0]))], batch_size = 512, shuffle=True)
    val_dataloader = torch.utils.data.DataLoader([[val_dataset[1][i], rescale_val_par[i]] for i in range(len(val_dataset[0]))], batch_size = 128)   

    n_epochs = 10
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


if run_mcmc_sampler:
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


if make_test_plots:

    with open(os.path.join(output_dir,"mcmc_samples.pkl"),"rb") as f:
        mcmc_samples = pickle.load(f)

    mc_samps = []
    for ind in range(len(mcmc_samples)):
        mc_samps.append(np.array([np.concatenate(np.array(getattr(mcmc_samples[ind].posterior,f"p{pnum}"))) for pnum in range(num_params)]))

    with open(os.path.join(output_dir, f"samples{appended}.pkl"), "rb") as f:
        vitamin_samples = pickle.load(f)

    with open(os.path.join(output_dir, f"analytic_meancov{appended}.pkl"), "rb") as f:
        meancov = pickle.load(f)

    an_samples = []
    for m,c in meancov:
        an_mvn = st.multivariate_normal(m.reshape(-1), c)
        an_samples.append(an_mvn.rvs(10000))

    kls_vit_mc = []
    kls_vit_an = []
    kls_an_mc = []

    for mc_samp, vit_samp, an_samp in zip(mc_samps, vitamin_samples, an_samples):
        vit_samp = vit_samp.squeeze()
        kls_vit_mc.append(vitamin.train_plots.compute_JS_div(np.array(vit_samp), np.array(mc_samp).T, Nsamp=1000, ntest = 100, nstep = 100))
        kls_vit_an.append(vitamin.train_plots.compute_JS_div(np.array(vit_samp), np.array(an_samp), Nsamp=1000, ntest = 100, nstep = 100))
        kls_an_mc.append(vitamin.train_plots.compute_JS_div(np.array(an_samp), np.array(mc_samp).T, Nsamp=1000, ntest = 100, nstep = 100))

    with open(os.path.join(output_dir,f"kl_vit_mc_divs{appended}.pkl"),"wb") as f:
        pickle.dump(kls_vit_mc,f)

    with open(os.path.join(output_dir,f"kl_vit_an_divs{appended}.pkl"),"wb") as f:
        pickle.dump(kls_vit_an,f)

    with open(os.path.join(output_dir,f"kl_an_mc_divs{appended}.pkl"),"wb") as f:
        pickle.dump(kls_an_mc,f)
