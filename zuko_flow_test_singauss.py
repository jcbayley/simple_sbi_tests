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
import zuko
import corner

output_dir = "./outputs_singauss_5par/"
appended = ""
# a few conditions to choose which samplers to run
generate_test = False
load_test = True
train_network = True
test_network = True
run_mcmc_sampler = False
make_test_plots = True

num_params = 5
length = 50
xdat = np.linspace(0,1,length)
sigma = 0.1
pars = ["A", "t0", "ph","f","width"]
prior_range = {
    "A_min": 0.1, 
    "A_max": 0.3,
    "t0_min": 0.6,
    "t0_max": 1.6,
    "ph_min": 0,
    "ph_max": 2*np.pi,
    "f_min":2,
    "f_max": 10,
    "width_min":0.05,
    "width_max": 0.7
}


if not os.path.isdir(output_dir):
    os.makedirs(output_dir)

def data_model(x, ms):
    #define a simple straight line model
    A, t0, ph, f, width = ms
    y = A*np.sin(2*np.pi*f*x + ph) * np.exp(-(x - t0)**2/width**2)
    return y

def prior_sample():
    A = np.random.uniform(prior_range["A_min"],prior_range["A_max"])
    t0 = np.random.uniform(prior_range["t0_min"],prior_range["t0_max"])
    ph = np.random.uniform(prior_range["ph_min"],prior_range["ph_max"])
    f =  np.random.uniform(prior_range["f_min"],prior_range["f_max"])
    width =  np.random.uniform(prior_range["width_min"],prior_range["width_max"])
    return A, t0, ph, f, width

def scale_pars(ms):
    new_ms = np.zeros(np.shape(ms))
    for parind in range(np.shape(ms)[0]):
        new_ms[parind] = (ms[parind] - prior_range[f"{pars[parind]}_min"])/(prior_range[f"{pars[parind]}_max"]- prior_range[f"{pars[parind]}_min"])
    return new_ms

def unscale_pars(ms):
    new_ms = np.zeros(np.shape(ms))
    for parind in range(np.shape(ms)[0]):
        new_ms[parind] = ms[parind]*(prior_range[f"{pars[parind]}_max"]- prior_range[f"{pars[parind]}_min"]) + prior_range[f"{pars[parind]}_min"]
    return new_ms



def get_dataset(xdat, num_data, length = 100, sigma=0.1, num_params = 2):
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
    #an_mvn = st.multivariate_normal(prior_mean, prior_cov)
    for i in range(num_data):
        ms = prior_sample()
        #ms = np.random.uniform(size = num_params)                                                                                                                    
        # make sure data has 3 dimensions (number_examples, number_datapoints, number_channels)                                                                       

        y.append(np.expand_dims(data_model(xdat,ms) + np.random.normal(0,sigma,size=len(xdat)),-1))
        rescale_ms = scale_pars(ms[:num_params])
        x.append(rescale_ms)
    return np.array(x),np.swapaxes(np.array(y), 2, 1)

if generate_test:   
    #generate test_data
    test_dat = get_dataset(xdat, 100, num_params=num_params, length=length, sigma=sigma)
    with open(os.path.join(output_dir, "test_data.pkl"), "wb") as f:
        pickle.dump(test_dat, f)
    test_dataloader = DataLoader([[test_dat[1][i], test_dat[0][i]] for i in range(len(test_dat[0]))], batch_size = 1)

elif load_test:
    # load test data if already generated
    with open(os.path.join(output_dir, "test_data.pkl"), "rb") as f:
        test_dat = pickle.load(f)
    test_dataloader = DataLoader([[test_dat[1][i], test_dat[0][i]] for i in range(len(test_dat[0]))], batch_size = 1)



def train_loop(model, dataset, optimiser):
    train_loss = []
    for index, (data, label) in enumerate(dataset):
        #print(data.size(), label.size())
        input_data = data.to(torch.float32)#.flatten(start_dim=1)
        input_data = back_model(input_data)
        #loss = -model.log_prob(label.to(torch.float32), conditional=input_data).mean()
        #z, loss = model.compute_loss(label.to(torch.float32), conditional=input_data)
        loss = -model(input_data).log_prob(label.to(torch.float32)).mean()
        train_loss.append(loss.item())

        optimiser.zero_grad()
        loss.backward()
        optimiser.step()
    avloss = np.mean(train_loss)
    print(f"Train Loss: {avloss}")
    return avloss

def test_loop(model, dataset):
    test_loss = []
    with torch.no_grad():
        for data, label in dataset:
            #loss = -model.log_prob(label.to(torch.float32), conditional=data.flatten(start_dim=1).to(torch.float32)).mean()
            #z, loss = model.compute_loss(label.to(torch.float32), conditional=data.flatten(start_dim=1).to(torch.float32))
            input_data = data.to(torch.float32)#.flatten(start_dim=1).to(torch.float32)
            input_data = back_model(input_data)
            loss = -model(input_data).log_prob(label.to(torch.float32)).mean()

            test_loss.append(loss.item())
    avloss =  np.mean(test_loss)
    print(f"Val Loss: {avloss}")
    return avloss

def sample_data(model, dataset, nsamples):
    model.eval()
    with torch.no_grad():
        outputs = []
        labels = []
        for data, label in dataset:
            #samples = model.prior.sample((nsamples, 1)).view(nsamples, -1)

            #print(samples.size(), data.repeat((nsamples, 1)).size())
            #output = model.inverse(samples.to(torch.float32), data.to(torch.float32).flatten(start_dim=1).repeat((nsamples, 1)))

            input_data = data.to(torch.float32)#.flatten(start_dim=1).to(torch.float32)
            input_data = back_model(input_data)
            output = model(input_data).sample((nsamples,)).squeeze()
            loc = np.where((output < 0 ) | (output > 1))[0]
            while len(loc) > 0:
                loc = np.where((output < 0 ) | (output > 1))[0]
                output_rep = model(input_data).sample((len(loc),)).squeeze()
                output[loc] = output_rep

            output = unscale_pars(output.T).T
            outputs.append(output)
            labels.append(label.numpy()[0])

    return outputs, labels


numpy_mask = np.random.randint(1, size = num_params)

prior = torch.distributions.MultivariateNormal(torch.zeros(num_params), torch.eye(num_params))

nsf_in_length = 32
#model = Ffjord((num_params,), cfg = args)
back_model = nn.Sequential(
    nn.Conv1d(1, 64, 16),
    nn.ReLU(),
    nn.Conv1d(64, 32, 16),
    nn.MaxPool1d(2),
    nn.ReLU(),
    nn.Conv1d(32, 16, 8),
    nn.ReLU(),
    nn.Flatten(),
    nn.LazyLinear(128),
    nn.ReLU(),
    nn.LazyLinear(128),
    nn.ReLU(),
    nn.LazyLinear(nsf_in_length)
)

model = zuko.flows.NSF(num_params, nsf_in_length, bins=16, transforms=4, hidden_features=(256,128,64)).to(torch.float32)
optimiser = torch.optim.AdamW(list(model.parameters()) + list(back_model.parameters()), 5e-6, weight_decay=1e-5)


if train_network == True:

    # generate the training dataset and the validation dataset
    # generate the training dataset and the validation dataset
    tr_dataset = get_dataset(xdat, 100000, num_params=num_params, length=length, sigma=sigma)
    val_dataset = get_dataset(xdat, 1000, num_params=num_params, length=length, sigma=sigma)
    # rescale training parameters to be between 0 and 1 (from the range [-1,1])                                                                                       
    rescale_train_par = tr_dataset[0]#(tr_dataset[0] - minbound)/(maxbound-minbound)
    rescale_val_par = val_dataset[0]#(val_dataset[0] - minbound)/(maxbound-minbound)
    train_dataloader = torch.utils.data.DataLoader([[tr_dataset[1][i], rescale_train_par[i]] for i in range(len(tr_dataset[0]))], batch_size = 512, shuffle=True)
    val_dataloader = torch.utils.data.DataLoader([[val_dataset[1][i], rescale_val_par[i]] for i in range(len(val_dataset[0]))], batch_size = 128)   

    n_epochs = 30
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
        "back_model_state_dict": back_model.state_dict(),
        "optimiser_state_dict": optimiser.state_dict()
        },
        os.path.join(output_dir,"model.pt"))

if test_network:
    # load the weights of pretrained model
    if train_network == False:
        checkpoint = torch.load(os.path.join(output_dir,"model.pt"))
        model.load_state_dict(checkpoint["model_state_dict"])
        back_model.load_state_dict(checkpoint["back_model_state_dict"])


    # generate some samples (Run each sample through individually with shape (1, datapoints, channels))
    samples, labels = sample_data(model, test_dataloader, 10000)

    with open(os.path.join(output_dir, "samples.pkl"), "wb") as f:
        pickle.dump(samples, f)


    with open(os.path.join(output_dir, f"samples{appended}.pkl"), "wb") as f:
        pickle.dump(samples, f)




if run_mcmc_sampler:
    # run mcmc on the same test data using pymc                                                                                                                       
    mcmc_samples = []
    # initialise the x data                                                                                                                                           
    #loop over all of the test data                                                                                                                                   
    for td in range(len(test_dat[1])):
        # setup pymc model                                                                                                                                            
        with pm.Model() as gauss_model:
            # uniform priors on each of the parameters as in the training data                                                                                        
            priors = [
                pm.Uniform(f"A",prior_range["A_min"],prior_range["A_max"]),
                pm.Uniform(f"t0",prior_range["t0_min"],prior_range["t0_max"]),
                pm.Uniform(f"ph",prior_range["ph_min"],prior_range["ph_max"]),
                pm.Uniform(f"f",prior_range["f_min"],prior_range["f_max"]),
                pm.Uniform(f"width",prior_range["width_min"],prior_range["width_max"])
                ]
            # Gaussian likelihood with fixed sigma as in training                                                                                                     
            lik = pm.Normal("lik", mu=data_model(xdat,priors), sigma=sigma, observed = np.squeeze(test_dat[1][td]))

            # setup sampler and generate samples                                                                                                                      
            mcmc_samples.append(pm.sample(2000, chains=5))

    with open(os.path.join(output_dir,"mcmc_samples.pkl"),"wb") as f:
        pickle.dump(mcmc_samples, f)


if make_test_plots:

    if not os.path.isdir(os.path.join(output_dir, "test_plots")):
        os.makedirs(os.path.join(output_dir, "test_plots"))

    for td in range(len(test_dat[1])):
        fig, ax = plt.subplots()
        ax.plot(np.squeeze(test_dat[1][td]))
        ax.set_title(test_dat[0][td])
        fig.savefig(os.path.join(output_dir, "test_plots", f"test_plot_{td}.png"))




    with open(os.path.join(output_dir,"mcmc_samples.pkl"),"rb") as f:
        mcmc_samples = pickle.load(f)

    mc_samps = []
    for ind in range(len(mcmc_samples)):
        mc_samps.append(np.array([np.concatenate(np.array(getattr(mcmc_samples[ind].posterior,pnum))) for pnum in pars]))

    with open(os.path.join(output_dir, f"samples{appended}.pkl"), "rb") as f:
        samples = pickle.load(f)

    if not os.path.isdir(os.path.join(output_dir, "post_plots")):
        os.makedirs(os.path.join(output_dir, "post_plots"))

    for td in range(len(test_dat[1])):
        print(td, np.shape(samples[td]), np.shape(mc_samps[td].T))
        fig = corner.corner(samples[td].squeeze(),truths=test_dat[0][td], color="C1", labels = pars)
        fig = corner.corner(mc_samps[td].T, fig=fig, color="C2")

        fig.savefig(os.path.join(output_dir, "post_plots", f"test_plot_{td}.png"))


    """
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
    """
