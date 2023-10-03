import numpy as np

def data_model(x, ms):
    """sin gaussian model

    Args:
        x (_type_): _description_
        ms (_type_): _description_

    Returns:
        _type_: _description_
    """
    #define a simple straight line model
    A, t0, ph, f, width = ms
    y = A*np.sin(2*np.pi*f*x + ph) * np.exp(-(x - t0)**2/width**2)
    return y

def prior_sample():
    """sample from sin gaussian prior

    Returns:
        _type_: _description_
    """
    A = np.random.uniform(prior_range["A_min"],prior_range["A_max"])
    t0 = np.random.uniform(prior_range["t0_min"],prior_range["t0_max"])
    ph = np.random.uniform(prior_range["ph_min"],prior_range["ph_max"])
    f =  np.random.uniform(prior_range["f_min"],prior_range["f_max"])
    width =  np.random.uniform(prior_range["width_min"],prior_range["width_max"])
    return A, t0, ph, f, width

def scale_pars(ms):
    """scale the parameters to the prior range

    Args:
        ms (_type_): _description_

    Returns:
        _type_: _description_
    """
    new_ms = np.zeros(np.shape(ms))
    for parind in range(np.shape(ms)[0]):
        new_ms[parind] = (ms[parind] - prior_range[f"{pars[parind]}_min"])/(prior_range[f"{pars[parind]}_max"]- prior_range[f"{pars[parind]}_min"])
    return new_ms

def unscale_pars(ms):
    """unscale the parameters

    Args:
        ms (_type_): _description_

    Returns:
        _type_: _description_
    """
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