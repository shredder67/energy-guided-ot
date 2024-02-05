class Config:
    N_SAMPLES = 1024 # number of samples in batch
    N_STEPS = 175 # number of langevin steps
    STEP_SIZE = 1. # langevin step size
    SAMPLING_NOISE = 0.04 # langevin noise variance
    
    ALPHA = 0, # coef. for potential norm regularization
    #"GAMMA": 0.01, # hyperparameter prob measure integrable on R (b)
    HREG = 1., # aka eps (parametrization of entropy)
    HREG_DECAY = False, # if true, apply decay from large value to target HREG
    BUFFER_SIZE = None, # replay buffer size (if None - no buffer used)
    SCALED_REGIME = True, # apply std scale to X and Y
    N_ITERS = 250, # number of training epochs
    HIDDEN_DIM_SIZE = 256,
    
    OPT_TYPE = "Adam",
    OPT_KWARGS = {
        "lr": 2e-4,
    },

    SEED=42
    DEVICE='cpu'

    @classmethod
    def update_from_dict(cls, kwargs):
        for var, value in kwargs.items():
            if hasattr(cls, var):
                if value is None: continue
                setattr(cls, var, value)
            else:
                raise ValueError(f"Error: provided argument {var} is not defined in config!")
            
    @classmethod
    def to_dict(cls):
        raw_dict = cls.__dict__
        return {k: v for k, v in raw_dict.items() if k.isupper()}

