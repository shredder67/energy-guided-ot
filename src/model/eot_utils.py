class BasicMLP(nn.Module):
    def __init__(self, hidden_dim):
        super().__init__()
        self.mlp = nn.Sequential(
            nn.Linear(2, hidden_dim),
            nn.LeakyReLU(0.2),
            nn.Linear(hidden_dim, hidden_dim),
            nn.LeakyReLU(0.2),
            nn.Linear(hidden_dim, 1),
        )

    def forward(self, x):
        return self.mlp(x)


class HREG_Scheduler:
    """Scheduler for HREG value through langevin steps - in order to make langevin more stable,
    introduce exponential decay from large value (making cost gradient small)
    to desired value"""
    hreg_0 = config.HREG
    hreg_m = 10_000  # arbitrary large value to start decay from
    tau = 20  # parameter of exponential decay (larger => steeper)

    @classmethod
    def get_hreg(cls, t):
        return cls.hreg_0 + cls.hreg_m*math.exp(-t/cls.tau)


class ReplayBuffer:
    """Joint replay buffer for x and y efficient sampling during langevin.

    Usage:
    ```
    buffer = ReplayBuffer(buffer_size=10_000)
    x, y = buffer.get_n_samples(1024)
    ... # langvein updates => x_new, y_new
    buffer.update(x_new, y_new)
    ```
    """

    def __init__(self, buffer_size=config.BUFFER_SIZE, shape=2):
        self._size = buffer_size
        self._buffer = torch.normal(0, 3., size=(buffer_size, 2, shape))
        self._buffer.requires_grad_ = False
        self._selected_indices = None

    def get_n_samples(self, n):
        self._selected_indices = np.random.choice(
            list(range(self._size)), size=n, replace=False)
        return self._buffer[self._selected_indices, 0].to(DEVICE), self._buffer[self._selected_indices, 1].to(DEVICE)

    def update(self, new_x, new_y):
        if self._selected_indices is None:
            raise ValueError("Error: you haven't sampled yet!")
        with torch.no_grad():
            self._buffer[self._selected_indices, 0] = new_x.cpu()
            self._buffer[self._selected_indices, 1] = new_y.cpu()
        self._selected_indices = None

    def visualize(self, return_img=True):
        plt.figure()

        x, y = self._buffer[:, 0, :], self._buffer[:, 1, :]
        plt.scatter(x[:, 0], x[:, 1], s=7, marker='o',
                    edgecolors='black', linewidths=0.5)
        plt.scatter(y[:, 0], y[:, 1], c='red', s=7, marker='o',
                    edgecolors='black', linewidths=0.5)

        if not return_img:
            plt.show()
        else:
            img_buf = io.BytesIO()
            plt.savefig(img_buf, format='png')
            plt.close()
            return PIL.Image.open(img_buf)
        

def function_grad(f_out, arg, *args):
    """function gradient w.r.t input argumnets (support for batch-dim)"""
    grad = autograd.grad(
        outputs=f_out,
        inputs=arg,
        grad_outputs=torch.ones_like(f_out),
        create_graph=True,
        retain_graph=True
    )[0]
    return grad


def c(x, y):
    """Wasserstein-2 distance"""
    return 0.5*torch.pow(torch.norm(x - y), 2)


### Langevin related function with configured coef. for each component ###
@torch.enable_grad
def get_potential_score(f, inp):
    """potential score w.r.t. inp"""
    inp.requires_grad_(True)
    POTENTIAL_COEF = 1.
    f_out = f(inp)
    score = function_grad(f_out, inp)
    return POTENTIAL_COEF * score


@torch.enable_grad
def get_cost_score(inp1, inp2, t):
    """cost score w.r.t inp1"""
    inp1.requires_grad_(True)
    if config.HREG_DECAY:
        COST_COEF = config.SAMPLING_NOISE**2/HREG_Scheduler.get_hreg(t)
    else:
        COST_COEF = config.SAMPLING_NOISE**2/config.HREG
    cost = c(inp1, inp2)
    score = function_grad(cost, inp1)
    return COST_COEF * score
##########################################################################


def sample_from_joint_pi_theta(
    u_model,
    v_model,
    n_samples=config.N_SAMPLES,
    num_steps=config.N_STEPS,
    sampling_noise=config.SAMPLING_NOISE,
    buffer=None
):
    u_model.eval()
    v_model.eval()

    if buffer is None:
        x, y = torch.normal(0, 1., size=(2*n_samples, 2),
                            device=DEVICE).split(n_samples, dim=0)
    else:
        x, y = buffer.get_n_samples(n_samples)

    for t in range(num_steps):
        energy_grad_x = get_potential_score(
            u_model, x) - get_cost_score(x, y, t)
        energy_grad_y = get_potential_score(
            v_model, y) - get_cost_score(y, x, t)

        z = torch.randn_like(x, device=DEVICE)

        # perform langevin step
        x = x + 0.5 * energy_grad_x + sampling_noise * z
        y = y + 0.5 * energy_grad_y + sampling_noise * z

    if buffer is not None:
        buffer.update(x, y)

    u_model.train()
    v_model.train()
    return x, y


def sample_conditional(
    model,  # potential function
    x_start,  # point, on which sampling is conditioned
    n_samples,
    n_steps=config.N_STEPS,
    sampling_noise=config.SAMPLING_NOISE,
    hreg=config.HREG,
):
    """inference algorithm for conditional sampling from ebm given point from one domain to other"""
    model.eval()
    x = torch.broadcast_to(x_start, (n_samples, 2))
    y = torch.normal(0, 1., size=x.shape, device=DEVICE)
    noise = sampling_noise
    for t in range(n_steps):
        energy_grad = get_potential_score(model, y) - get_cost_score(y, x, t)
        z = torch.randn_like(x, device=DEVICE)
        y = y + 0.5 * energy_grad + noise * z

    return y
