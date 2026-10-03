import math

import torch
import torch.nn as nn
from torch.distributions import Normal, kl_divergence

from hierprior.torch_stub import Data, MultilevelModel, train, evaluate


class LogisticRegression(MultilevelModel):
    """LR baseline.
    ŷ = p(y|x,θ,β)
    """

    def __init__(self, data: Data, sd=1):
        super().__init__(data)
        self.weights = nn.Parameter(torch.zeros(data.n_codes))
        self.sd = sd

    def forward(self, X: torch.Tensor):
        """X: (batch, n_codes) -> (batch,) logits."""
        Z      = X @ self.weights # (batch, )
        logits = Z + self.bias
        return logits
        
    def penalty(self):
        return 0.5 * (self.weights ** 2).sum() / self.sd**2

def indicators(X, A):
    """Binary 'any code under node v' matrix from leaf counts X and A = leaf -> nodes."""
    Z = (X @ A).tocsr()
    Z.data[:] = 1.0
    return Z

class MultiLevel_ICD10(MultilevelModel):
    """
    logit = b + Σ_c w_c
    w_c = Σ_{a ∈ path(c)} θ_a,  θ_a ~ N(0, var_level(a))

    """
    def __init__(self, data: Data, sd=0.5):
        """Initialise the multilevel logistic regression model
        
        :param sd: the standard deviation of each coefficient
        """
        super().__init__(data)

        # One weight for every single node in the ICD10 graph
        self.weights = nn.Parameter(torch.zeros(data.n_nodes))
        self.level = data.depth # (n_nodes,) level index

        # use log σ instead of σ for numerical reasons
        # i.e. it can never become negative when optimising
        self.log_std = torch.full((self.level.max() + 1, ), fill_value=math.log(sd))

    def forward(self, X: torch.Tensor):
        """X: (batch, n_codes) sparse CSR -> (batch,) logits."""
        w = (self.data.A @ self.weights)        # (n_codes, ) path sums
        return X @ w + self.bias # (batch,)

    def penalty(self):
        # normal density fn:
        #   p(θ) = 1/√(2πσ²) · exp( −θ² / (2σ²) )
        
        # bayesian prior, inversely proportional to the variance of each level.
        # (high variance == weak prior)
        variance = torch.exp(2 * self.log_std)[self.level]  # variance per node
        return 0.5 * (self.weights[1:]**2 / variance[1:]).sum() # node 0 is the hierarchy root; not used

class VariationalMultiLevel(MultilevelModel):
    def __init__(self, data, init_sd: float = 1e-3):
        """
        :param init_sd: the initial standard deviation of each parameter
        """
        super().__init__(data)
        self.weights = nn.Parameter(torch.zeros(data.n_nodes))
        """Per-node learnt coefficients"""

        self.log_s = nn.Parameter(
            torch.full((data.n_nodes,), fill_value=math.log(init_sd))
        )
        """Per-node log standard deviation."""
        
        self.level = data.depth
        self.log_std = nn.Parameter(torch.zeros(int(data.depth.max() + 1)))
        """Per-level log s.d."""

    def _theta(self):
        """During training weights are drawn from the normal distribution.
        N(μ, σ²) plus some gaussian noise dependent on it's variance (s)
            θ = μ + σ · ε
        """
        if self.training:
            return self.weights + torch.exp(self.log_s) * torch.randn_like(self.weights)
        return self.weights

    def forward(self, X):
        theta = self._theta()
        w = self.data.A @ theta
        return self.bias + (X @ w)
    
    def penalty(self):
        """The Kullback-Leibler divergence between the posterior and the prior.
        
        KL(posterior ‖ prior) =
            E_posterior[ - log prior(θ) ]         # L2 averaged over the posterior
            - entropy(posterior)                  # discourage unwarranted confidence
        """
        posterior = Normal(self.weights[1:], torch.exp(self.log_s[1:]))
        prior     = Normal(0., torch.exp(self.log_std)[self.level][1:])
        return kl_divergence(posterior, prior).sum()


if __name__ == "__main__":
    import argparse
    ap = argparse.ArgumentParser()
    ap.add_argument("--task", default="mortality", choices=["readmission", "mortality"])
    ap.add_argument("--mode", default="any", choices=["any", "primary"])
    ap.add_argument("--tree", default="prefix", choices=["prefix", "official_block", "official_category", "official_prefix"])
    a = ap.parse_args()
    d = Data(a.task, a.mode, a.tree)

    model = train(VariationalMultiLevel(d), d)
    print(evaluate(model, d))
    print("BEST FOR VARIATIONAL MULTILEVEL MODEL")

    multilevel = train(MultiLevel_ICD10(d), d)
    print(evaluate(multilevel, d))
    print("BEST FOR MULTILEVEL MODEL")
    
    model = train(LogisticRegression(d), d)
    print(evaluate(model, d))
    print("BEST FOR LR MODEL")
