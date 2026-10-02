import torch
import torch.nn as nn

from hierprior.torch_stub import Data, MultilevelModel, train, evaluate


class LogisticRegression(MultilevelModel):
    """LR baseline.
    ŷ = p(y|x,θ,β)
    """

    def __init__(self, data: Data):
        super().__init__(data)
        self.weights = nn.Parameter(torch.zeros(data.n_codes), requires_grad=True)

    def forward(self, X: torch.Tensor):
        """X: (batch, n_codes) -> (batch,) logits."""
        Z      = X @ self.weights # (batch, )
        logits = Z + self.bias
        return logits
        
    def penalty(self):
        return 0.5 * (self.weights ** 2).sum()

def indicators(X, A):
    """Binary 'any code under node v' matrix from leaf counts X and A = leaf -> nodes."""
    Z = (X @ A).tocsr()
    Z.data[:] = 1.0
    return Z

class MultiLevel_ICD10(MultilevelModel):
    """
    y ~ ∑ⱼ N(θ, σ²)ⱼ 
    
    where j = 0..|max ICD10 chars|
    """
    def __init__(self, data: Data):
        super().__init__(data)

        # One weight for every single node in the ICD10 graph
        self.weights = nn.Parameter(torch.zeros(data.n_nodes), requires_grad=True)
        self.level = data.depth # (n_nodes,) level index

        # use log σ instead of σ for numerical reasons:
        # i.e. it can never become negative when optimising
        self.log_std = torch.full((self.level.max() + 1, ), 100)
        """Per-level variance"""    
        
    def forward(self, X: torch.Tensor):
        """X: (batch, n_codes) sparse CSR -> (batch,) logits."""
        w = (self.data.A @ self.weights[:, None]).squeeze(1)        # (n_codes, ) path sums
        return (X @ w[:, None]).squeeze(1) + self.bias # (batch,)

    def penalty(self):
        # normal density fn:
        #   p(θ) = 1/√(2πσ²) · exp( −θ² / (2σ²) )
        
        # bayesian prior, inversely proportional to the variance of each level.
        # (high variance == weak prior)
        variance = torch.exp(2 * self.log_std)[self.level]  # variance per node
        return 0.5 * (self.weights[1:]**2 / variance[1:]).sum() # node 0 is the hierarchy root; not used

class VariationalMultiLevel(MultilevelModel):
    def __init__(self, data):
        super().__init__(data)
        self.nu = nn.Parameter(torch.zeros(data.n_nodes))
        """Per-node learnt coefficients"""
        self.log_s = nn.Parameter(torch.full((data.n_nodes,), -3.0))
        """Per-node log s.d."""
        
        self.level = data.depth
        self.log_std = nn.Parameter(torch.zeros(int(data.depth.max() + 1)))
        """Per-level log s.d."""
    
    def forward(self, X):
        theta = self.nu + torch.exp(self.log_s) * torch.randn_like(self.nu) \
                        if self.training else self.nu
        """During training, this weight is drawn from the value we are modelling
        it as having (μ) plus some gaussian noise dependent on it's variance (s)
         
            θ = μ + s · ε 
        
        During inference, we simply use the best guess i.e.:
        
            θ = μ
        """
        w = (self.data.A @ theta[:, None]).squeeze(1)
        return self.bias + (X @ w[:, None]).squeeze(1)
    
    def penalty(self): # Kullback-Leibler
        variance = torch.exp(2 * self.log_std)[self.level][1:]
        s2       = torch.exp(2 * self.log_s)[1:]
        nu = self.nu[1:]
        return 0.5 * (((nu**2 + s2) / variance) -1 -torch.log(s2 / variance)).sum()


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
