import torch
import numpy as np
import torch.nn as nn

from hierprior.torch_stub import Data, MultilevelModel, train, evaluate, MortalityModel


class LogisticRegression(nn.Module):
    """LR baseline.
    ŷ = p(y|x,θ,β)
    """

    def __init__(self, data: Data):
        super().__init__()
        # params = nn.Linear(data.n_codes, out_features=1, bias=True)
        initial_params = (torch.ones(data.n_codes).detach().clone()*1e-5)
        self.θ = nn.Parameter(torch.tensor(data=initial_params, requires_grad=True))
        self.β = nn.Parameter(torch.tensor(data=[1e-5],                          requires_grad=True))

    def forward(self, X: torch.Tensor):
        """X: (batch, n_codes) -> (batch,) logits."""
        Z = X @ self.θ                  # (batch, n_codes)
        logits = Z + self.β

        return logits; assert logits.shape[0] == X.shape[0]
        
    def penalty(self):
        return torch.zeros(())


if __name__ == "__main__":
    import argparse
    ap = argparse.ArgumentParser()
    ap.add_argument("--task", default="mortality", choices=["readmission", "mortality"])
    ap.add_argument("--mode", default="any", choices=["any", "primary"])
    ap.add_argument("--tree", default="prefix", choices=["prefix", "official_block", "official_category", "official_prefix"])
    a = ap.parse_args()
    d = Data(a.task, a.mode, a.tree)
    # model = train(MultiLevel_ICD10(d), d)
    model = train(LogisticRegression(d), d)

    print(evaluate(model, d))