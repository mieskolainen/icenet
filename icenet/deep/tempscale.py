# Temperature scaling for calibrated classifier model output probabilities
# 
# Based on:
#
#   https://en.wikipedia.org/wiki/Platt_scaling
#   https://arxiv.org/abs/1706.04599
#   https://github.com/gpleiss/temperature_scaling
# 
# m.mieskolainen@imperial.ac.uk, 2024

import torch
from torch import nn, optim
from torch.nn import functional as F


class ModelWithTemperature(nn.Module):
    """
    "Temperate calibration" wrapper class.
    
    Output of the original network needs to be in logits,
    not softmax or log softmax.

    Functions expects model(input) to return logits.
    """
    def __init__(self, model, device='cpu'):
        super(ModelWithTemperature, self).__init__()
        self.model       = model.to(device)
        self.temperature = nn.Parameter(1.0 * torch.ones(1, device=device))
        self.device      = device

    def forward(self, input):
        logits = self.model(input)
        return self.temperature_scale(logits)

    def temperature_scale(self, logits):
        """
        Perform temperature scaling on logits
        """
        temperature = self.temperature.unsqueeze(1).expand(logits.size(0), logits.size(1))
        return logits / temperature

    def set_temperature(self, valid_loader, lr=0.01, max_iter=50):
        """
        Tune the tempearature of the model with NLL loss (using the validation set)
        
        Args:
            valid_loader: validation set loader (DataLoader)
        """

        nll_criterion = nn.CrossEntropyLoss(reduction='none').to(self.device)
        ece_criterion = _ECELoss().to(self.device)

        # 1. Collect all the logits, labels and weights for the validation set
        logits_list  = []
        labels_list  = []
        weights_list = []

        with torch.no_grad():
            for x, label, weights in valid_loader:
                logits = self.model(x.to(self.device))

                logits_list.append(logits)
                labels_list.append(label)

                if len(weights.shape) != 1: # Per class weights [say N x K] --> contract
                    weights = torch.sum(weights, dim=-1)

                weights_list.append(weights)

            logits  = torch.cat(logits_list).to(self.device)
            labels  = torch.cat(labels_list).to(self.device)
            weights = torch.cat(weights_list).to(self.device)

        # Calculate NLL and ECE before temperature scaling
        # -----------------------------

        before_temperature_nll = torch.mean(weights * nll_criterion(logits, labels)).item()
        before_temperature_ece = ece_criterion(logits, labels).item()
        print(f'Before temperature scale: NLL: {before_temperature_nll:.3f}, ECE: {before_temperature_ece:.3f}')

        # ----------------------------------------------------------------

        # Optimize the temperature parameter w.r.t. cross entropy NLL
        optimizer = optim.LBFGS([self.temperature], lr = lr, max_iter = max_iter)

        def eval():
            optimizer.zero_grad()
            loss = torch.mean(weights * nll_criterion(self.temperature_scale(logits), labels))
            loss.backward()
            return loss

        optimizer.step(eval)

        # Calculate NLL and ECE after temperature scaling
        after_temperature_nll = torch.mean(weights * nll_criterion(self.temperature_scale(logits), labels)).item()
        after_temperature_ece = ece_criterion(self.temperature_scale(logits), labels).item()

        print(f'Optimal temperature: {self.temperature.item():.3f}')
        print(f'After temperature scale:   NLL: {after_temperature_nll:.3f}, ECE: {after_temperature_ece:.3f}')
        
        return self


class _ECELoss(nn.Module):
    """
    Expected calibration error of the model.

    References:
        Naeini, Mahdi Pakdaman, Gregory F. Cooper, and Milos Hauskrecht.
        "Obtaining Well Calibrated Probabilities Using Bayesian Binning." AAAI, 2015.
    """

    def __init__(self, n_bins=15):
        """
        Args:
            n_bins: number of confidence interval bins
        """
        super(_ECELoss, self).__init__()
        bin_boundaries = torch.linspace(0, 1, n_bins + 1)
        self.bin_lowers = bin_boundaries[:-1]
        self.bin_uppers = bin_boundaries[1:]

    def forward(self, logits, labels):
        softmaxes = F.softmax(logits, dim=1)
        confidences, predictions = torch.max(softmaxes, 1)
        accuracies = predictions.eq(labels)

        ece = torch.zeros(1, device=logits.device)
        for bin_lower, bin_upper in zip(self.bin_lowers, self.bin_uppers):

            # Calculated |confidence - accuracy| in each bin
            in_bin = confidences.gt(bin_lower.item()) * confidences.le(bin_upper.item())
            prop_in_bin = in_bin.float().mean()

            if prop_in_bin.item() > 0:
                accuracy_in_bin = accuracies[in_bin].float().mean()
                avg_confidence_in_bin = confidences[in_bin].mean()
                ece += torch.abs(avg_confidence_in_bin - accuracy_in_bin) * prop_in_bin

        return ece
