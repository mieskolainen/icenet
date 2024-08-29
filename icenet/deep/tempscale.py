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

# ------------------------------------------
from icenet import print
# ------------------------------------------

class LogitsWithTemperature(nn.Module):
    """
    "Temperate calibration" wrapper class.
    
    Use with original raw logits and class labels as an input.
    """
    def __init__(self, mode='softmax', device='cpu'):
        super().__init__()
        
        self.temperature = nn.Parameter(1.0 * torch.ones(1, device=device))
        self.device      = device
        self.mode        = mode

    def temperature_scale(self, logits):
        """
        Temperature scaling on logits
        """
        return logits / self.temperature

    def calibrate(self, logits: torch.Tensor, labels: torch.Tensor,
                        weights: torch.Tensor=None, lr: float=0.01, max_iter: int=50):
        """
        Tune the temperature of the model with NLL loss (using the validation set)
        
        Args:
            logits:  model output logits per event (single or softmax type)
            labels:  class label per event (torch.float32)
            weights: weights per event
        """
        
        try:
            
            if weights is None:
                weights = torch.ones_like(logits).to(self.device)
            
            if   self.mode == 'softmax':
                nll_criterion = nn.CrossEntropyLoss(reduction='none').to(self.device)
            elif self.mode == 'binary':
                nll_criterion = nn.BCEWithLogitsLoss(reduction='none').to(self.device)
            else:
                raise Exception(__name__ + f'.set_temperature: Unknown mode, should be "softmax" or "binary')
            
            weights = weights / torch.sum(weights.flatten())
            
            ece_criterion_before = _ECELoss(weights=weights, mode=self.mode).to(self.device)
            
            # Calculate NLL and ECE before temperature scaling
            before_temperature_nll = torch.sum(weights * nll_criterion(logits, labels)).item()
            before_temperature_ece = ece_criterion_before(logits, labels).item()
            print(f'Before temperature scale: NLL: {before_temperature_nll:0.4f}, ECE: {ece_criterion_before.ECE.item():0.4f}, ECE2: {ece_criterion_before.ECE2.item():0.4f}')
            ece_criterion_before.print()
            
            # Optimize the temperature parameter
            optimizer = optim.LBFGS([self.temperature], lr = lr, max_iter = max_iter)

            ece_criterion_after = _ECELoss(weights=weights, mode=self.mode).to(self.device)

            def eval():
                optimizer.zero_grad()
                
                #loss = torch.sum(weights * nll_criterion(self.temperature_scale(logits), labels))
                loss = torch.sum(weights * ece_criterion_after(self.temperature_scale(logits), labels))
                
                loss.backward()
                return loss
            
            optimizer.step(eval)

            # Calculate NLL and ECE after temperature scaling
            after_temperature_nll = torch.sum(weights * nll_criterion(self.temperature_scale(logits), labels)).item()
            after_temperature_ece = ece_criterion_after(self.temperature_scale(logits), labels).item()
            
            print('')
            print(f'Optimal temperature: {self.temperature.item():0.4f}', 'green')
            print(f'After temperature scale: NLL: {after_temperature_nll:0.4f}, ECE: {ece_criterion_after.ECE.item():0.4f}, ECE2: {ece_criterion_after.ECE2.item():0.4f}')
            
            ece_criterion_after.print()
            
            return ece_criterion_before, ece_criterion_after

        except Exception as e:
            print(f'Error: {e}')
            return -1, -1

class ModelWithTemperature(nn.Module):
    """
    "Temperate calibration" wrapper class.
    
    Output of the original network needs to be in logits,
    not softmax or log softmax.

    Expects model(input) to return logits.
    """
    def __init__(self, model, mode='softmax', device='cpu'):
        
        super().__init__()
        
        self.model       = model
        self.temperature = nn.Parameter(1.0 * torch.ones(1, device=device))
        self.device      = device
        self.mode        = mode

    def forward(self, input: torch.Tensor):
        logits = self.model(input)
        return self.temperature_scale(logits)

    def temperature_scale(self, logits: torch.Tensor):
        """
        Temperature scaling on logits
        """
        return logits / self.temperature
    
    def calibrate(self, valid_loader, lr: float=0.01, max_iter: int=50):
        """
        Tune the temperature of the model with NLL loss (using the validation set)
        
        Args:
            valid_loader: validation set loader (DataLoader)
        """
        
        try:
            
            self.model.eval() #!
            
            # Collect all the logits, labels and weights for the validation set
            logits_list  = []
            labels_list  = []
            weights_list = []

            with torch.no_grad():
                
                for i, batch_ in enumerate(valid_loader):
                    
                    # -----------------------------------------
                    # Torch models
                    if type(batch_) is dict:
                        x,y,w = batch_['x'], batch_['y'], batch_['w']

                        if 'u' in batch_: # Dual models
                            x = {'x': batch_['x'], 'u': batch_['u']}

                    # Torch-geometric
                    else:
                        x,y,w = batch_, batch_.y, batch_.w
                    # -----------------------------------------
                    
                    # logits
                    logits = self.model(x.to(self.device))
                    logits_list.append(logits)
                    
                    # y
                    labels_list.append(y)

                    # w
                    weights_list.append(w)

                logits  = torch.cat(logits_list).to(self.device)
                labels  = torch.cat(labels_list).to(self.device)
                weights = torch.cat(weights_list).to(self.device)

            logits  = logits.squeeze()
            labels  = labels.squeeze()
            weights = weights.squeeze()
            
            if self.mode == 'binary':
                labels = labels.float()
            
            weights = weights / torch.sum(weights.flatten())
            
            # Losses
            if   self.mode == 'softmax':
                nll_criterion = nn.CrossEntropyLoss(reduction='none').to(self.device)
            elif self.mode == 'binary':
                nll_criterion = nn.BCEWithLogitsLoss(reduction='none').to(self.device)
            else:
                raise Exception(__name__ + f'.set_temperature: Unknown mode, should be "softmax" or "binary')
            
            ece_criterion_before = _ECELoss(weights=weights, mode=self.mode).to(self.device)
            
            # Calculate NLL and ECE before temperature scaling
            before_temperature_nll = torch.sum(weights * nll_criterion(logits, labels)).item()
            before_temperature_ece = ece_criterion_before(logits, labels).item()
            print(f'Before temperature scale: NLL: {before_temperature_nll:0.4f}, ECE: {ece_criterion_before.ECE.item():0.4f}, ECE2: {ece_criterion_before.ECE2.item():0.4f}')
            ece_criterion_before.print()
            
            # Optimize the temperature parameter
            optimizer = optim.LBFGS([self.temperature], lr = lr, max_iter = max_iter)

            ece_criterion_after = _ECELoss(weights=weights, mode=self.mode).to(self.device)

            def eval():
                optimizer.zero_grad()
                
                #loss = torch.sum(weights * nll_criterion(self.temperature_scale(logits), labels))
                loss = torch.sum(weights * ece_criterion_after(self.temperature_scale(logits), labels))
                
                loss.backward()
                return loss
            
            optimizer.step(eval)

            # Calculate NLL and ECE after temperature scaling
            after_temperature_nll = torch.sum(weights * nll_criterion(self.temperature_scale(logits), labels)).item()
            after_temperature_ece = ece_criterion_after(self.temperature_scale(logits), labels).item()
            
            print('')
            print(f'Optimal temperature: {self.temperature.item():0.4f}', 'green')
            print(f'After temperature scale: NLL: {after_temperature_nll:0.4f}, ECE: {ece_criterion_after.ECE.item():0.4f}, ECE2: {ece_criterion_after.ECE2.item():0.4f}')
            
            ece_criterion_after.print()
            
            return ece_criterion_before, ece_criterion_after

        except Exception as e:
            print(f'Error: {e}')
            return -1, -1

class _ECELoss(nn.Module):
    """
    Expected calibration error of the model.

    References:
        Naeini, Mahdi Pakdaman, Gregory F. Cooper, and Milos Hauskrecht.
        "Obtaining Well Calibrated Probabilities Using Bayesian Binning." AAAI, 2015.
    """

    def __init__(self, mode='softmax', n_bins=20, weights=None):
        """
        Args:
            mode:   'softmax' or 'binary'
            n_bins: number of confidence interval bins
        """
        super().__init__()
        
        bin_boundaries  = torch.linspace(0, 1, n_bins + 1)
        self.bin_lowers = bin_boundaries[:-1]
        self.bin_uppers = bin_boundaries[1:]
        self.mode       = mode
        self.weights    = weights
        
        self.ECE_binned  = torch.zeros(n_bins)
        self.ECE2_binned = torch.zeros(n_bins)
        self.ECE         = 0
        self.ECE2        = 0
    
    
    def print(self):
        """
        For terminal print out
        """
        k = 0
        for bin_lower, bin_upper in zip(self.bin_lowers, self.bin_uppers):
            print(f'[{bin_lower:00.4f}, {bin_upper:00.4f}]: ECE = {self.ECE_binned[k]:0.4f} | ECE2 = {self.ECE2_binned[k]:0.4f}')
            k += 1
        
        print(f'ECE = {self.ECE.item():0.4f} | ECE2 = {self.ECE2.item():0.4f}')
    
    
    def forward(self, logits: torch.Tensor, labels: torch.Tensor, return_mode='ECE2'):
        """
        For optimization
        """
        
        if self.weights is None:
            self.weights = torch.ones(logits.shape[0]).to(logits.device)
        
        if   self.mode == 'softmax':
            target_class = 0
            confidences  = F.softmax(logits, dim=-1)[:,target_class] # Pick the target class
            labels_hat   = torch.round(confidences).to(torch.int32)  # Class label prediction (yhat)
            accuracies   = labels_hat.eq(labels.to(torch.int32))     # Compute accuracy I(y == y_hat)
        
        elif self.mode == 'binary':
            confidences = F.sigmoid(logits)                          # Apply sigmoid
            labels_hat  = torch.round(confidences).to(torch.int32)   # Class label prediction (yhat)
            accuracies  = labels_hat.eq(labels.to(torch.int32))      # Compute accuracy I(y == y_hat)
        
        else:
            raise Exception(__name__ + f'.forward: Unknown mode, use either "softmax" or "binary"')
        
        # ----------------------------------------
        
        ECE  = torch.zeros(1, device=logits.device)
        ECE2 = torch.zeros(1, device=logits.device)
        
        weight_sum = torch.sum(self.weights)
        
        k = 0
        for bin_lower, bin_upper in zip(self.bin_lowers, self.bin_uppers):
            
            in_bin = confidences.gt(bin_lower.item()) * confidences.le(bin_upper.item())
            w      = self.weights[in_bin]
            w_sum  = torch.sum(w)
            
            # Per bin accuracy and confidence
            if torch.sum(in_bin) > 0:
                acc  = torch.sum(w * accuracies[in_bin])  / w_sum
                freq = torch.sum(w * labels[in_bin]) / w_sum
                conf = torch.sum(w * confidences[in_bin]) / w_sum
                
                # Calculated |confidence - accuracy| per bin    
                ECE_bin  = (w_sum / weight_sum) * (conf - acc)
                ECE += torch.abs(ECE_bin)
                
                # Calculated |confidence - frequency| per bin
                ECE2_bin = (w_sum / weight_sum) * (conf - freq)
                ECE2 += torch.abs(ECE2_bin)
                
                self.ECE_binned[k]  = ECE_bin
                self.ECE2_binned[k] = ECE2_bin
                
            else:
                ECE_bin  = float('-999')
                ECE2_bin = float('-999')

                self.ECE_binned[k]  = float('-999')
                self.ECE2_binned[k] = float('-999')
            
            k += 1
        
        self.ECE  = ECE
        self.ECE2 = ECE2
        
        if return_mode == 'ECE':
            return ECE
        else:
            return ECE2
