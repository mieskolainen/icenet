# Conditional AI-Reweighting Models

m.mieskolainen@imperial.ac.uk, 2025

In all derivation steps here we assume that class priors are balanced (`equal_frac = true`),
i.e. class prior fractions are made equal by reweight.


## Type I: Amortized Conditional Reweighting

[this strategy is set in the steering cards for `configs/icezee`]

In this strategy, the **Stage-1** classifier weights are used as 
LR-weights for the **Stage-2** training. This makes Stage-2 model to learn
a conditional re-weighting model ~ $p_1(x|z) / p_0(x|z)$, in an amortized way.
Key is that this allows one to re-use (recycle) the learned Stage-2 model,
e.g. by training with a control sample, and then applied to another sample.

**Theory**: Stage-2 training with Stage-1 density ratios $r(z) = p_1(z) / p_0(z)$ applied
to the $p_0$ sample with re-weighted events, results in *an amortized conditional ratio estimator*. This is given by
```
# Stage-1: z-marginal
r(z) = p1(z) / p0(z) 

# Stage-2: (x,z)-joint with r(z) reweights for p0
p1(x,z) / (p0(x,z) x r(z)) = p1(x,z) / (p0(x|z) p1(z)) = p1(x|z) / p0(x|z),
```
which was obtained by utilizing the definition $p(x|z) = p(x,z) / p(z)$. This shows that the Stage-2 model is now a *conditional density ratio estimator*.

Now if the final weight applied in deployment is a multiplication of the weights from the two stages as follows
```
Stage-1 applied   Stage-2 applied
[p1(z) / p0(z)] x [p1(x|z) / p0(x|z)] = p1(x,z) / p0(x,z),
```
we obtain *a joint ratio re-weight factor*.


### Steering Card Setup

```
# Stage-1 model training

> train_runmode.reweight_param.reweight                   = true
> train_runmode.reweight_param.reweight_mode              = 'write'
> train_runmode.reweight_param.differential               = true
> train_runmode.reweight_param.equal_frac                 = true
> train_runmode.reweight_param.diff_param.type            = 'AIRW'
> train_runmode.reweight_param.var                        = ['z0', 'z1', ..., 'z{m-1}']
> train_runmode.reweight_param.diff_param.AIRW_param.mode = 'LR'

# Stage-2 model training

> Train a classifier with input variables:
   concat([x,z]) = ['x0', 'x1', ..., 'x{n-1}', 'z0', 'z1', ..., 'z{m-1}']

# Stage-1 model applied in evaluation

> eval_runmode.reweight_param.reweight                    = true
> eval_runmode.reweight_param.reweight_mode               = 'load'
> eval_runmode.reweight_param.differential                = true
> eval_runmode.reweight_param.equal_frac                  = true
> eval_runmode.reweight_param.diff_param.type             = 'AIRW'
> eval_runmode.reweight_param.diff_param.AIRW_param.mode  = 'LR'

# Stage-2 model applied in evaluation

> plot_param.OBS_reweight.transform_mode                  = 'LR'
```


## Type II: Non-Amortized Conditional Reweighting

This *non-amortized* strategy is somewhat different than the Type I described above.

**Theory**: Stage-1 and Stage-2 are simply trained to estimate the two ratios:
```
# Stage-1: z-marginal
p1(z) / p0(z)    

# Stage-2: (x,z)-joint
p1(x,z) / p0(x,z)
```
without any re-weighting based on Stage-1 applied to the Stage-2 training.

Then, a deployment using the estimators as follows
``` 
Stage-1 (inverse-LR)   Stage-2 applied
[p0(z) / p1(z)]      x [p1(x,z) / p0(x,z)] = p1(x|z) / p0(x|z),
```
gives us a *conditional ratio re-weight factor*.

Note that the marginals $p_1(z)$ and $p_0(z)$ are *not* aligned to be the same via
a generic conditional re-weighting, only $p_1(x|z)$ and $p_0(x|z)$.

However, if $p_0(z) = p_1(z)$ i.e. their ratio is one for all $z$, then conditional and joint ratios will be the same.

### Steering Card Setup

```
# Stage-1 model training:

> train_runmode.reweight_param.reweight                   = true
> train_runmode.reweight_param.reweight_mode              = 'write-skip' [NOTE THIS, do not apply the weights!]
> train_runmode.reweight_param.differential               = true
> train_runmode.reweight_param.equal_frac                 = true
> train_runmode.reweight_param.diff_param.type            = 'AIRW'
> train_runmode.reweight_param.var                        = ['z0', 'z1', ..., 'z{m-1}']
> train_runmode.reweight_param.diff_param.AIRW_param.mode = (irrelevant because of 'write-skip')

# Stage-2 model training:

> Train a classifier with input variables:
   concat([x,z]) = ['x0', 'x1', ..., 'x{n-1}', 'z0', 'z1', ..., 'z{m-1}']

# Stage-1 model applied in evaluation:

> eval_runmode.reweight_param.reweight                    = true
> eval_runmode.reweight_param.reweight_mode               = 'load'
> eval_runmode.reweight_param.differential                = true
> eval_runmode.reweight_param.equal_frac                  = true
> eval_runmode.reweight_param.diff_param.type             = 'AIRW'
> eval_runmode.reweight_param.diff_param.AIRW_param.mode  = 'inverse-LR' [NOTE THIS, apply inverse!]

# Stage-2 model applied in evaluation:

> plot_param.OBS_reweight.transform_mode                  = 'LR'
```
