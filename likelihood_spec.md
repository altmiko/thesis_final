# Likelihood and Prior Specification for the VAE Latent-Attack Framework

This note explains how the thesis should discuss the Gaussian, Laplace, and
GMM distributional choices used in the VAE and latent attack pipeline. The main
point is terminological: these distributions do not all play the same role.

Use the umbrella phrase **distributional assumptions** when discussing them
together. Then separate them into:

| Distributional object | Correct name in thesis | Where it appears | Main role |
|---|---|---|---|
| `p(z) = N(0, I)` | VAE latent prior | VAE training loss | Regularizes latent codes toward a simple reference distribution |
| `q_phi(z | x) = N(mu_phi(x), diag(sigma_phi(x)^2))` | Approximate posterior / encoder distribution | VAE encoder | Encodes each traffic row into a stochastic latent representation |
| Gaussian continuous NLL | Decoder likelihood | VAE reconstruction loss | L2-style reconstruction for continuous features |
| Laplace continuous NLL | Alternative decoder likelihood | VAE reconstruction loss ablation | L1-style robust reconstruction for heavy-tailed continuous features |
| Class-conditional GMM | Empirical latent restart prior | Latent PGD/CW attack restarts | Samples plausible source-class latent starting points |

## Why This Distinction Matters

It is tempting to say that "the thesis uses Gaussian, Laplace, and GMM priors."
That would be imprecise.

Only the standard normal distribution `p(z) = N(0, I)` is the formal VAE latent
prior used in the evidence lower bound (ELBO). Gaussian and Laplace, in this
codebase, describe the **continuous-feature decoder likelihood**. The GMM is a
separate empirical distribution fitted after VAE training over encoded
validation latents. It is used to initialize latent attacks, not to train the
VAE itself.

The clean thesis phrasing is:

> The VAE uses a standard Gaussian latent prior with a diagonal Gaussian
> approximate posterior. Two continuous decoder likelihoods are evaluated:
> Gaussian and Laplace. In the attack phase, an additional class-conditional GMM
> is fitted over validation latent means and used only as a restart prior for
> latent-space adversarial search.

## 1. VAE Latent Prior

For each traffic sample `x`, the encoder outputs a mean vector and log-variance
vector:

```text
encoder(x) -> mu_phi(x), logvar_phi(x)
```

This defines the approximate posterior:

```text
q_phi(z | x) = N(mu_phi(x), diag(sigma_phi(x)^2))
```

where:

```text
sigma_phi(x)^2 = exp(logvar_phi(x))
```

The latent variable is sampled using the reparameterization trick:

```text
epsilon ~ N(0, I)
z = mu_phi(x) + sigma_phi(x) * epsilon
```

The prior is:

```text
p(z) = N(0, I)
```

This is the formal VAE prior. It is simple, isotropic, and class-independent
inside each per-category VAE. The model is trained with a KL term that pushes
the approximate posterior toward this prior:

```text
KL(q_phi(z | x) || p(z))
```

In this project, each of the 8 traffic categories has its own beta-VAE, so the
standard normal prior is applied separately within each category-specific
latent space.

### Thesis Interpretation

The standard Gaussian latent prior prevents the encoder from assigning arbitrary
latent coordinates to training samples. It makes the latent space smoother and
encourages nearby latent points to decode into plausible traffic rows. This is
important for latent attacks because PGD and CW optimize directly over `z`; the
attack depends on the decoder mapping nearby latent points into meaningful
network-flow feature vectors.

## 2. Beta-VAE KL Weight and Free Bits

The implementation uses a beta-VAE objective:

```text
loss = reconstruction_loss + beta * KL(q_phi(z | x) || p(z)) + constraint terms
```

The Gaussian anti-collapse VAE run uses:

```text
beta_target = 0.5
free_bits_lambda = 0.1
latent_dim = 16
```

`beta` controls the strength of the KL regularization. A larger beta pushes the
posterior closer to the prior, which can improve smoothness but may reduce
information in the latent code. A smaller beta allows the encoder to preserve
more sample-specific information, which can improve reconstruction.

`free_bits_lambda` gives each latent dimension a small KL allowance before that
dimension is heavily penalized. This reduces posterior collapse, where many
latent dimensions carry almost no information. In this thesis, avoiding collapse
is important because latent-space attacks need an expressive enough latent space
to search for valid adversarial examples.

### Thesis Interpretation

The beta and free-bits settings are not separate priors. They are regularization
controls on how strongly the approximate posterior is matched to the Gaussian
latent prior. The thesis should describe them as anti-collapse and latent
capacity controls.

## 3. Gaussian Decoder Likelihood

The Gaussian VAE setting uses a Gaussian negative log-likelihood for continuous
features. In simplified form:

```text
p_theta(x_cont | z) = N(mu_theta(z), diag(sigma_theta(z)^2))
```

The loss per continuous feature is:

```text
0.5 * [log(sigma^2) + (x - mu)^2 / sigma^2 + log(2*pi)]
```

This behaves like a variance-weighted squared reconstruction error. Large
errors receive quadratic penalty.

### Why Use It

Gaussian reconstruction is the standard VAE choice. It works well when residual
errors are roughly symmetric and not dominated by extreme outliers. It also
gives a clear probabilistic interpretation of the decoder output: the decoder
predicts both a location `mu` and uncertainty `sigma^2`.

### Limitation for Traffic Data

CICIoT-style flow features can be heavy-tailed. Features such as rate,
inter-arrival time, packet count, and byte totals can contain extreme values.
For such features, squared error can be dominated by a relatively small number
of large-magnitude rows.

### Thesis Wording

> In the Gaussian VAE variant, continuous features are reconstructed with a
> heteroscedastic Gaussian negative log-likelihood. This corresponds to a
> variance-weighted L2 reconstruction objective and serves as the standard
> probabilistic baseline for the mixed-input beta-VAE.

## 4. Laplace Decoder Likelihood

The Laplace VAE setting changes the continuous-feature likelihood, not the
latent prior:

```text
p_theta(x_cont | z) = Laplace(mu_theta(z), b_theta(z))
```

The loss per continuous feature is:

```text
log(2b) + |x - mu| / b
```

This behaves like a scale-weighted absolute reconstruction error. Large errors
receive linear rather than quadratic penalty.

### Why Use It

The Laplace likelihood is more robust to heavy-tailed continuous features. When
traffic features contain extreme values, the Laplace objective reduces the
chance that training is dominated by a few large residuals.

In the implementation, the decoder still predicts a continuous location. The
second continuous output head is interpreted as log-scale under Laplace rather
than log-variance under Gaussian.

### Thesis Wording

> The Laplace VAE variant keeps the same encoder, decoder architecture, latent
> dimension, and standard Gaussian latent prior, but replaces the continuous
> Gaussian reconstruction likelihood with a heteroscedastic Laplace likelihood.
> This changes the continuous reconstruction term from an L2-style penalty to an
> L1-style robust penalty, which is better suited to heavy-tailed network-flow
> statistics.

## 5. Mixed-Input Likelihood Structure

The VAE is not a single Gaussian over all 39 input features. The feature schema
is mixed:

- Continuous features use Gaussian or Laplace NLL.
- Independent binary features use binary cross-entropy.
- Protocol type uses categorical cross-entropy.
- Derived protocol binaries such as TCP, UDP, ICMP, and IGMP are reconstructed
  from the decoded protocol choice rather than generated independently.

This matters because CICIoT traffic rows contain several feature types. Treating
all columns as ordinary continuous variables would allow invalid outputs such as
fractional binary flags or inconsistent protocol indicators.

### Thesis Wording

> Because the traffic schema contains continuous, binary, and categorical
> protocol fields, the decoder likelihood is factorized by feature type. The
> continuous block uses either Gaussian or Laplace NLL, independent binary
> fields use binary cross-entropy, and protocol type uses categorical
> cross-entropy. Protocol-derived binary indicators are tied to the decoded
> protocol decision to prevent mutually inconsistent protocol encodings.

## 6. Class-Conditional GMM Restart Prior

The GMM is fitted after VAE training. It is not part of the VAE ELBO.

For each source class `c`, validation samples from that class are passed through
the corresponding class-specific VAE encoder:

```text
X_val[class = c] -> encoder -> z_mu
```

Then a Bayesian Gaussian mixture model is fitted to the encoded posterior means:

```text
p_gmm(z | y = c) = sum_k w_k N(mu_k, Sigma_k)
```

In the current pipeline:

```text
GMM split = validation
GMM components = 5
maximum fit samples = 50000
```

The GMM is cached under:

```text
outputs/latent_gmm_priors/
```

During attack generation, a GMM restart samples a latent starting point from the
source-class empirical latent density. The sampled point is then clipped into
the allowed class-specific latent radius around the original encoded point.

### Why Use GMM Restarts

Single-start latent attacks begin from:

```text
z_orig = encoder(x_original)
```

This is conservative, but it can fail if the local gradient path is poor.
Restart-aware attacks use several initial points:

```text
encoded, jitter, gmm, jitter, gmm
```

The three restart types have different meanings:

| Restart | Meaning | Purpose |
|---|---|---|
| `encoded` | Start from `z_orig` | Most local and conservative attack path |
| `jitter` | Start from `z_orig + Uniform(-epsilon, epsilon)` | Local random exploration |
| `gmm` | Start from class-conditional latent GMM sample | Manifold-guided exploration |

The GMM restart is more informed than random noise because it proposes latent
points from regions where real validation samples of the same source class tend
to appear. It helps the optimizer explore plausible source-class manifold
regions while still respecting the latent budget.

### What the GMM Is Not

The GMM is not:

- a replacement for the VAE prior;
- a target-class classifier;
- fitted on adversarial examples;
- fitted on test labels for attack optimization;
- used to relax the attack success definition.

It is only an empirical latent density used to seed restarts.

### Thesis Wording

> After VAE training, a class-conditional Bayesian GMM is fitted over validation
> posterior means in each source-class latent space. This GMM is used only as a
> restart prior for latent PGD and latent CW. It provides attack initializations
> from statistically plausible source-class latent regions, while the attack
> remains constrained by the same class-specific latent radius and validity
> checks.

## 7. How to Explain the Full Objective

A compact thesis-level objective is:

```text
L =
  L_continuous
  + L_binary
  + lambda_protocol * L_protocol
  + beta * KL(q_phi(z | x) || N(0, I))
  + lambda_constraint * L_constraint
  + lambda_physics * L_physics
```

where:

```text
L_continuous = Gaussian NLL or Laplace NLL
L_binary = binary cross-entropy
L_protocol = categorical cross-entropy
KL = latent prior regularization
L_constraint = differentiable raw-space validity penalty
L_physics = optional packet-size consistency penalty
```

This is the safest way to present the method. It makes clear that Gaussian and
Laplace are alternatives for the continuous reconstruction term, while the
standard Gaussian prior remains the latent-space regularizer.

## 8. Suggested Placement in the Thesis Report

### Methodology: VAE Architecture

Add the latent prior immediately after the architecture description:

> Each category-specific VAE uses a 16-dimensional latent variable. The encoder
> parameterizes a diagonal Gaussian approximate posterior
> `q_phi(z | x) = N(mu_phi(x), diag(sigma_phi(x)^2))`, and the latent prior is
> the standard normal `p(z) = N(0, I)`. The KL term in the beta-VAE objective
> regularizes the encoded traffic representation toward this prior, encouraging
> smooth latent interpolation and making latent-space adversarial search
> meaningful.

### Methodology: VAE Loss

Add the likelihood explanation in the loss section:

> The continuous reconstruction term is evaluated under two likelihood choices.
> The Gaussian variant uses a heteroscedastic Gaussian NLL, equivalent to a
> variance-weighted L2 reconstruction penalty. The Laplace variant uses a
> heteroscedastic Laplace NLL, equivalent to a scale-weighted L1 penalty. The
> Laplace version is included because network-flow statistics are often
> heavy-tailed, and an L1-style likelihood is less dominated by extreme
> continuous-feature residuals.

### Methodology: Restart-Aware Latent Attacks

Add the GMM explanation near the restart table:

> The GMM restart prior is fitted after VAE training by encoding validation
> samples from each source class and fitting a 5-component Bayesian Gaussian
> mixture to their latent posterior means. During latent PGD/CW, GMM starts are
> sampled from this source-class density and clipped to the permitted latent
> radius around the original code. Thus, the GMM does not change the VAE prior
> or attack success criterion; it only provides more plausible restart seeds.

### Results

Use the Gaussian vs Laplace attack results as an empirical comparison of decoder
likelihood choices:

> The Gaussian and Laplace VAE reruns test whether the continuous-feature
> likelihood affects validity-preserving attack performance. Both variants keep
> the same latent prior and restart protocol; only the continuous reconstruction
> likelihood changes. Therefore, differences in ASR or joint validity should be
> interpreted as effects of the decoder likelihood and reconstruction geometry,
> not as changes to the formal latent prior.

## 9. Short Version for the Thesis

If space is limited, use this paragraph:

> The VAE uses a standard Gaussian latent prior `p(z)=N(0,I)` and a diagonal
> Gaussian encoder posterior `q_phi(z|x)`. The KL term regularizes encoded
> traffic samples toward this prior, while beta scheduling and free bits reduce
> posterior collapse. Separately, the decoder likelihood for continuous features
> is evaluated in Gaussian and Laplace variants: Gaussian NLL gives an L2-style
> reconstruction objective, whereas Laplace NLL gives an L1-style objective more
> robust to heavy-tailed flow statistics. Finally, the GMM used in the attack
> pipeline is not the VAE prior; it is an empirical class-conditional latent
> density fitted on validation posterior means and used only to seed
> restart-aware latent PGD/CW from plausible source-class manifold regions.

## 10. Implementation Anchors

The relevant implementation points are:

| Concept | Source |
|---|---|
| Default beta, free bits, latent dimension, likelihood setting | `src/vae/config.py` |
| Gaussian and Laplace continuous NLL | `src/vae/losses.py` |
| Beta scheduler and ELBO call | `src/vae/train.py` |
| Per-class VAE training and diagnostics | `src/vae/train_all.py` |
| Latent GMM fitting and cache | `src/attack/latent_gmm.py` |
| Restart schedule and class-specific epsilon | `src/attack/latent_restarts.py` |
| Gaussian VAE config | `configs/vae_gaussian_anticollapse_beta05_freebits01_20260529_173512.json` |
| Laplace VAE config | `configs/vae_laplace_rerun_20260529.json` |

