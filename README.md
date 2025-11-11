# Chapter 278: Variational LOB (Limit Order Book) Trading

## 1. Introduction

The Limit Order Book (LOB) is the central data structure powering modern electronic exchanges. At any instant, it records every outstanding buy (bid) and sell (ask) order across all price levels, giving a high-dimensional snapshot of market supply and demand. For algorithmic traders, the LOB is the richest source of microstructure information -- it encodes liquidity, order imbalance, spread dynamics, and the intentions of competing agents.

However, raw LOB data is unwieldy. A typical snapshot with 20 levels on each side produces an 80-dimensional vector (price and quantity for each level on both sides). Sequences of such snapshots are noisy, highly correlated, and expensive to store and process. Standard supervised learning on raw LOB features often overfits to idiosyncratic patterns or fails to capture the latent structure that drives price movements.

**Variational Autoencoders (VAEs)** offer an elegant solution. By learning a probabilistic mapping from high-dimensional LOB snapshots to a compact latent space, VAEs simultaneously achieve dimensionality reduction, generative modeling, and uncertainty quantification. The latent representation captures the essential "market state" in just a handful of variables, while the decoder can reconstruct or generate realistic LOB configurations on demand.

This chapter presents a deep treatment of Variational LOB Trading: applying VAEs specifically to limit order book data for compression, generation, anomaly detection, and regime analysis. We develop the mathematical foundations, implement a complete system in Rust with Bybit API integration, and demonstrate practical trading applications.

### Why Variational LOB?

1. **Compact representation**: Compress 80+ dimensional LOB snapshots into 8-16 latent variables that capture essential market state.
2. **Anomaly detection**: Reconstruction error naturally flags unusual LOB configurations -- spoofing, flash crashes, or liquidity vacuums.
3. **LOB interpolation**: Generate plausible intermediate LOB states between observed snapshots by interpolating in latent space.
4. **Regime-conditioned generation**: Condition the VAE on market regimes (trending, mean-reverting, volatile) to generate regime-specific synthetic LOB data.
5. **Uncertainty quantification**: The probabilistic framework provides calibrated confidence intervals around LOB predictions.
6. **Data augmentation**: Generate unlimited synthetic LOB data for training downstream models.

## 2. Mathematical Foundations

### 2.1 LOB State Representation

An LOB snapshot at time $t$ is represented as a feature vector. Given $K$ price levels on each side, the raw snapshot is:

$$\mathbf{L}_t = \{(p_k^{bid}, q_k^{bid}, p_k^{ask}, q_k^{ask})\}_{k=1}^{K}$$

We apply standard normalizations to make the representation invariant to absolute price levels:

**Price normalization** (relative to mid-price):
$$\tilde{p}_k^{bid} = \frac{p_k^{bid} - m_t}{m_t}, \quad \tilde{p}_k^{ask} = \frac{p_k^{ask} - m_t}{m_t}$$

where $m_t = \frac{p_1^{bid} + p_1^{ask}}{2}$ is the mid-price.

**Quantity normalization** (log-transform):
$$\tilde{q}_k = \log(1 + q_k)$$

**Derived features** augment the raw levels:
- Bid-ask spread: $s_t = p_1^{ask} - p_1^{bid}$
- Order imbalance at level $k$: $\text{OI}_k = \frac{q_k^{bid} - q_k^{ask}}{q_k^{bid} + q_k^{ask}}$
- Cumulative depth: $D_k^{bid} = \sum_{j=1}^{k} q_j^{bid}$
- Weighted mid-price: $w_t = \frac{p_1^{bid} \cdot q_1^{ask} + p_1^{ask} \cdot q_1^{bid}}{q_1^{bid} + q_1^{ask}}$

The complete feature vector $\mathbf{x}_t \in \mathbb{R}^{n}$ concatenates all normalized prices, quantities, and derived features.

### 2.2 Variational Autoencoder Architecture

A VAE consists of an encoder network $q_\phi(\mathbf{z}|\mathbf{x})$ and a decoder network $p_\theta(\mathbf{x}|\mathbf{z})$, connected through a stochastic latent variable $\mathbf{z} \in \mathbb{R}^d$.

**Encoder**: Maps the LOB feature vector to parameters of a Gaussian posterior:
$$\boldsymbol{\mu} = f_\mu(\mathbf{x}; \phi), \quad \log \boldsymbol{\sigma}^2 = f_\sigma(\mathbf{x}; \phi)$$

**Reparameterization trick**: Enables backpropagation through the stochastic sampling step:
$$\mathbf{z} = \boldsymbol{\mu} + \boldsymbol{\sigma} \odot \boldsymbol{\epsilon}, \quad \boldsymbol{\epsilon} \sim \mathcal{N}(\mathbf{0}, \mathbf{I})$$

This separates the stochasticity ($\boldsymbol{\epsilon}$) from the learned parameters ($\boldsymbol{\mu}, \boldsymbol{\sigma}$), making the gradient $\nabla_\phi \mathbf{z}$ well-defined.

**Decoder**: Reconstructs the LOB features from the latent code:
$$\hat{\mathbf{x}} = g(\mathbf{z}; \theta)$$

### 2.3 ELBO Loss Function

The VAE is trained by maximizing the Evidence Lower Bound (ELBO):

$$\mathcal{L}(\theta, \phi; \mathbf{x}) = \underbrace{\mathbb{E}_{q_\phi(\mathbf{z}|\mathbf{x})}[\log p_\theta(\mathbf{x}|\mathbf{z})]}_{\text{Reconstruction term}} - \underbrace{D_{KL}(q_\phi(\mathbf{z}|\mathbf{x}) \| p(\mathbf{z}))}_{\text{KL divergence term}}$$

**Reconstruction term**: For Gaussian outputs, this becomes the negative mean squared error:
$$\mathbb{E}_{q_\phi}[\log p_\theta(\mathbf{x}|\mathbf{z})] \approx -\frac{1}{2}\|\mathbf{x} - \hat{\mathbf{x}}\|^2$$

**KL divergence term**: With a standard normal prior $p(\mathbf{z}) = \mathcal{N}(\mathbf{0}, \mathbf{I})$, the KL divergence has a closed-form solution:
$$D_{KL} = -\frac{1}{2}\sum_{j=1}^{d}(1 + \log\sigma_j^2 - \mu_j^2 - \sigma_j^2)$$

**Beta-VAE weighting**: We introduce a hyperparameter $\beta$ to control the trade-off:
$$\mathcal{L}_\beta = \text{Recon} - \beta \cdot D_{KL}$$

- $\beta < 1$: Better reconstruction, less structured latent space
- $\beta = 1$: Standard VAE
- $\beta > 1$: More disentangled latent factors, potentially worse reconstruction

For LOB data, we find $\beta \in [0.5, 2.0]$ works well, with lower values preferred when reconstruction fidelity is critical (e.g., for anomaly detection) and higher values when latent space structure matters (e.g., for regime discovery).

### 2.4 Anomaly Detection via Reconstruction Error

The reconstruction error provides a natural anomaly score:

$$A(\mathbf{x}) = \|\mathbf{x} - \hat{\mathbf{x}}\|^2 = \|\mathbf{x} - g(\text{encode}(\mathbf{x}); \theta)\|^2$$

Under normal conditions, the VAE reconstructs LOB snapshots well. When the LOB enters an unusual configuration (spoofing, flash crash, extreme imbalance), the reconstruction error spikes because the model has not learned to represent such states efficiently.

We threshold the anomaly score using the empirical distribution:
$$\text{Anomaly if } A(\mathbf{x}) > \mu_A + k \cdot \sigma_A$$

where $\mu_A$ and $\sigma_A$ are the mean and standard deviation of anomaly scores on the training set, and $k$ is typically 2-3.

### 2.5 Latent Space Interpolation

Given two LOB snapshots $\mathbf{x}_a$ and $\mathbf{x}_b$, we can generate smooth intermediate states by interpolating in latent space:

$$\mathbf{z}_\alpha = (1 - \alpha)\mathbf{z}_a + \alpha \mathbf{z}_b, \quad \alpha \in [0, 1]$$

$$\hat{\mathbf{x}}_\alpha = g(\mathbf{z}_\alpha; \theta)$$

Because the latent space is regularized to be approximately Gaussian, linear interpolation produces plausible intermediate LOB states. This is useful for:
- Filling gaps in LOB data at higher temporal resolution
- Studying how the market transitions between different states
- Generating smooth trajectories for simulation

### 2.6 Regime-Conditioned VAE

We extend the standard VAE to a Conditional VAE (CVAE) by providing a regime label $c$:

$$q_\phi(\mathbf{z}|\mathbf{x}, c), \quad p_\theta(\mathbf{x}|\mathbf{z}, c)$$

The regime $c$ can encode:
- **Volatility regime**: Low, medium, high (based on realized volatility)
- **Trend regime**: Uptrend, downtrend, sideways (based on price momentum)
- **Liquidity regime**: Deep, normal, thin (based on total book depth)

By conditioning on regimes, the CVAE generates LOB snapshots consistent with specific market conditions, enabling targeted scenario analysis and stress testing.

## 3. Applications

### 3.1 LOB State Compression for Real-Time Trading

In high-frequency trading, processing full LOB snapshots at microsecond granularity is computationally expensive. The VAE encoder compresses each snapshot into a fixed-size latent vector (e.g., 10 dimensions), which can be fed to downstream trading models. This achieves:
- 8-10x dimensionality reduction
- Denoising (the latent representation filters out noise)
- Fixed-size representation regardless of the number of book levels

### 3.2 Anomaly Detection and Risk Management

The reconstruction-based anomaly detector can be deployed in real time to:
- Flag potential spoofing (artificially large orders that distort the book)
- Detect flash crashes before they fully develop (unusual book asymmetry)
- Identify liquidity vacuums (sudden thinning of the book)
- Trigger risk management actions (reduce position size, widen quotes)

### 3.3 Synthetic LOB Generation for Backtesting

By sampling from the latent space, we generate unlimited synthetic LOB data:
- **Random sampling**: $\mathbf{z} \sim \mathcal{N}(\mathbf{0}, \mathbf{I})$, decode to get a random LOB state
- **Conditional sampling**: Fix regime label, sample $\mathbf{z}$, decode to get regime-consistent LOB
- **Trajectory generation**: Sample a sequence of latent codes with temporal correlation

### 3.4 Market Regime Discovery

By clustering the latent representations $\mathbf{z}_t$ (e.g., with K-means or Gaussian Mixture Models), we can discover natural market regimes without manual labeling. Transitions between clusters correspond to regime changes that may predict future price behavior.

## 4. Rust Implementation

Our Rust implementation provides a complete variational LOB trading system:

### Architecture Overview

```
┌─────────────────────────────────────────────────┐
│  Bybit WebSocket / REST API                     │
│  (Live LOB snapshots)                           │
└──────────────────┬──────────────────────────────┘
                   │
┌──────────────────▼──────────────────────────────┐
│  LOB Feature Extractor                          │
│  - Normalize prices (relative to mid)           │
│  - Log-transform quantities                     │
│  - Compute derived features (spread, imbalance) │
└──────────────────┬──────────────────────────────┘
                   │
┌──────────────────▼──────────────────────────────┐
│  VAE Encoder                                    │
│  - Input layer → Hidden layer (ReLU)            │
│  - Hidden → μ (mean)                            │
│  - Hidden → log σ² (log variance)               │
│  - Reparameterization: z = μ + σ·ε              │
└──────────────────┬──────────────────────────────┘
                   │
┌──────────────────▼──────────────────────────────┐
│  Latent Space (z ∈ ℝ^d)                        │
│  - Anomaly scoring                              │
│  - Interpolation                                │
│  - Regime clustering                            │
└──────────────────┬──────────────────────────────┘
                   │
┌──────────────────▼──────────────────────────────┐
│  VAE Decoder                                    │
│  - Latent → Hidden layer (ReLU)                 │
│  - Hidden → Reconstructed LOB features          │
└──────────────────┬──────────────────────────────┘
                   │
┌──────────────────▼──────────────────────────────┐
│  Trading Signals                                │
│  - Anomaly alerts                               │
│  - Regime change detection                      │
│  - LOB quality scoring                          │
└─────────────────────────────────────────────────┘
```

### Key Components

- **`LobSnapshot`**: Represents a raw order book snapshot with bid/ask levels
- **`LobFeatureExtractor`**: Transforms raw snapshots into normalized feature vectors
- **`VaeLob`**: The core VAE model with encoder, decoder, reparameterization, and ELBO loss
- **`AnomalyDetector`**: Computes anomaly scores from reconstruction errors
- **`BybitClient`**: Fetches live order book data from the Bybit REST API

### Usage Example

```rust
use variational_lob::{BybitClient, LobFeatureExtractor, VaeLob, AnomalyDetector};

#[tokio::main]
async fn main() -> anyhow::Result<()> {
    // Fetch live orderbook
    let client = BybitClient::new();
    let snapshot = client.fetch_orderbook("BTCUSDT", 25).await?;

    // Extract features
    let extractor = LobFeatureExtractor::new(10);
    let features = extractor.extract(&snapshot);

    // Encode with VAE
    let vae = VaeLob::new(features.len(), 32, 10);
    let (mu, log_var) = vae.encode(&features);
    let z = vae.reparameterize(&mu, &log_var);
    let reconstructed = vae.decode(&z);

    // Check for anomalies
    let detector = AnomalyDetector::new(2.5);
    let score = detector.score(&features, &reconstructed);
    println!("Anomaly score: {:.4}", score);

    Ok(())
}
```

## 5. Bybit Data Integration

The implementation connects to Bybit's V5 REST API to fetch real-time order book data:

- **Endpoint**: `GET /v5/market/orderbook`
- **Parameters**: `category=linear`, `symbol=BTCUSDT`, `limit=50`
- **Response**: Arrays of `[price, quantity]` pairs for bids and asks

The `BybitClient` handles:
- HTTP request construction and error handling
- JSON deserialization into typed Rust structs
- Rate limiting compliance
- Conversion to internal `LobSnapshot` format

### Data Pipeline

1. **Fetch**: Pull orderbook snapshot from Bybit REST API
2. **Parse**: Deserialize JSON into `LobSnapshot` struct
3. **Normalize**: Apply price/quantity normalization via `LobFeatureExtractor`
4. **Encode**: Pass through VAE encoder to get latent representation
5. **Analyze**: Compute anomaly scores, regime labels, or generate synthetic data
6. **Act**: Generate trading signals based on latent space analysis

## 6. Key Takeaways

1. **VAEs provide a principled framework** for learning compact, probabilistic representations of LOB data. The ELBO objective balances reconstruction fidelity with latent space regularity.

2. **The reparameterization trick** is essential for training -- it decouples stochastic sampling from gradient computation, enabling end-to-end backpropagation.

3. **Anomaly detection via reconstruction error** is a powerful, unsupervised approach to identifying unusual market conditions. No labeled anomaly data is needed.

4. **Latent space interpolation** enables generation of plausible intermediate LOB states, useful for data augmentation and market simulation.

5. **Beta-VAE weighting** controls the trade-off between reconstruction quality and latent space structure. The optimal $\beta$ depends on the downstream application.

6. **Regime-conditioned generation** with CVAEs allows targeted synthesis of LOB data under specific market conditions, enabling more rigorous backtesting and stress testing.

7. **Rust implementation** provides the performance characteristics needed for real-time LOB processing, with memory safety guarantees critical for production trading systems.

8. **The latent space itself is informative** -- clustering latent representations reveals natural market regimes, and tracking latent trajectories over time captures market dynamics more efficiently than raw features.
