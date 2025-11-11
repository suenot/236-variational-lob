//! Variational LOB (Limit Order Book) Trading
//!
//! This crate implements a Variational Autoencoder (VAE) for limit order book
//! modeling, including LOB feature extraction, encoding/decoding, ELBO loss,
//! anomaly detection, and Bybit API integration.

use anyhow::{Context, Result};
use ndarray::Array1;
use rand::Rng;
use serde::{Deserialize, Serialize};

// ---------------------------------------------------------------------------
// LOB Data Structures
// ---------------------------------------------------------------------------

/// A single price level in the order book.
#[derive(Debug, Clone, Serialize, Deserialize)]
pub struct PriceLevel {
    pub price: f64,
    pub quantity: f64,
}

/// A snapshot of the limit order book at a single point in time.
#[derive(Debug, Clone, Serialize, Deserialize)]
pub struct LobSnapshot {
    pub bids: Vec<PriceLevel>,
    pub asks: Vec<PriceLevel>,
    pub timestamp: u64,
}

impl LobSnapshot {
    /// Create a new LOB snapshot.
    pub fn new(bids: Vec<PriceLevel>, asks: Vec<PriceLevel>, timestamp: u64) -> Self {
        Self {
            bids,
            asks,
            timestamp,
        }
    }

    /// Compute the mid-price.
    pub fn mid_price(&self) -> f64 {
        if self.bids.is_empty() || self.asks.is_empty() {
            return 0.0;
        }
        (self.bids[0].price + self.asks[0].price) / 2.0
    }

    /// Compute the bid-ask spread.
    pub fn spread(&self) -> f64 {
        if self.bids.is_empty() || self.asks.is_empty() {
            return 0.0;
        }
        self.asks[0].price - self.bids[0].price
    }

    /// Compute order imbalance at the top of book.
    pub fn top_imbalance(&self) -> f64 {
        if self.bids.is_empty() || self.asks.is_empty() {
            return 0.0;
        }
        let bid_q = self.bids[0].quantity;
        let ask_q = self.asks[0].quantity;
        if bid_q + ask_q == 0.0 {
            return 0.0;
        }
        (bid_q - ask_q) / (bid_q + ask_q)
    }
}

// ---------------------------------------------------------------------------
// LOB Feature Extractor
// ---------------------------------------------------------------------------

/// Extracts normalized feature vectors from LOB snapshots.
///
/// For each of `num_levels` levels on each side, extracts:
/// - Normalized bid price (relative to mid)
/// - Log-transformed bid quantity
/// - Normalized ask price (relative to mid)
/// - Log-transformed ask quantity
/// Plus derived features: spread, imbalance per level, cumulative depths.
pub struct LobFeatureExtractor {
    pub num_levels: usize,
}

impl LobFeatureExtractor {
    /// Create a new feature extractor with the given number of levels per side.
    pub fn new(num_levels: usize) -> Self {
        Self { num_levels }
    }

    /// Compute the feature dimension for this extractor.
    pub fn feature_dim(&self) -> usize {
        // 4 features per level (norm_bid_price, log_bid_qty, norm_ask_price, log_ask_qty)
        // + 1 spread + num_levels imbalances + 2 cumulative depths (bid + ask)
        4 * self.num_levels + 1 + self.num_levels + 2
    }

    /// Extract a normalized feature vector from a LOB snapshot.
    pub fn extract(&self, snapshot: &LobSnapshot) -> Array1<f64> {
        let mid = snapshot.mid_price();
        let n = self.num_levels;
        let dim = self.feature_dim();
        let mut features = Array1::zeros(dim);

        let mut idx = 0;

        // Per-level features
        for k in 0..n {
            // Bid price (normalized)
            let bid_price = if k < snapshot.bids.len() {
                snapshot.bids[k].price
            } else {
                0.0
            };
            features[idx] = if mid > 0.0 {
                (bid_price - mid) / mid
            } else {
                0.0
            };
            idx += 1;

            // Bid quantity (log-transformed)
            let bid_qty = if k < snapshot.bids.len() {
                snapshot.bids[k].quantity
            } else {
                0.0
            };
            features[idx] = (1.0 + bid_qty).ln();
            idx += 1;

            // Ask price (normalized)
            let ask_price = if k < snapshot.asks.len() {
                snapshot.asks[k].price
            } else {
                0.0
            };
            features[idx] = if mid > 0.0 {
                (ask_price - mid) / mid
            } else {
                0.0
            };
            idx += 1;

            // Ask quantity (log-transformed)
            let ask_qty = if k < snapshot.asks.len() {
                snapshot.asks[k].quantity
            } else {
                0.0
            };
            features[idx] = (1.0 + ask_qty).ln();
            idx += 1;
        }

        // Spread (normalized by mid)
        features[idx] = if mid > 0.0 {
            snapshot.spread() / mid
        } else {
            0.0
        };
        idx += 1;

        // Per-level order imbalance
        for k in 0..n {
            let bid_q = if k < snapshot.bids.len() {
                snapshot.bids[k].quantity
            } else {
                0.0
            };
            let ask_q = if k < snapshot.asks.len() {
                snapshot.asks[k].quantity
            } else {
                0.0
            };
            features[idx] = if bid_q + ask_q > 0.0 {
                (bid_q - ask_q) / (bid_q + ask_q)
            } else {
                0.0
            };
            idx += 1;
        }

        // Cumulative bid depth (log-transformed)
        let cum_bid: f64 = snapshot
            .bids
            .iter()
            .take(n)
            .map(|l| l.quantity)
            .sum();
        features[idx] = (1.0 + cum_bid).ln();
        idx += 1;

        // Cumulative ask depth (log-transformed)
        let cum_ask: f64 = snapshot
            .asks
            .iter()
            .take(n)
            .map(|l| l.quantity)
            .sum();
        features[idx] = (1.0 + cum_ask).ln();

        features
    }
}

// ---------------------------------------------------------------------------
// Dense Layer (simple matrix-vector multiply)
// ---------------------------------------------------------------------------

/// A simple dense (fully connected) layer: y = activation(Wx + b).
#[derive(Debug, Clone)]
pub struct DenseLayer {
    pub weights: Vec<Vec<f64>>,
    pub biases: Vec<f64>,
    pub use_relu: bool,
}

impl DenseLayer {
    /// Create a new dense layer with random Xavier initialization.
    pub fn new(input_dim: usize, output_dim: usize, use_relu: bool) -> Self {
        let mut rng = rand::thread_rng();
        let scale = (2.0 / (input_dim + output_dim) as f64).sqrt();
        let weights = (0..output_dim)
            .map(|_| {
                (0..input_dim)
                    .map(|_| rng.gen_range(-scale..scale))
                    .collect()
            })
            .collect();
        let biases = vec![0.0; output_dim];
        Self {
            weights,
            biases,
            use_relu,
        }
    }

    /// Forward pass through the layer.
    pub fn forward(&self, input: &Array1<f64>) -> Array1<f64> {
        let output_dim = self.biases.len();
        let mut output = Array1::zeros(output_dim);
        for i in 0..output_dim {
            let mut sum = self.biases[i];
            for (j, &w) in self.weights[i].iter().enumerate() {
                if j < input.len() {
                    sum += w * input[j];
                }
            }
            output[i] = if self.use_relu {
                sum.max(0.0)
            } else {
                sum
            };
        }
        output
    }
}

// ---------------------------------------------------------------------------
// VAE for LOB
// ---------------------------------------------------------------------------

/// Variational Autoencoder for Limit Order Book data.
///
/// Architecture:
///   Encoder: input → hidden (ReLU) → mu, log_var
///   Decoder: latent → hidden (ReLU) → output
pub struct VaeLob {
    pub input_dim: usize,
    pub hidden_dim: usize,
    pub latent_dim: usize,
    pub beta: f64,

    // Encoder layers
    pub enc_hidden: DenseLayer,
    pub enc_mu: DenseLayer,
    pub enc_log_var: DenseLayer,

    // Decoder layers
    pub dec_hidden: DenseLayer,
    pub dec_output: DenseLayer,
}

impl VaeLob {
    /// Create a new VAE with given dimensions.
    pub fn new(input_dim: usize, hidden_dim: usize, latent_dim: usize) -> Self {
        Self {
            input_dim,
            hidden_dim,
            latent_dim,
            beta: 1.0,
            enc_hidden: DenseLayer::new(input_dim, hidden_dim, true),
            enc_mu: DenseLayer::new(hidden_dim, latent_dim, false),
            enc_log_var: DenseLayer::new(hidden_dim, latent_dim, false),
            dec_hidden: DenseLayer::new(latent_dim, hidden_dim, true),
            dec_output: DenseLayer::new(hidden_dim, input_dim, false),
        }
    }

    /// Set the beta parameter for the Beta-VAE loss.
    pub fn with_beta(mut self, beta: f64) -> Self {
        self.beta = beta;
        self
    }

    /// Encode an input vector to (mu, log_var).
    pub fn encode(&self, x: &Array1<f64>) -> (Array1<f64>, Array1<f64>) {
        let hidden = self.enc_hidden.forward(x);
        let mu = self.enc_mu.forward(&hidden);
        let log_var = self.enc_log_var.forward(&hidden);
        (mu, log_var)
    }

    /// Reparameterization trick: z = mu + sigma * epsilon.
    pub fn reparameterize(&self, mu: &Array1<f64>, log_var: &Array1<f64>) -> Array1<f64> {
        let mut rng = rand::thread_rng();
        let d = mu.len();
        let mut z = Array1::zeros(d);
        for i in 0..d {
            let sigma = (0.5 * log_var[i]).exp();
            let eps: f64 = rng.gen_range(-3.0..3.0) * 0.3; // Truncated normal approximation
            z[i] = mu[i] + sigma * eps;
        }
        z
    }

    /// Decode a latent vector to reconstructed output.
    pub fn decode(&self, z: &Array1<f64>) -> Array1<f64> {
        let hidden = self.dec_hidden.forward(z);
        self.dec_output.forward(&hidden)
    }

    /// Full forward pass: encode, reparameterize, decode.
    pub fn forward(&self, x: &Array1<f64>) -> (Array1<f64>, Array1<f64>, Array1<f64>, Array1<f64>) {
        let (mu, log_var) = self.encode(x);
        let z = self.reparameterize(&mu, &log_var);
        let x_recon = self.decode(&z);
        (x_recon, mu, log_var, z)
    }

    /// Compute the reconstruction loss (MSE).
    pub fn reconstruction_loss(x: &Array1<f64>, x_recon: &Array1<f64>) -> f64 {
        let n = x.len();
        let mut mse = 0.0;
        for i in 0..n {
            let diff = x[i] - x_recon[i];
            mse += diff * diff;
        }
        mse / n as f64
    }

    /// Compute the KL divergence: -0.5 * sum(1 + log_var - mu^2 - exp(log_var)).
    pub fn kl_divergence(mu: &Array1<f64>, log_var: &Array1<f64>) -> f64 {
        let d = mu.len();
        let mut kl = 0.0;
        for i in 0..d {
            kl += 1.0 + log_var[i] - mu[i] * mu[i] - log_var[i].exp();
        }
        -0.5 * kl
    }

    /// Compute the full ELBO loss = reconstruction_loss + beta * KL_divergence.
    pub fn elbo_loss(&self, x: &Array1<f64>, x_recon: &Array1<f64>, mu: &Array1<f64>, log_var: &Array1<f64>) -> f64 {
        let recon = Self::reconstruction_loss(x, x_recon);
        let kl = Self::kl_divergence(mu, log_var);
        recon + self.beta * kl
    }

    /// Interpolate between two latent codes.
    pub fn interpolate(z_a: &Array1<f64>, z_b: &Array1<f64>, alpha: f64) -> Array1<f64> {
        let alpha = alpha.clamp(0.0, 1.0);
        (1.0 - alpha) * z_a + alpha * z_b
    }

    /// Sample a random latent code from the prior N(0, I).
    pub fn sample_prior(&self) -> Array1<f64> {
        let mut rng = rand::thread_rng();
        Array1::from_vec(
            (0..self.latent_dim)
                .map(|_| rng.gen_range(-2.0..2.0) * 0.5)
                .collect(),
        )
    }

    /// Generate a synthetic LOB feature vector by sampling from the prior.
    pub fn generate(&self) -> Array1<f64> {
        let z = self.sample_prior();
        self.decode(&z)
    }
}

// ---------------------------------------------------------------------------
// Anomaly Detector
// ---------------------------------------------------------------------------

/// Detects anomalous LOB states based on VAE reconstruction error.
pub struct AnomalyDetector {
    /// Threshold multiplier (k in: anomaly if score > mean + k * std).
    pub threshold_k: f64,
    /// Running statistics for calibration.
    pub scores: Vec<f64>,
}

impl AnomalyDetector {
    /// Create a new anomaly detector with the given threshold multiplier.
    pub fn new(threshold_k: f64) -> Self {
        Self {
            threshold_k,
            scores: Vec::new(),
        }
    }

    /// Compute the anomaly score (MSE between original and reconstructed).
    pub fn score(&self, original: &Array1<f64>, reconstructed: &Array1<f64>) -> f64 {
        VaeLob::reconstruction_loss(original, reconstructed)
    }

    /// Add a score to the running statistics for calibration.
    pub fn add_score(&mut self, score: f64) {
        self.scores.push(score);
    }

    /// Get the mean of recorded scores.
    pub fn mean_score(&self) -> f64 {
        if self.scores.is_empty() {
            return 0.0;
        }
        self.scores.iter().sum::<f64>() / self.scores.len() as f64
    }

    /// Get the standard deviation of recorded scores.
    pub fn std_score(&self) -> f64 {
        if self.scores.len() < 2 {
            return 0.0;
        }
        let mean = self.mean_score();
        let variance = self.scores.iter().map(|s| (s - mean).powi(2)).sum::<f64>()
            / (self.scores.len() - 1) as f64;
        variance.sqrt()
    }

    /// Compute the anomaly threshold.
    pub fn threshold(&self) -> f64 {
        self.mean_score() + self.threshold_k * self.std_score()
    }

    /// Check if a score is anomalous.
    pub fn is_anomalous(&self, score: f64) -> bool {
        if self.scores.len() < 2 {
            return false; // Not enough data to determine
        }
        score > self.threshold()
    }
}

// ---------------------------------------------------------------------------
// Bybit Client
// ---------------------------------------------------------------------------

/// Bybit API response structures.
#[derive(Debug, Deserialize)]
pub struct BybitResponse {
    #[serde(rename = "retCode")]
    pub ret_code: i32,
    #[serde(rename = "retMsg")]
    pub ret_msg: String,
    pub result: Option<BybitOrderbookResult>,
}

#[derive(Debug, Deserialize)]
pub struct BybitOrderbookResult {
    pub s: String,
    pub b: Vec<Vec<String>>,
    pub a: Vec<Vec<String>>,
    pub ts: u64,
}

/// Client for fetching orderbook data from Bybit V5 API.
pub struct BybitClient {
    base_url: String,
    client: reqwest::Client,
}

impl BybitClient {
    /// Create a new Bybit client.
    pub fn new() -> Self {
        Self {
            base_url: "https://api.bybit.com".to_string(),
            client: reqwest::Client::new(),
        }
    }

    /// Create a client with a custom base URL (useful for testing).
    pub fn with_base_url(base_url: &str) -> Self {
        Self {
            base_url: base_url.to_string(),
            client: reqwest::Client::new(),
        }
    }

    /// Fetch orderbook for a given symbol.
    ///
    /// Uses the `/v5/market/orderbook` endpoint.
    pub async fn fetch_orderbook(&self, symbol: &str, limit: usize) -> Result<LobSnapshot> {
        let url = format!(
            "{}/v5/market/orderbook?category=linear&symbol={}&limit={}",
            self.base_url, symbol, limit
        );

        let response: BybitResponse = self
            .client
            .get(&url)
            .send()
            .await
            .context("Failed to send request to Bybit")?
            .json()
            .await
            .context("Failed to parse Bybit response")?;

        if response.ret_code != 0 {
            anyhow::bail!(
                "Bybit API error: {} (code {})",
                response.ret_msg,
                response.ret_code
            );
        }

        let result = response
            .result
            .context("Missing result in Bybit response")?;

        let bids: Vec<PriceLevel> = result
            .b
            .iter()
            .filter_map(|level| {
                if level.len() >= 2 {
                    Some(PriceLevel {
                        price: level[0].parse().unwrap_or(0.0),
                        quantity: level[1].parse().unwrap_or(0.0),
                    })
                } else {
                    None
                }
            })
            .collect();

        let asks: Vec<PriceLevel> = result
            .a
            .iter()
            .filter_map(|level| {
                if level.len() >= 2 {
                    Some(PriceLevel {
                        price: level[0].parse().unwrap_or(0.0),
                        quantity: level[1].parse().unwrap_or(0.0),
                    })
                } else {
                    None
                }
            })
            .collect();

        Ok(LobSnapshot::new(bids, asks, result.ts))
    }
}

impl Default for BybitClient {
    fn default() -> Self {
        Self::new()
    }
}

// ---------------------------------------------------------------------------
// Tests
// ---------------------------------------------------------------------------

#[cfg(test)]
mod tests {
    use super::*;

    fn sample_snapshot() -> LobSnapshot {
        LobSnapshot::new(
            vec![
                PriceLevel { price: 100.0, quantity: 5.0 },
                PriceLevel { price: 99.5, quantity: 10.0 },
                PriceLevel { price: 99.0, quantity: 15.0 },
                PriceLevel { price: 98.5, quantity: 8.0 },
                PriceLevel { price: 98.0, quantity: 12.0 },
            ],
            vec![
                PriceLevel { price: 100.5, quantity: 4.0 },
                PriceLevel { price: 101.0, quantity: 8.0 },
                PriceLevel { price: 101.5, quantity: 12.0 },
                PriceLevel { price: 102.0, quantity: 6.0 },
                PriceLevel { price: 102.5, quantity: 9.0 },
            ],
            1000,
        )
    }

    #[test]
    fn test_mid_price() {
        let snap = sample_snapshot();
        let mid = snap.mid_price();
        assert!((mid - 100.25).abs() < 1e-10);
    }

    #[test]
    fn test_spread() {
        let snap = sample_snapshot();
        let spread = snap.spread();
        assert!((spread - 0.5).abs() < 1e-10);
    }

    #[test]
    fn test_top_imbalance() {
        let snap = sample_snapshot();
        let imb = snap.top_imbalance();
        // (5 - 4) / (5 + 4) = 1/9
        assert!((imb - 1.0 / 9.0).abs() < 1e-10);
    }

    #[test]
    fn test_feature_extraction() {
        let extractor = LobFeatureExtractor::new(5);
        let snap = sample_snapshot();
        let features = extractor.extract(&snap);
        assert_eq!(features.len(), extractor.feature_dim());

        // First feature: normalized bid price at level 0
        // (100.0 - 100.25) / 100.25 = -0.002493...
        assert!((features[0] - (-0.25 / 100.25)).abs() < 1e-10);

        // Spread feature (at index 4*5 = 20)
        // 0.5 / 100.25 = 0.004987...
        assert!((features[20] - (0.5 / 100.25)).abs() < 1e-10);
    }

    #[test]
    fn test_vae_encode_decode_shapes() {
        let extractor = LobFeatureExtractor::new(5);
        let snap = sample_snapshot();
        let features = extractor.extract(&snap);
        let input_dim = features.len();

        let vae = VaeLob::new(input_dim, 16, 8);

        let (mu, log_var) = vae.encode(&features);
        assert_eq!(mu.len(), 8);
        assert_eq!(log_var.len(), 8);

        let z = vae.reparameterize(&mu, &log_var);
        assert_eq!(z.len(), 8);

        let recon = vae.decode(&z);
        assert_eq!(recon.len(), input_dim);
    }

    #[test]
    fn test_elbo_loss_components() {
        let mu = Array1::from_vec(vec![0.0; 4]);
        let log_var = Array1::from_vec(vec![0.0; 4]);

        // KL with mu=0, log_var=0: -0.5 * sum(1 + 0 - 0 - 1) = 0
        let kl = VaeLob::kl_divergence(&mu, &log_var);
        assert!((kl - 0.0).abs() < 1e-10);

        // Non-zero mu
        let mu2 = Array1::from_vec(vec![1.0; 4]);
        let kl2 = VaeLob::kl_divergence(&mu2, &log_var);
        assert!(kl2 > 0.0); // KL should be positive when mu != 0
    }

    #[test]
    fn test_reconstruction_loss() {
        let x = Array1::from_vec(vec![1.0, 2.0, 3.0]);
        let x_recon = Array1::from_vec(vec![1.0, 2.0, 3.0]);
        let loss = VaeLob::reconstruction_loss(&x, &x_recon);
        assert!((loss - 0.0).abs() < 1e-10);

        let x_bad = Array1::from_vec(vec![2.0, 3.0, 4.0]);
        let loss2 = VaeLob::reconstruction_loss(&x, &x_bad);
        // MSE = (1+1+1)/3 = 1.0
        assert!((loss2 - 1.0).abs() < 1e-10);
    }

    #[test]
    fn test_anomaly_detector() {
        let mut detector = AnomalyDetector::new(2.0);

        // Add some "normal" scores
        for s in &[0.1, 0.12, 0.09, 0.11, 0.10, 0.13, 0.08, 0.11, 0.10, 0.12] {
            detector.add_score(*s);
        }

        let mean = detector.mean_score();
        assert!(mean > 0.09 && mean < 0.13);

        let threshold = detector.threshold();
        assert!(threshold > mean);

        // Normal score should not be anomalous
        assert!(!detector.is_anomalous(0.11));

        // Very high score should be anomalous
        assert!(detector.is_anomalous(1.0));
    }

    #[test]
    fn test_interpolation() {
        let z_a = Array1::from_vec(vec![0.0, 0.0, 0.0]);
        let z_b = Array1::from_vec(vec![1.0, 1.0, 1.0]);

        let z_mid = VaeLob::interpolate(&z_a, &z_b, 0.5);
        assert!((z_mid[0] - 0.5).abs() < 1e-10);
        assert!((z_mid[1] - 0.5).abs() < 1e-10);

        let z_start = VaeLob::interpolate(&z_a, &z_b, 0.0);
        assert!((z_start[0] - 0.0).abs() < 1e-10);

        let z_end = VaeLob::interpolate(&z_a, &z_b, 1.0);
        assert!((z_end[0] - 1.0).abs() < 1e-10);
    }

    #[test]
    fn test_sample_and_generate() {
        let extractor = LobFeatureExtractor::new(5);
        let input_dim = extractor.feature_dim();
        let vae = VaeLob::new(input_dim, 16, 8);

        let z = vae.sample_prior();
        assert_eq!(z.len(), 8);

        let generated = vae.generate();
        assert_eq!(generated.len(), input_dim);
    }

    #[test]
    fn test_full_forward_pass() {
        let extractor = LobFeatureExtractor::new(5);
        let snap = sample_snapshot();
        let features = extractor.extract(&snap);
        let input_dim = features.len();

        let vae = VaeLob::new(input_dim, 16, 8);
        let (x_recon, mu, log_var, z) = vae.forward(&features);

        assert_eq!(x_recon.len(), input_dim);
        assert_eq!(mu.len(), 8);
        assert_eq!(log_var.len(), 8);
        assert_eq!(z.len(), 8);

        // ELBO loss should be computable
        let loss = vae.elbo_loss(&features, &x_recon, &mu, &log_var);
        assert!(loss.is_finite());
    }

    #[test]
    fn test_empty_snapshot() {
        let snap = LobSnapshot::new(vec![], vec![], 0);
        assert_eq!(snap.mid_price(), 0.0);
        assert_eq!(snap.spread(), 0.0);
        assert_eq!(snap.top_imbalance(), 0.0);
    }
}
