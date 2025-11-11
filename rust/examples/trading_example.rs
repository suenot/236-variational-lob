//! Trading example: Fetch BTCUSDT orderbook from Bybit, encode LOB snapshots
//! with a VAE, explore latent space, and detect anomalies.

use anyhow::Result;
use variational_lob::{
    AnomalyDetector, BybitClient, LobFeatureExtractor, LobSnapshot, PriceLevel, VaeLob,
};

/// Create a synthetic LOB snapshot for demonstration when API is unavailable.
fn synthetic_snapshot(mid: f64, spread: f64, depth: f64, timestamp: u64) -> LobSnapshot {
    let half_spread = spread / 2.0;
    let bids: Vec<PriceLevel> = (0..10)
        .map(|i| PriceLevel {
            price: mid - half_spread - i as f64 * 0.5,
            quantity: depth * (1.0 + 0.3 * (i as f64)),
        })
        .collect();
    let asks: Vec<PriceLevel> = (0..10)
        .map(|i| PriceLevel {
            price: mid + half_spread + i as f64 * 0.5,
            quantity: depth * (1.0 + 0.2 * (i as f64)),
        })
        .collect();
    LobSnapshot::new(bids, asks, timestamp)
}

#[tokio::main]
async fn main() -> Result<()> {
    println!("=== Variational LOB Trading Example ===\n");

    let num_levels = 10;
    let extractor = LobFeatureExtractor::new(num_levels);
    let input_dim = extractor.feature_dim();
    let hidden_dim = 32;
    let latent_dim = 10;

    println!(
        "Feature dimension: {} (from {} levels per side)",
        input_dim, num_levels
    );
    println!(
        "VAE architecture: {} -> {} -> {} -> {} -> {}",
        input_dim, hidden_dim, latent_dim, hidden_dim, input_dim
    );
    println!();

    // -----------------------------------------------------------------------
    // 1. Try to fetch live orderbook from Bybit
    // -----------------------------------------------------------------------
    println!("--- Step 1: Fetch Orderbook Data ---");
    let client = BybitClient::new();
    let snapshot = match client.fetch_orderbook("BTCUSDT", 25).await {
        Ok(snap) => {
            println!("Fetched live BTCUSDT orderbook from Bybit");
            println!(
                "  Mid price: {:.2}, Spread: {:.2}, Top imbalance: {:.4}",
                snap.mid_price(),
                snap.spread(),
                snap.top_imbalance()
            );
            println!("  Bids: {} levels, Asks: {} levels", snap.bids.len(), snap.asks.len());
            snap
        }
        Err(e) => {
            println!("Could not fetch live data: {}. Using synthetic data.", e);
            synthetic_snapshot(50000.0, 1.0, 2.0, 1000)
        }
    };
    println!();

    // -----------------------------------------------------------------------
    // 2. Extract features and encode with VAE
    // -----------------------------------------------------------------------
    println!("--- Step 2: VAE Encoding ---");
    let features = extractor.extract(&snapshot);
    let vae = VaeLob::new(input_dim, hidden_dim, latent_dim).with_beta(1.0);

    let (mu, log_var) = vae.encode(&features);
    let z = vae.reparameterize(&mu, &log_var);
    let reconstructed = vae.decode(&z);

    println!("Latent representation (mu):");
    for (i, val) in mu.iter().enumerate() {
        print!("  z[{}]={:.4}", i, val);
        if i < mu.len() - 1 {
            print!(",");
        }
    }
    println!("\n");

    println!("Latent representation (log_var):");
    for (i, val) in log_var.iter().enumerate() {
        print!("  z[{}]={:.4}", i, val);
        if i < log_var.len() - 1 {
            print!(",");
        }
    }
    println!("\n");

    // -----------------------------------------------------------------------
    // 3. Compute ELBO loss
    // -----------------------------------------------------------------------
    println!("--- Step 3: ELBO Loss ---");
    let recon_loss = VaeLob::reconstruction_loss(&features, &reconstructed);
    let kl_loss = VaeLob::kl_divergence(&mu, &log_var);
    let total_loss = vae.elbo_loss(&features, &reconstructed, &mu, &log_var);
    println!("  Reconstruction loss: {:.6}", recon_loss);
    println!("  KL divergence:      {:.6}", kl_loss);
    println!("  Total ELBO loss:    {:.6}", total_loss);
    println!();

    // -----------------------------------------------------------------------
    // 4. Anomaly detection
    // -----------------------------------------------------------------------
    println!("--- Step 4: Anomaly Detection ---");
    let mut detector = AnomalyDetector::new(2.5);

    // Calibrate with multiple normal snapshots
    println!("Calibrating anomaly detector with normal snapshots...");
    for i in 0..20 {
        let normal_snap = synthetic_snapshot(
            50000.0 + (i as f64) * 10.0,
            1.0 + (i as f64) * 0.05,
            2.0,
            1000 + i,
        );
        let feat = extractor.extract(&normal_snap);
        let (r, _, _, _) = vae.forward(&feat);
        let score = detector.score(&feat, &r);
        detector.add_score(score);
    }

    println!(
        "  Calibration: mean={:.6}, std={:.6}, threshold={:.6}",
        detector.mean_score(),
        detector.std_score(),
        detector.threshold()
    );

    // Test with normal snapshot
    let normal_feat = extractor.extract(&synthetic_snapshot(50000.0, 1.0, 2.0, 2000));
    let (normal_recon, _, _, _) = vae.forward(&normal_feat);
    let normal_score = detector.score(&normal_feat, &normal_recon);
    println!(
        "  Normal snapshot:  score={:.6}, anomalous={}",
        normal_score,
        detector.is_anomalous(normal_score)
    );

    // Test with anomalous snapshot (extreme imbalance)
    let anomaly_snap = {
        let mut snap = synthetic_snapshot(50000.0, 10.0, 100.0, 3000);
        // Make bids extremely thin (liquidity vacuum)
        for bid in &mut snap.bids {
            bid.quantity = 0.001;
        }
        snap
    };
    let anomaly_feat = extractor.extract(&anomaly_snap);
    let (anomaly_recon, _, _, _) = vae.forward(&anomaly_feat);
    let anomaly_score = detector.score(&anomaly_feat, &anomaly_recon);
    println!(
        "  Anomaly snapshot: score={:.6}, anomalous={}",
        anomaly_score,
        detector.is_anomalous(anomaly_score)
    );
    println!();

    // -----------------------------------------------------------------------
    // 5. Latent space interpolation
    // -----------------------------------------------------------------------
    println!("--- Step 5: Latent Space Interpolation ---");
    let snap_a = synthetic_snapshot(50000.0, 0.5, 5.0, 4000); // Tight, deep
    let snap_b = synthetic_snapshot(50000.0, 5.0, 0.5, 4001); // Wide, thin

    let feat_a = extractor.extract(&snap_a);
    let feat_b = extractor.extract(&snap_b);
    let (mu_a, _) = vae.encode(&feat_a);
    let (mu_b, _) = vae.encode(&feat_b);

    println!("Interpolating from tight/deep LOB to wide/thin LOB:");
    for step in 0..=5 {
        let alpha = step as f64 / 5.0;
        let z_interp = VaeLob::interpolate(&mu_a, &mu_b, alpha);
        let x_interp = vae.decode(&z_interp);
        let norm: f64 = x_interp.iter().map(|v| v * v).sum::<f64>().sqrt();
        println!("  alpha={:.1}: ||x||={:.4}, z[0]={:.4}", alpha, norm, z_interp[0]);
    }
    println!();

    // -----------------------------------------------------------------------
    // 6. Synthetic LOB generation
    // -----------------------------------------------------------------------
    println!("--- Step 6: Synthetic LOB Generation ---");
    println!("Generating 5 random LOB feature vectors from prior:");
    for i in 0..5 {
        let generated = vae.generate();
        let norm: f64 = generated.iter().map(|v| v * v).sum::<f64>().sqrt();
        let mean: f64 = generated.iter().sum::<f64>() / generated.len() as f64;
        println!(
            "  Sample {}: dim={}, ||x||={:.4}, mean={:.4}",
            i + 1,
            generated.len(),
            norm,
            mean
        );
    }
    println!();

    println!("=== Variational LOB Trading Example Complete ===");
    Ok(())
}
