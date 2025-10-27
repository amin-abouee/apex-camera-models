//! Compare tiny-solver vs apex-solver for Double Sphere Camera Model Optimization
//!
//! This example compares the performance of two optimization approaches:
//! 1. **tiny-solver**: Uses automatic differentiation for Jacobians
//! 2. **apex-solver**: Uses hand-derived analytical Jacobians
//!
//! Both optimizers use the Levenberg-Marquardt algorithm to minimize reprojection error.
//!
//! Usage:
//! ```bash
//! cargo run --example compare_ds_optimizers -- \
//!   --input-path samples/double_sphere.yaml \
//!   --num-points 500
//! ```

use apex_camera_models::camera::{CameraModel, DoubleSphereModel};
use apex_camera_models::optimization::{DoubleSphereOptimizationCost, Optimizer};
use apex_camera_models::util;
use clap::Parser;
use log::info;
use nalgebra::DVector;
use std::time::Instant;

/// Command-line arguments for the optimizer comparison tool
#[derive(Parser, Debug)]
#[command(author, version, about, long_about = None)]
struct Cli {
    /// Path to the input Double Sphere model YAML file
    #[arg(short = 'p', long, default_value = "samples/double_sphere.yaml")]
    input_path: String,

    /// Number of sample points to generate for optimization
    #[arg(short = 'n', long, default_value = "500")]
    num_points: usize,

    /// Noise level for initial parameter perturbation (0.0 to 1.0)
    #[arg(long, default_value = "0.2")]
    noise_level: f64,

    /// Enable verbose output for both optimizers
    #[arg(short = 'v', long)]
    verbose: bool,
}

/// Stores the results of an optimization run
#[derive(Debug, Clone)]
struct OptimizationResult {
    optimizer_name: String,
    success: bool,
    iterations: Option<usize>,
    optimization_time_ms: f64,
    initial_rmse: f64,
    final_rmse: f64,
    improvement_percent: f64,
    final_params: Vec<f64>,
}

fn main() -> Result<(), Box<dyn std::error::Error>> {
    // Initialize logger
    env_logger::init();

    let cli = Cli::parse();

    println!("\n{}", "=".repeat(80));
    println!("🔬 DOUBLE SPHERE OPTIMIZER COMPARISON");
    println!("{}", "=".repeat(80));
    println!("Comparing tiny-solver (auto-diff) vs apex-solver (analytical Jacobians)\n");
    println!("Input file: {}", cli.input_path);
    println!("Sample points: {}", cli.num_points);
    println!("Noise level: {:.1}%\n", cli.noise_level * 100.0);

    // Step 1: Load reference model
    info!("Loading reference Double Sphere model...");
    let reference_model = DoubleSphereModel::load_from_yaml(&cli.input_path)?;

    println!("📐 Reference Model Parameters:");
    println!(
        "  Intrinsics: fx={:.2}, fy={:.2}, cx={:.2}, cy={:.2}",
        reference_model.intrinsics.fx,
        reference_model.intrinsics.fy,
        reference_model.intrinsics.cx,
        reference_model.intrinsics.cy
    );
    println!(
        "  Distortion: alpha={:.6}, xi={:.6}",
        reference_model.alpha, reference_model.xi
    );
    println!(
        "  Resolution: {}x{}\n",
        reference_model.resolution.width, reference_model.resolution.height
    );

    // Step 2: Generate sample points
    info!("Generating {} sample points...", cli.num_points);
    let (points_2d, points_3d) = util::sample_points(Some(&reference_model), cli.num_points)?;
    println!(
        "✅ Generated {} 3D-2D point correspondences\n",
        points_2d.ncols()
    );

    // Step 3: Create noisy initial model
    let noisy_model = create_noisy_model(&reference_model, cli.noise_level);

    println!("🔀 Initial (Noisy) Model Parameters:");
    println!(
        "  Intrinsics: fx={:.2}, fy={:.2}, cx={:.2}, cy={:.2}",
        noisy_model.intrinsics.fx,
        noisy_model.intrinsics.fy,
        noisy_model.intrinsics.cx,
        noisy_model.intrinsics.cy
    );
    println!(
        "  Distortion: alpha={:.6}, xi={:.6}\n",
        noisy_model.alpha, noisy_model.xi
    );

    // Compute initial reprojection error
    let initial_error =
        util::compute_reprojection_error(Some(&noisy_model), &points_3d, &points_2d)?;
    println!(
        "📊 Initial Reprojection Error: {:.6} pixels (RMSE)\n",
        initial_error.rmse
    );

    // Step 4: Run tiny-solver optimization
    println!("{}", "-".repeat(80));
    println!("🚀 Running tiny-solver (automatic differentiation)...");
    println!("{}", "-".repeat(80));

    let tiny_result = run_tiny_solver_optimization(
        noisy_model.clone(),
        &points_3d,
        &points_2d,
        &reference_model,
        initial_error.rmse,
        cli.verbose,
    )?;

    // Step 5: Run apex-solver optimization
    println!("\n{}", "-".repeat(80));
    println!("🚀 Running apex-solver (analytical Jacobians)...");
    println!("{}", "-".repeat(80));

    let apex_result = run_apex_solver_optimization(
        noisy_model.clone(),
        &points_3d,
        &points_2d,
        &reference_model,
        initial_error.rmse,
        cli.verbose,
    )?;

    // Step 6: Display comparison results
    println!("\n{}", "=".repeat(80));
    println!("📊 OPTIMIZATION COMPARISON RESULTS");
    println!("{}", "=".repeat(80));

    display_comparison_table(&[tiny_result.clone(), apex_result.clone()]);

    // Display parameter comparison
    display_parameter_comparison(&reference_model, &tiny_result, &apex_result);

    // Determine winner
    determine_winner(&tiny_result, &apex_result);

    Ok(())
}

/// Creates a noisy version of the reference model by perturbing parameters
fn create_noisy_model(reference: &DoubleSphereModel, noise_level: f64) -> DoubleSphereModel {
    DoubleSphereModel {
        intrinsics: apex_camera_models::camera::Intrinsics {
            fx: reference.intrinsics.fx * (1.0 + noise_level * 0.25),
            fy: reference.intrinsics.fy * (1.0 - noise_level * 0.35),
            cx: reference.intrinsics.cx + noise_level * 15.0,
            cy: reference.intrinsics.cy - noise_level * 17.0,
        },
        resolution: reference.resolution.clone(),
        alpha: (reference.alpha * (1.0 - noise_level * 0.3)).clamp(1e-6, 1.0),
        xi: reference.xi * (1.0 + noise_level * 0.5),
    }
}

/// Run optimization using tiny-solver
fn run_tiny_solver_optimization(
    initial_model: DoubleSphereModel,
    points_3d: &nalgebra::Matrix3xX<f64>,
    points_2d: &nalgebra::Matrix2xX<f64>,
    reference_model: &DoubleSphereModel,
    initial_rmse: f64,
    verbose: bool,
) -> Result<OptimizationResult, Box<dyn std::error::Error>> {
    let mut optimizer =
        DoubleSphereOptimizationCost::new(initial_model, points_3d.clone(), points_2d.clone());

    // Run linear estimation
    optimizer.linear_estimation()?;

    // Measure optimization time
    let start = Instant::now();
    let result = optimizer.optimize(verbose);
    let elapsed = start.elapsed().as_millis() as f64;

    let success = result.is_ok();

    // Compute final error
    let final_model = DoubleSphereModel {
        intrinsics: optimizer.get_intrinsics(),
        resolution: optimizer.get_resolution(),
        alpha: optimizer.get_distortion()[0],
        xi: optimizer.get_distortion()[1],
    };

    let final_error = util::compute_reprojection_error(Some(&final_model), points_3d, points_2d)?;
    let improvement = ((initial_rmse - final_error.rmse) / initial_rmse) * 100.0;

    // Extract final parameters
    let final_params = vec![
        final_model.intrinsics.fx,
        final_model.intrinsics.fy,
        final_model.intrinsics.cx,
        final_model.intrinsics.cy,
        final_model.alpha,
        final_model.xi,
    ];

    println!(
        "\n✅ tiny-solver optimization completed in {:.2} ms",
        elapsed
    );
    println!("   Final RMSE: {:.6} pixels", final_error.rmse);
    println!("   Improvement: {:.2}%", improvement);

    Ok(OptimizationResult {
        optimizer_name: "tiny-solver".to_string(),
        success,
        iterations: None, // tiny-solver doesn't expose iteration count easily
        optimization_time_ms: elapsed,
        initial_rmse,
        final_rmse: final_error.rmse,
        improvement_percent: improvement,
        final_params,
    })
}

/// Run optimization using apex-solver
fn run_apex_solver_optimization(
    initial_model: DoubleSphereModel,
    points_3d: &nalgebra::Matrix3xX<f64>,
    points_2d: &nalgebra::Matrix2xX<f64>,
    reference_model: &DoubleSphereModel,
    initial_rmse: f64,
    verbose: bool,
) -> Result<OptimizationResult, Box<dyn std::error::Error>> {
    let mut optimizer =
        DoubleSphereOptimizationCost::new(initial_model, points_3d.clone(), points_2d.clone());

    // Run linear estimation
    optimizer.linear_estimation()?;

    // Measure optimization time
    let start = Instant::now();
    let result = optimizer.optimize_with_apex(verbose);
    let elapsed = start.elapsed().as_millis() as f64;

    let success = result.is_ok();

    // Compute final error
    let final_model = DoubleSphereModel {
        intrinsics: optimizer.get_intrinsics(),
        resolution: optimizer.get_resolution(),
        alpha: optimizer.get_distortion()[0],
        xi: optimizer.get_distortion()[1],
    };

    let final_error = util::compute_reprojection_error(Some(&final_model), points_3d, points_2d)?;
    let improvement = ((initial_rmse - final_error.rmse) / initial_rmse) * 100.0;

    // Extract final parameters
    let final_params = vec![
        final_model.intrinsics.fx,
        final_model.intrinsics.fy,
        final_model.intrinsics.cx,
        final_model.intrinsics.cy,
        final_model.alpha,
        final_model.xi,
    ];

    println!(
        "\n✅ apex-solver optimization completed in {:.2} ms",
        elapsed
    );
    println!("   Final RMSE: {:.6} pixels", final_error.rmse);
    println!("   Improvement: {:.2}%", improvement);

    Ok(OptimizationResult {
        optimizer_name: "apex-solver".to_string(),
        success,
        iterations: None, // We could extract this from verbose output
        optimization_time_ms: elapsed,
        initial_rmse,
        final_rmse: final_error.rmse,
        improvement_percent: improvement,
        final_params,
    })
}

/// Display comparison table
fn display_comparison_table(results: &[OptimizationResult]) {
    println!(
        "\n┌{0:─<16}┬{0:─<12}┬{0:─<16}┬{0:─<16}┬{0:─<16}┬{0:─<14}┐",
        ""
    );
    println!(
        "│ {:^14} │ {:^10} │ {:^14} │ {:^14} │ {:^14} │ {:^12} │",
        "Optimizer", "Success", "Initial RMSE", "Final RMSE", "Improvement", "Time (ms)"
    );
    println!(
        "├{0:─<16}┼{0:─<12}┼{0:─<16}┼{0:─<16}┼{0:─<16}┼{0:─<14}┤",
        ""
    );

    for result in results {
        let success_str = if result.success { "✓" } else { "✗" };
        println!(
            "│ {:14} │ {:^10} │ {:14.6} │ {:14.6} │ {:13.2}% │ {:12.2} │",
            result.optimizer_name,
            success_str,
            result.initial_rmse,
            result.final_rmse,
            result.improvement_percent,
            result.optimization_time_ms
        );
    }

    println!(
        "└{0:─<16}┴{0:─<12}┴{0:─<16}┴{0:─<16}┴{0:─<16}┴{0:─<14}┘",
        ""
    );
}

/// Display detailed parameter comparison
fn display_parameter_comparison(
    reference: &DoubleSphereModel,
    tiny_result: &OptimizationResult,
    apex_result: &OptimizationResult,
) {
    println!("\n📋 PARAMETER COMPARISON");
    println!("{}", "-".repeat(80));

    let param_names = ["fx", "fy", "cx", "cy", "alpha", "xi"];
    let reference_params = vec![
        reference.intrinsics.fx,
        reference.intrinsics.fy,
        reference.intrinsics.cx,
        reference.intrinsics.cy,
        reference.alpha,
        reference.xi,
    ];

    println!(
        "\n┌{0:─<12}┬{0:─<16}┬{0:─<16}┬{0:─<16}┬{0:─<16}┬{0:─<16}┐",
        ""
    );
    println!(
        "│ {:^10} │ {:^14} │ {:^14} │ {:^14} │ {:^14} │ {:^14} │",
        "Parameter", "Reference", "tiny-solver", "Δ tiny", "apex-solver", "Δ apex"
    );
    println!(
        "├{0:─<12}┼{0:─<16}┼{0:─<16}┼{0:─<16}┼{0:─<16}┼{0:─<16}┤",
        ""
    );

    for i in 0..param_names.len() {
        let ref_val = reference_params[i];
        let tiny_val = tiny_result.final_params[i];
        let apex_val = apex_result.final_params[i];
        let tiny_diff = (tiny_val - ref_val).abs();
        let apex_diff = (apex_val - ref_val).abs();

        println!(
            "│ {:10} │ {:14.6} │ {:14.6} │ {:14.6} │ {:14.6} │ {:14.6} │",
            param_names[i], ref_val, tiny_val, tiny_diff, apex_val, apex_diff
        );
    }

    println!(
        "└{0:─<12}┴{0:─<16}┴{0:─<16}┴{0:─<16}┴{0:─<16}┴{0:─<16}┘",
        ""
    );
}

/// Determine and announce the winner
fn determine_winner(tiny_result: &OptimizationResult, apex_result: &OptimizationResult) {
    println!("\n{}", "=".repeat(80));
    println!("🏆 WINNER DETERMINATION");
    println!("{}", "=".repeat(80));

    let mut tiny_score = 0;
    let mut apex_score = 0;

    // Criterion 1: Lower final RMSE
    println!("\n1. Final RMSE:");
    if tiny_result.final_rmse < apex_result.final_rmse {
        println!(
            "   ✓ tiny-solver: {:.6} < {:.6}",
            tiny_result.final_rmse, apex_result.final_rmse
        );
        tiny_score += 1;
    } else {
        println!(
            "   ✓ apex-solver: {:.6} < {:.6}",
            apex_result.final_rmse, tiny_result.final_rmse
        );
        apex_score += 1;
    }

    // Criterion 2: Faster optimization time
    println!("\n2. Optimization Time:");
    if tiny_result.optimization_time_ms < apex_result.optimization_time_ms {
        println!(
            "   ✓ tiny-solver: {:.2} ms < {:.2} ms",
            tiny_result.optimization_time_ms, apex_result.optimization_time_ms
        );
        tiny_score += 1;
    } else {
        println!(
            "   ✓ apex-solver: {:.2} ms < {:.2} ms",
            apex_result.optimization_time_ms, tiny_result.optimization_time_ms
        );
        apex_score += 1;
    }

    // Criterion 3: Greater improvement
    println!("\n3. Improvement Percentage:");
    if tiny_result.improvement_percent > apex_result.improvement_percent {
        println!(
            "   ✓ tiny-solver: {:.2}% > {:.2}%",
            tiny_result.improvement_percent, apex_result.improvement_percent
        );
        tiny_score += 1;
    } else {
        println!(
            "   ✓ apex-solver: {:.2}% > {:.2}%",
            apex_result.improvement_percent, tiny_result.improvement_percent
        );
        apex_score += 1;
    }

    println!("\n{}", "-".repeat(80));
    println!(
        "Final Score: tiny-solver ({}) vs apex-solver ({})",
        tiny_score, apex_score
    );
    println!("{}", "-".repeat(80));

    if tiny_score > apex_score {
        println!("\n🏆 Winner: tiny-solver (automatic differentiation)");
    } else if apex_score > tiny_score {
        println!("\n🏆 Winner: apex-solver (analytical Jacobians)");
    } else {
        println!("\n🤝 Result: Tie - Both optimizers performed equally well");
    }

    println!("\n{}", "=".repeat(80));
}
