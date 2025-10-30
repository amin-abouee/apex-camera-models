//! Compare tiny-solver vs apex-solver for Camera Model Optimization
//!
//! This example compares the performance of two optimization approaches:
//! 1. **tiny-solver**: Uses automatic differentiation for Jacobians
//! 2. **apex-solver**: Uses hand-derived analytical Jacobians
//!
//! Supports Double Sphere (DS), Kannala-Brandt (KB), Unified Camera Model (UCM), and Extended Unified Camera Model (EUCM).
//!
//! Usage:
//! ```bash
//! # Compare Double Sphere optimizers
//! cargo run --example compare_optimizers -- --model ds --num-points 100
//!
//! # Compare Kannala-Brandt optimizers
//! cargo run --example compare_optimizers -- --model kb --num-points 100 --verbose
//!
//! # Compare UCM optimizers
//! cargo run --example compare_optimizers -- --model ucm --num-points 100
//!
//! # Compare EUCM optimizers
//! cargo run --example compare_optimizers -- --model eucm --num-points 100
//! ```

use apex_camera_models::camera::{
    CameraModel, DoubleSphereModel, EucmModel, KannalaBrandtModel, RadTanModel, UcmModel,
};
use apex_camera_models::optimization::{
    DoubleSphereOptimizationCost, EucmOptimizationCost, KannalaBrandtOptimizationCost, Optimizer,
    RadTanOptimizationCost, UcmOptimizationCost,
};
use apex_camera_models::util;
use clap::Parser;
use std::time::Instant;

/// Command-line arguments for the optimizer comparison tool
#[derive(Parser, Debug)]
#[command(author, version, about, long_about = None)]
struct Cli {
    /// Camera model type: "ds" (Double Sphere), "kb" (Kannala-Brandt), "ucm" (Unified Camera Model), "eucm" (Extended Unified Camera Model), or "radtan" (Radial-Tangential)
    #[arg(short = 'm', long, value_parser = ["ds", "kb", "ucm", "eucm", "radtan"])]
    model: String,

    /// Path to the input model YAML file
    #[arg(short = 'p', long)]
    input_path: Option<String>,

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
    optimization_time_ms: f64,
    initial_rmse: f64,
    final_rmse: f64,
    improvement_percent: f64,
    final_params: Vec<f64>,
}

fn main() -> Result<(), Box<dyn std::error::Error>> {
    env_logger::init();
    let cli = Cli::parse();

    // Determine default input path based on model type
    let input_path = cli.input_path.unwrap_or_else(|| match cli.model.as_str() {
        "ds" => "samples/double_sphere.yaml".to_string(),
        "kb" => "samples/kannala_brandt.yaml".to_string(),
        "ucm" => "samples/ucm.yaml".to_string(),
        "eucm" => "samples/eucm.yaml".to_string(),
        "radtan" => "samples/rad_tan.yaml".to_string(),
        _ => unreachable!(),
    });

    println!("\n{}", "=".repeat(80));
    println!("🔬 CAMERA MODEL OPTIMIZER COMPARISON");
    println!("{}", "=".repeat(80));
    println!(
        "Model: {} | tiny-solver (auto-diff) vs apex-solver (analytical Jacobians)\n",
        match cli.model.as_str() {
            "ds" => "Double Sphere",
            "kb" => "Kannala-Brandt",
            "ucm" => "Unified Camera Model",
            "eucm" => "Extended Unified Camera Model",
            "radtan" => "Radial-Tangential",
            _ => unreachable!(),
        }
    );
    println!("Input file: {}", input_path);
    println!("Sample points: {}", cli.num_points);
    println!("Noise level: {:.1}%\n", cli.noise_level * 100.0);

    match cli.model.as_str() {
        "ds" => run_ds_comparison(&input_path, cli.num_points, cli.noise_level, cli.verbose),
        "kb" => run_kb_comparison(&input_path, cli.num_points, cli.noise_level, cli.verbose),
        "ucm" => run_ucm_comparison(&input_path, cli.num_points, cli.noise_level, cli.verbose),
        "eucm" => run_eucm_comparison(&input_path, cli.num_points, cli.noise_level, cli.verbose),
        "radtan" => {
            run_radtan_comparison(&input_path, cli.num_points, cli.noise_level, cli.verbose)
        }
        _ => unreachable!(),
    }
}

// ==================== DOUBLE SPHERE COMPARISON ====================

fn run_ds_comparison(
    input_path: &str,
    num_points: usize,
    noise_level: f64,
    verbose: bool,
) -> Result<(), Box<dyn std::error::Error>> {
    let reference_model = DoubleSphereModel::load_from_yaml(input_path)?;

    display_ds_model_params(&reference_model, "Reference");

    let (points_2d, points_3d) = util::sample_points(Some(&reference_model), num_points)?;
    println!(
        "✅ Generated {} 3D-2D point correspondences\n",
        points_2d.ncols()
    );

    let noisy_model = create_noisy_ds_model(&reference_model, noise_level);
    display_ds_model_params(&noisy_model, "Initial (Noisy)");

    let initial_error =
        util::compute_reprojection_error(Some(&noisy_model), &points_3d, &points_2d)?;
    println!(
        "📊 Initial Reprojection Error: {:.6} pixels (RMSE)\n",
        initial_error.rmse
    );

    println!("{}", "-".repeat(80));
    println!("🚀 Running tiny-solver (automatic differentiation)...");
    println!("{}", "-".repeat(80));
    let tiny_result = run_ds_tiny_solver(
        noisy_model.clone(),
        &points_3d,
        &points_2d,
        initial_error.rmse,
        verbose,
    )?;

    println!("\n{}", "-".repeat(80));
    println!("🚀 Running apex-solver (analytical Jacobians)...");
    println!("{}", "-".repeat(80));
    let apex_result = run_ds_apex_solver(
        noisy_model,
        &points_3d,
        &points_2d,
        initial_error.rmse,
        verbose,
    )?;

    println!("\n{}", "=".repeat(80));
    println!("📊 OPTIMIZATION COMPARISON RESULTS");
    println!("{}", "=".repeat(80));

    display_comparison_table(&[tiny_result.clone(), apex_result.clone()]);
    display_ds_parameter_comparison(&reference_model, &tiny_result, &apex_result);
    determine_winner(&tiny_result, &apex_result);

    Ok(())
}

fn create_noisy_ds_model(reference: &DoubleSphereModel, noise_level: f64) -> DoubleSphereModel {
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

fn run_ds_tiny_solver(
    initial_model: DoubleSphereModel,
    points_3d: &nalgebra::Matrix3xX<f64>,
    points_2d: &nalgebra::Matrix2xX<f64>,
    initial_rmse: f64,
    verbose: bool,
) -> Result<OptimizationResult, Box<dyn std::error::Error>> {
    let mut optimizer =
        DoubleSphereOptimizationCost::new(initial_model, points_3d.clone(), points_2d.clone());
    optimizer.linear_estimation()?;

    let start = Instant::now();
    let result = optimizer.optimize(verbose);
    let elapsed = start.elapsed().as_millis() as f64;

    let final_model = DoubleSphereModel {
        intrinsics: optimizer.get_intrinsics(),
        resolution: optimizer.get_resolution(),
        alpha: optimizer.get_distortion()[0],
        xi: optimizer.get_distortion()[1],
    };

    let final_error = util::compute_reprojection_error(Some(&final_model), points_3d, points_2d)?;
    let improvement = ((initial_rmse - final_error.rmse) / initial_rmse) * 100.0;

    println!("\n✅ tiny-solver completed in {:.2} ms", elapsed);
    println!(
        "   Final RMSE: {:.6} pixels | Improvement: {:.2}%",
        final_error.rmse, improvement
    );

    Ok(OptimizationResult {
        optimizer_name: "tiny-solver".to_string(),
        success: result.is_ok(),
        optimization_time_ms: elapsed,
        initial_rmse,
        final_rmse: final_error.rmse,
        improvement_percent: improvement,
        final_params: vec![
            final_model.intrinsics.fx,
            final_model.intrinsics.fy,
            final_model.intrinsics.cx,
            final_model.intrinsics.cy,
            final_model.alpha,
            final_model.xi,
        ],
    })
}

fn run_ds_apex_solver(
    initial_model: DoubleSphereModel,
    points_3d: &nalgebra::Matrix3xX<f64>,
    points_2d: &nalgebra::Matrix2xX<f64>,
    initial_rmse: f64,
    verbose: bool,
) -> Result<OptimizationResult, Box<dyn std::error::Error>> {
    let mut optimizer =
        DoubleSphereOptimizationCost::new(initial_model, points_3d.clone(), points_2d.clone());
    optimizer.linear_estimation()?;

    let start = Instant::now();
    let result = optimizer.optimize_with_apex(verbose);
    let elapsed = start.elapsed().as_millis() as f64;

    let final_model = DoubleSphereModel {
        intrinsics: optimizer.get_intrinsics(),
        resolution: optimizer.get_resolution(),
        alpha: optimizer.get_distortion()[0],
        xi: optimizer.get_distortion()[1],
    };

    let final_error = util::compute_reprojection_error(Some(&final_model), points_3d, points_2d)?;
    let improvement = ((initial_rmse - final_error.rmse) / initial_rmse) * 100.0;

    println!("\n✅ apex-solver completed in {:.2} ms", elapsed);
    println!(
        "   Final RMSE: {:.6} pixels | Improvement: {:.2}%",
        final_error.rmse, improvement
    );

    Ok(OptimizationResult {
        optimizer_name: "apex-solver".to_string(),
        success: result.is_ok(),
        optimization_time_ms: elapsed,
        initial_rmse,
        final_rmse: final_error.rmse,
        improvement_percent: improvement,
        final_params: vec![
            final_model.intrinsics.fx,
            final_model.intrinsics.fy,
            final_model.intrinsics.cx,
            final_model.intrinsics.cy,
            final_model.alpha,
            final_model.xi,
        ],
    })
}

// ==================== KANNALA-BRANDT COMPARISON ====================

fn run_kb_comparison(
    input_path: &str,
    num_points: usize,
    noise_level: f64,
    verbose: bool,
) -> Result<(), Box<dyn std::error::Error>> {
    let reference_model = KannalaBrandtModel::load_from_yaml(input_path)?;

    display_kb_model_params(&reference_model, "Reference");

    let (points_2d, points_3d) = util::sample_points(Some(&reference_model), num_points)?;
    println!(
        "✅ Generated {} 3D-2D point correspondences\n",
        points_2d.ncols()
    );

    let noisy_model = create_noisy_kb_model(&reference_model, noise_level);
    display_kb_model_params(&noisy_model, "Initial (Noisy)");

    let initial_error =
        util::compute_reprojection_error(Some(&noisy_model), &points_3d, &points_2d)?;
    println!(
        "📊 Initial Reprojection Error: {:.6} pixels (RMSE)\n",
        initial_error.rmse
    );

    println!("{}", "-".repeat(80));
    println!("🚀 Running tiny-solver (automatic differentiation)...");
    println!("{}", "-".repeat(80));
    let tiny_result = run_kb_tiny_solver(
        noisy_model.clone(),
        &points_3d,
        &points_2d,
        initial_error.rmse,
        verbose,
    )?;

    println!("\n{}", "-".repeat(80));
    println!("🚀 Running apex-solver (analytical Jacobians)...");
    println!("{}", "-".repeat(80));
    let apex_result = run_kb_apex_solver(
        noisy_model,
        &points_3d,
        &points_2d,
        initial_error.rmse,
        verbose,
    )?;

    println!("\n{}", "=".repeat(80));
    println!("📊 OPTIMIZATION COMPARISON RESULTS");
    println!("{}", "=".repeat(80));

    display_comparison_table(&[tiny_result.clone(), apex_result.clone()]);
    display_kb_parameter_comparison(&reference_model, &tiny_result, &apex_result);
    determine_winner(&tiny_result, &apex_result);

    Ok(())
}

fn create_noisy_kb_model(reference: &KannalaBrandtModel, noise_level: f64) -> KannalaBrandtModel {
    KannalaBrandtModel {
        intrinsics: apex_camera_models::camera::Intrinsics {
            fx: reference.intrinsics.fx * (1.0 + noise_level * 0.2),
            fy: reference.intrinsics.fy * (1.0 - noise_level * 0.25),
            cx: reference.intrinsics.cx + noise_level * 12.0,
            cy: reference.intrinsics.cy - noise_level * 15.0,
        },
        resolution: reference.resolution.clone(),
        distortions: [
            reference.distortions[0] * (1.0 + noise_level * 0.3),
            reference.distortions[1] * (1.0 - noise_level * 0.4),
            reference.distortions[2] * (1.0 + noise_level * 0.35),
            reference.distortions[3] * (1.0 - noise_level * 0.3),
        ],
    }
}

fn run_kb_tiny_solver(
    initial_model: KannalaBrandtModel,
    points_3d: &nalgebra::Matrix3xX<f64>,
    points_2d: &nalgebra::Matrix2xX<f64>,
    initial_rmse: f64,
    verbose: bool,
) -> Result<OptimizationResult, Box<dyn std::error::Error>> {
    let mut optimizer =
        KannalaBrandtOptimizationCost::new(initial_model, points_3d.clone(), points_2d.clone());
    optimizer.linear_estimation()?;

    let start = Instant::now();
    let result = optimizer.optimize(verbose);
    let elapsed = start.elapsed().as_millis() as f64;

    let final_model = KannalaBrandtModel {
        intrinsics: optimizer.get_intrinsics(),
        resolution: optimizer.get_resolution(),
        distortions: [
            optimizer.get_distortion()[0],
            optimizer.get_distortion()[1],
            optimizer.get_distortion()[2],
            optimizer.get_distortion()[3],
        ],
    };

    let final_error = util::compute_reprojection_error(Some(&final_model), points_3d, points_2d)?;
    let improvement = ((initial_rmse - final_error.rmse) / initial_rmse) * 100.0;

    println!("\n✅ tiny-solver completed in {:.2} ms", elapsed);
    println!(
        "   Final RMSE: {:.6} pixels | Improvement: {:.2}%",
        final_error.rmse, improvement
    );

    Ok(OptimizationResult {
        optimizer_name: "tiny-solver".to_string(),
        success: result.is_ok(),
        optimization_time_ms: elapsed,
        initial_rmse,
        final_rmse: final_error.rmse,
        improvement_percent: improvement,
        final_params: vec![
            final_model.intrinsics.fx,
            final_model.intrinsics.fy,
            final_model.intrinsics.cx,
            final_model.intrinsics.cy,
            final_model.distortions[0],
            final_model.distortions[1],
            final_model.distortions[2],
            final_model.distortions[3],
        ],
    })
}

fn run_kb_apex_solver(
    initial_model: KannalaBrandtModel,
    points_3d: &nalgebra::Matrix3xX<f64>,
    points_2d: &nalgebra::Matrix2xX<f64>,
    initial_rmse: f64,
    verbose: bool,
) -> Result<OptimizationResult, Box<dyn std::error::Error>> {
    let mut optimizer =
        KannalaBrandtOptimizationCost::new(initial_model, points_3d.clone(), points_2d.clone());
    optimizer.linear_estimation()?;

    let start = Instant::now();
    let result = optimizer.optimize_with_apex(verbose);
    let elapsed = start.elapsed().as_millis() as f64;

    let final_model = KannalaBrandtModel {
        intrinsics: optimizer.get_intrinsics(),
        resolution: optimizer.get_resolution(),
        distortions: [
            optimizer.get_distortion()[0],
            optimizer.get_distortion()[1],
            optimizer.get_distortion()[2],
            optimizer.get_distortion()[3],
        ],
    };

    let final_error = util::compute_reprojection_error(Some(&final_model), points_3d, points_2d)?;
    let improvement = ((initial_rmse - final_error.rmse) / initial_rmse) * 100.0;

    println!("\n✅ apex-solver completed in {:.2} ms", elapsed);
    println!(
        "   Final RMSE: {:.6} pixels | Improvement: {:.2}%",
        final_error.rmse, improvement
    );

    Ok(OptimizationResult {
        optimizer_name: "apex-solver".to_string(),
        success: result.is_ok(),
        optimization_time_ms: elapsed,
        initial_rmse,
        final_rmse: final_error.rmse,
        improvement_percent: improvement,
        final_params: vec![
            final_model.intrinsics.fx,
            final_model.intrinsics.fy,
            final_model.intrinsics.cx,
            final_model.intrinsics.cy,
            final_model.distortions[0],
            final_model.distortions[1],
            final_model.distortions[2],
            final_model.distortions[3],
        ],
    })
}

// ==================== UCM COMPARISON ====================
// NOTE: UCM apex-solver support is not yet implemented
// Uncomment when UcmProjectionFactor and optimize_with_apex are added

/*
fn run_ucm_comparison(
    input_path: &str,
    num_points: usize,
    noise_level: f64,
    verbose: bool,
) -> Result<(), Box<dyn std::error::Error>> {
    // Load reference model
    let ref_model = UcmModel::load_from_yaml(input_path)?;

    display_ucm_model_params(&ref_model, "Reference");

    // Generate point correspondences
    let (points_2d, points_3d) = util::sample_points(Some(&ref_model), num_points)?;
    println!(
        "✅ Generated {} 3D-2D point correspondences\n",
        points_2d.ncols()
    );

    // Create noisy initial model
    let noisy_model = create_noisy_ucm_model(&ref_model, noise_level);
    display_ucm_model_params(&noisy_model, "Initial (Noisy)");

    // Compute initial error
    let initial_error =
        util::compute_reprojection_error(Some(&noisy_model), &points_3d, &points_2d)?;
    println!(
        "📊 Initial Reprojection Error: {:.6} pixels (RMSE)\n",
        initial_error.rmse
    );

    // Run tiny-solver optimization
    let tiny_result = run_ucm_tiny_solver(
        &noisy_model,
        &points_3d,
        &points_2d,
        verbose,
        initial_error.rmse,
    )?;

    // Run apex-solver optimization
    let apex_result = run_ucm_apex_solver(
        &noisy_model,
        &points_3d,
        &points_2d,
        verbose,
        initial_error.rmse,
    )?;

    // Display comparison
    display_comparison_table(&[tiny_result.clone(), apex_result.clone()]);
    display_ucm_parameter_comparison(&ref_model, &tiny_result, &apex_result);
    determine_winner(&tiny_result, &apex_result);

    Ok(())
}

fn create_noisy_ucm_model(model: &UcmModel, noise_level: f64) -> UcmModel {
    UcmModel {
        intrinsics: apex_camera_models::camera::Intrinsics {
            fx: model.intrinsics.fx * (1.0 + noise_level * 0.3),
            fy: model.intrinsics.fy * (1.0 - noise_level * 0.25),
            cx: model.intrinsics.cx + noise_level * 10.0,
            cy: model.intrinsics.cy - noise_level * 12.0,
        },
        resolution: model.resolution.clone(),
        alpha: (model.alpha * (1.0 - noise_level * 0.2)).clamp(1e-6, 1.0),
    }
}

fn run_ucm_tiny_solver(
    initial_model: &UcmModel,
    points_3d: &nalgebra::Matrix3xX<f64>,
    points_2d: &nalgebra::Matrix2xX<f64>,
    verbose: bool,
    initial_rmse: f64,
) -> Result<OptimizationResult, Box<dyn std::error::Error>> {
    println!("{}", "-".repeat(80));
    println!("🚀 Running tiny-solver (automatic differentiation)...");
    println!("{}", "-".repeat(80));
    println!();

    let mut optimizer =
        UcmOptimizationCost::new(initial_model.clone(), points_3d.clone(), points_2d.clone());

    let start = Instant::now();
    optimizer.optimize(verbose)?;
    let elapsed = start.elapsed();

    let final_model = UcmModel::new(&nalgebra::DVector::from_vec(vec![
        optimizer.get_intrinsics().fx,
        optimizer.get_intrinsics().fy,
        optimizer.get_intrinsics().cx,
        optimizer.get_intrinsics().cy,
        optimizer.get_distortion()[0],
    ]))?;

    let final_error = util::compute_reprojection_error(Some(&final_model), points_3d, points_2d)?;
    let improvement = ((initial_rmse - final_error.rmse) / initial_rmse) * 100.0;

    println!(
        "✅ tiny-solver completed in {:.2} ms",
        elapsed.as_secs_f64() * 1000.0
    );
    println!(
        "   Final RMSE: {:.6} pixels | Improvement: {:.2}%\n",
        final_error.rmse, improvement
    );

    Ok(OptimizationResult {
        optimizer_name: "tiny-solver".to_string(),
        success: true,
        optimization_time_ms: elapsed.as_secs_f64() * 1000.0,
        initial_rmse,
        final_rmse: final_error.rmse,
        improvement_percent: improvement,
        final_params: vec![
            final_model.intrinsics.fx,
            final_model.intrinsics.fy,
            final_model.intrinsics.cx,
            final_model.intrinsics.cy,
            final_model.alpha,
        ],
    })
}

fn run_ucm_apex_solver(
    initial_model: &UcmModel,
    points_3d: &nalgebra::Matrix3xX<f64>,
    points_2d: &nalgebra::Matrix2xX<f64>,
    verbose: bool,
    initial_rmse: f64,
) -> Result<OptimizationResult, Box<dyn std::error::Error>> {
    println!("{}", "-".repeat(80));
    println!("🚀 Running apex-solver (analytical Jacobians)...");
    println!("{}", "-".repeat(80));
    println!();

    let mut optimizer =
        UcmOptimizationCost::new(initial_model.clone(), points_3d.clone(), points_2d.clone());

    let start = Instant::now();
    optimizer.optimize_with_apex(verbose)?;
    let elapsed = start.elapsed();

    let final_model = UcmModel::new(&nalgebra::DVector::from_vec(vec![
        optimizer.get_intrinsics().fx,
        optimizer.get_intrinsics().fy,
        optimizer.get_intrinsics().cx,
        optimizer.get_intrinsics().cy,
        optimizer.get_distortion()[0],
    ]))?;

    let final_error = util::compute_reprojection_error(Some(&final_model), points_3d, points_2d)?;
    let improvement = ((initial_rmse - final_error.rmse) / initial_rmse) * 100.0;

    println!(
        "✅ apex-solver completed in {:.2} ms",
        elapsed.as_secs_f64() * 1000.0
    );
    println!(
        "   Final RMSE: {:.6} pixels | Improvement: {:.2}%\n",
        final_error.rmse, improvement
    );

    Ok(OptimizationResult {
        optimizer_name: "apex-solver".to_string(),
        success: true,
        optimization_time_ms: elapsed.as_secs_f64() * 1000.0,
        initial_rmse,
        final_rmse: final_error.rmse,
        improvement_percent: improvement,
        final_params: vec![
            final_model.intrinsics.fx,
            final_model.intrinsics.fy,
            final_model.intrinsics.cx,
            final_model.intrinsics.cy,
            final_model.alpha,
        ],
    })
}
*/

// ==================== EUCM COMPARISON ====================

// ==================== UCM COMPARISON ====================

fn run_ucm_comparison(
    input_path: &str,
    num_points: usize,
    noise_level: f64,
    verbose: bool,
) -> Result<(), Box<dyn std::error::Error>> {
    // Load reference model
    let ref_model = UcmModel::load_from_yaml(input_path)?;

    display_ucm_model_params(&ref_model, "Reference");

    // Generate point correspondences
    let (points_2d, points_3d) = util::sample_points(Some(&ref_model), num_points)?;
    println!(
        "✅ Generated {} 3D-2D point correspondences\n",
        points_2d.ncols()
    );

    // Create noisy initial model
    let noisy_model = create_noisy_ucm_model(&ref_model, noise_level);
    display_ucm_model_params(&noisy_model, "Initial (Noisy)");

    // Compute initial error
    let initial_error =
        util::compute_reprojection_error(Some(&noisy_model), &points_3d, &points_2d)?;
    println!(
        "📊 Initial Reprojection Error: {:.6} pixels (RMSE)\n",
        initial_error.rmse
    );

    // Run tiny-solver optimization
    let tiny_result = run_ucm_tiny_solver(
        &noisy_model,
        &points_3d,
        &points_2d,
        verbose,
        initial_error.rmse,
    )?;

    // Run apex-solver optimization
    let apex_result = run_ucm_apex_solver(
        &noisy_model,
        &points_3d,
        &points_2d,
        verbose,
        initial_error.rmse,
    )?;

    // Display comparison
    display_comparison_table(&[tiny_result.clone(), apex_result.clone()]);
    display_ucm_parameter_comparison(&ref_model, &tiny_result, &apex_result);
    determine_winner(&tiny_result, &apex_result);

    Ok(())
}

fn create_noisy_ucm_model(model: &UcmModel, noise_level: f64) -> UcmModel {
    UcmModel {
        intrinsics: apex_camera_models::camera::Intrinsics {
            fx: model.intrinsics.fx * (1.0 + noise_level * 0.3),
            fy: model.intrinsics.fy * (1.0 - noise_level * 0.25),
            cx: model.intrinsics.cx + noise_level * 10.0,
            cy: model.intrinsics.cy - noise_level * 12.0,
        },
        resolution: model.resolution.clone(),
        alpha: (model.alpha * (1.0 - noise_level * 0.2)).clamp(1e-6, 1.0),
    }
}

fn run_ucm_tiny_solver(
    initial_model: &UcmModel,
    points_3d: &nalgebra::Matrix3xX<f64>,
    points_2d: &nalgebra::Matrix2xX<f64>,
    verbose: bool,
    initial_rmse: f64,
) -> Result<OptimizationResult, Box<dyn std::error::Error>> {
    println!("{}", "-".repeat(80));
    println!("🚀 Running tiny-solver (automatic differentiation)...");
    println!("{}", "-".repeat(80));
    println!();

    let mut optimizer =
        UcmOptimizationCost::new(initial_model.clone(), points_3d.clone(), points_2d.clone());

    optimizer.linear_estimation()?;

    let start = Instant::now();
    optimizer.optimize(verbose)?;
    let elapsed = start.elapsed();

    let final_model = UcmModel::new(&nalgebra::DVector::from_vec(vec![
        optimizer.get_intrinsics().fx,
        optimizer.get_intrinsics().fy,
        optimizer.get_intrinsics().cx,
        optimizer.get_intrinsics().cy,
        optimizer.get_distortion()[0],
    ]))?;

    let final_error = util::compute_reprojection_error(Some(&final_model), points_3d, points_2d)?;
    let improvement = ((initial_rmse - final_error.rmse) / initial_rmse) * 100.0;

    println!(
        "✅ tiny-solver completed in {:.2} ms",
        elapsed.as_secs_f64() * 1000.0
    );
    println!(
        "   Final RMSE: {:.6} pixels | Improvement: {:.2}%\n",
        final_error.rmse, improvement
    );

    Ok(OptimizationResult {
        optimizer_name: "tiny-solver".to_string(),
        success: true,
        optimization_time_ms: elapsed.as_secs_f64() * 1000.0,
        initial_rmse,
        final_rmse: final_error.rmse,
        improvement_percent: improvement,
        final_params: vec![
            final_model.intrinsics.fx,
            final_model.intrinsics.fy,
            final_model.intrinsics.cx,
            final_model.intrinsics.cy,
            final_model.alpha,
        ],
    })
}

fn run_ucm_apex_solver(
    initial_model: &UcmModel,
    points_3d: &nalgebra::Matrix3xX<f64>,
    points_2d: &nalgebra::Matrix2xX<f64>,
    verbose: bool,
    initial_rmse: f64,
) -> Result<OptimizationResult, Box<dyn std::error::Error>> {
    println!("{}", "-".repeat(80));
    println!("🚀 Running apex-solver (analytical Jacobians)...");
    println!("{}", "-".repeat(80));
    println!();

    let mut optimizer =
        UcmOptimizationCost::new(initial_model.clone(), points_3d.clone(), points_2d.clone());

    optimizer.linear_estimation()?;

    let start = Instant::now();
    optimizer.optimize_with_apex(verbose)?;
    let elapsed = start.elapsed();

    let final_model = UcmModel::new(&nalgebra::DVector::from_vec(vec![
        optimizer.get_intrinsics().fx,
        optimizer.get_intrinsics().fy,
        optimizer.get_intrinsics().cx,
        optimizer.get_intrinsics().cy,
        optimizer.get_distortion()[0],
    ]))?;

    let final_error = util::compute_reprojection_error(Some(&final_model), points_3d, points_2d)?;
    let improvement = ((initial_rmse - final_error.rmse) / initial_rmse) * 100.0;

    println!(
        "✅ apex-solver completed in {:.2} ms",
        elapsed.as_secs_f64() * 1000.0
    );
    println!(
        "   Final RMSE: {:.6} pixels | Improvement: {:.2}%\n",
        final_error.rmse, improvement
    );

    Ok(OptimizationResult {
        optimizer_name: "apex-solver".to_string(),
        success: true,
        optimization_time_ms: elapsed.as_secs_f64() * 1000.0,
        initial_rmse,
        final_rmse: final_error.rmse,
        improvement_percent: improvement,
        final_params: vec![
            final_model.intrinsics.fx,
            final_model.intrinsics.fy,
            final_model.intrinsics.cx,
            final_model.intrinsics.cy,
            final_model.alpha,
        ],
    })
}

fn display_ucm_model_params(model: &UcmModel, label: &str) {
    println!("📷 {} UCM Model Parameters:", label);
    println!("   fx: {:.6}", model.intrinsics.fx);
    println!("   fy: {:.6}", model.intrinsics.fy);
    println!("   cx: {:.6}", model.intrinsics.cx);
    println!("   cy: {:.6}", model.intrinsics.cy);
    println!("   alpha: {:.6}", model.alpha);
    println!();
}

fn display_ucm_parameter_comparison(
    reference: &UcmModel,
    tiny_result: &OptimizationResult,
    apex_result: &OptimizationResult,
) {
    println!("\n📋 PARAMETER COMPARISON");
    println!("{}", "-".repeat(80));

    let param_names = ["fx", "fy", "cx", "cy", "alpha"];
    let reference_params = vec![
        reference.intrinsics.fx,
        reference.intrinsics.fy,
        reference.intrinsics.cx,
        reference.intrinsics.cy,
        reference.alpha,
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

// ==================== EUCM COMPARISON ====================

fn run_eucm_comparison(
    input_path: &str,
    num_points: usize,
    noise_level: f64,
    verbose: bool,
) -> Result<(), Box<dyn std::error::Error>> {
    // Load reference model
    let ref_model = EucmModel::load_from_yaml(input_path)?;

    display_eucm_model_params(&ref_model, "Reference");

    // Generate point correspondences
    let (points_2d, points_3d) = util::sample_points(Some(&ref_model), num_points)?;
    println!(
        "✅ Generated {} 3D-2D point correspondences\n",
        points_2d.ncols()
    );

    // Create noisy initial model
    let noisy_model = create_noisy_eucm_model(&ref_model, noise_level);
    display_eucm_model_params(&noisy_model, "Initial (Noisy)");

    // Compute initial error
    let initial_error =
        util::compute_reprojection_error(Some(&noisy_model), &points_3d, &points_2d)?;
    println!(
        "📊 Initial Reprojection Error: {:.6} pixels (RMSE)\n",
        initial_error.rmse
    );

    // Run tiny-solver optimization
    let tiny_result = run_eucm_tiny_solver(
        &noisy_model,
        &points_3d,
        &points_2d,
        verbose,
        initial_error.rmse,
    )?;

    // Run apex-solver optimization
    let apex_result = run_eucm_apex_solver(
        &noisy_model,
        &points_3d,
        &points_2d,
        verbose,
        initial_error.rmse,
    )?;

    // Display comparison
    display_comparison_table(&[tiny_result.clone(), apex_result.clone()]);
    display_eucm_parameter_comparison(&ref_model, &tiny_result, &apex_result);
    determine_winner(&tiny_result, &apex_result);

    Ok(())
}

fn create_noisy_eucm_model(model: &EucmModel, noise_level: f64) -> EucmModel {
    EucmModel {
        intrinsics: apex_camera_models::camera::Intrinsics {
            fx: model.intrinsics.fx * (1.0 + noise_level * 0.3),
            fy: model.intrinsics.fy * (1.0 - noise_level * 0.25),
            cx: model.intrinsics.cx + noise_level * 10.0,
            cy: model.intrinsics.cy - noise_level * 12.0,
        },
        resolution: model.resolution.clone(),
        alpha: model.alpha * (1.0 - noise_level * 0.2),
        beta: model.beta * (1.0 + noise_level * 0.15),
    }
}

fn run_eucm_tiny_solver(
    initial_model: &EucmModel,
    points_3d: &nalgebra::Matrix3xX<f64>,
    points_2d: &nalgebra::Matrix2xX<f64>,
    verbose: bool,
    initial_rmse: f64,
) -> Result<OptimizationResult, Box<dyn std::error::Error>> {
    println!("{}", "-".repeat(80));
    println!("🚀 Running tiny-solver (automatic differentiation)...");
    println!("{}", "-".repeat(80));
    println!();

    let mut optimizer =
        EucmOptimizationCost::new(initial_model.clone(), points_3d.clone(), points_2d.clone());

    let start = Instant::now();
    optimizer.optimize(verbose)?;
    let elapsed = start.elapsed();

    let final_model = EucmModel::new(&nalgebra::DVector::from_vec(vec![
        optimizer.get_intrinsics().fx,
        optimizer.get_intrinsics().fy,
        optimizer.get_intrinsics().cx,
        optimizer.get_intrinsics().cy,
        optimizer.get_distortion()[0],
        optimizer.get_distortion()[1],
    ]))?;

    let final_error = util::compute_reprojection_error(Some(&final_model), points_3d, points_2d)?;
    let improvement = ((initial_rmse - final_error.rmse) / initial_rmse) * 100.0;

    println!(
        "✅ tiny-solver completed in {:.2} ms",
        elapsed.as_secs_f64() * 1000.0
    );
    println!(
        "   Final RMSE: {:.6} pixels | Improvement: {:.2}%\n",
        final_error.rmse, improvement
    );

    Ok(OptimizationResult {
        optimizer_name: "tiny-solver".to_string(),
        success: true,
        optimization_time_ms: elapsed.as_secs_f64() * 1000.0,
        initial_rmse,
        final_rmse: final_error.rmse,
        improvement_percent: improvement,
        final_params: vec![
            final_model.intrinsics.fx,
            final_model.intrinsics.fy,
            final_model.intrinsics.cx,
            final_model.intrinsics.cy,
            final_model.alpha,
            final_model.beta,
        ],
    })
}

fn run_eucm_apex_solver(
    initial_model: &EucmModel,
    points_3d: &nalgebra::Matrix3xX<f64>,
    points_2d: &nalgebra::Matrix2xX<f64>,
    verbose: bool,
    initial_rmse: f64,
) -> Result<OptimizationResult, Box<dyn std::error::Error>> {
    println!("{}", "-".repeat(80));
    println!("🚀 Running apex-solver (analytical Jacobians)...");
    println!("{}", "-".repeat(80));
    println!();

    let mut optimizer =
        EucmOptimizationCost::new(initial_model.clone(), points_3d.clone(), points_2d.clone());

    let start = Instant::now();
    optimizer.optimize_with_apex(verbose)?;
    let elapsed = start.elapsed();

    let final_model = EucmModel::new(&nalgebra::DVector::from_vec(vec![
        optimizer.get_intrinsics().fx,
        optimizer.get_intrinsics().fy,
        optimizer.get_intrinsics().cx,
        optimizer.get_intrinsics().cy,
        optimizer.get_distortion()[0],
        optimizer.get_distortion()[1],
    ]))?;

    let final_error = util::compute_reprojection_error(Some(&final_model), points_3d, points_2d)?;
    let improvement = ((initial_rmse - final_error.rmse) / initial_rmse) * 100.0;

    println!(
        "✅ apex-solver completed in {:.2} ms",
        elapsed.as_secs_f64() * 1000.0
    );
    println!(
        "   Final RMSE: {:.6} pixels | Improvement: {:.2}%\n",
        final_error.rmse, improvement
    );

    Ok(OptimizationResult {
        optimizer_name: "apex-solver".to_string(),
        success: true,
        optimization_time_ms: elapsed.as_secs_f64() * 1000.0,
        initial_rmse,
        final_rmse: final_error.rmse,
        improvement_percent: improvement,
        final_params: vec![
            final_model.intrinsics.fx,
            final_model.intrinsics.fy,
            final_model.intrinsics.cx,
            final_model.intrinsics.cy,
            final_model.alpha,
            final_model.beta,
        ],
    })
}

// ==================== RADTAN COMPARISON ====================

fn run_radtan_comparison(
    input_path: &str,
    num_points: usize,
    noise_level: f64,
    verbose: bool,
) -> Result<(), Box<dyn std::error::Error>> {
    // Load reference model
    let ref_model = RadTanModel::load_from_yaml(input_path)?;

    display_radtan_model_params(&ref_model, "Reference");

    // Generate point correspondences
    let (points_2d, points_3d) = util::sample_points(Some(&ref_model), num_points)?;
    println!(
        "✅ Generated {} 3D-2D point correspondences\n",
        points_2d.ncols()
    );

    // Create noisy initial model
    let noisy_model = create_noisy_radtan_model(&ref_model, noise_level);
    display_radtan_model_params(&noisy_model, "Initial (Noisy)");

    // Compute initial error
    let initial_error =
        util::compute_reprojection_error(Some(&noisy_model), &points_3d, &points_2d)?;
    println!(
        "📊 Initial Reprojection Error: {:.6} pixels (RMSE)\n",
        initial_error.rmse
    );

    // Run tiny-solver optimization
    let tiny_result = run_radtan_tiny_solver(
        &noisy_model,
        &points_3d,
        &points_2d,
        verbose,
        initial_error.rmse,
    )?;

    // Run apex-solver optimization
    let apex_result = run_radtan_apex_solver(
        &noisy_model,
        &points_3d,
        &points_2d,
        verbose,
        initial_error.rmse,
    )?;

    // Display comparison
    display_comparison_table(&[tiny_result.clone(), apex_result.clone()]);
    display_radtan_parameter_comparison(&ref_model, &tiny_result, &apex_result);
    determine_winner(&tiny_result, &apex_result);

    Ok(())
}

fn create_noisy_radtan_model(model: &RadTanModel, noise_level: f64) -> RadTanModel {
    RadTanModel {
        intrinsics: apex_camera_models::camera::Intrinsics {
            fx: model.intrinsics.fx * (1.0 + noise_level * 0.3),
            fy: model.intrinsics.fy * (1.0 - noise_level * 0.25),
            cx: model.intrinsics.cx + noise_level * 10.0,
            cy: model.intrinsics.cy - noise_level * 12.0,
        },
        resolution: model.resolution.clone(),
        distortions: [
            model.distortions[0] * (1.0 + noise_level * 0.4), // k1
            model.distortions[1] * (1.0 - noise_level * 0.35), // k2
            model.distortions[2] * (1.0 + noise_level * 0.3), // p1
            model.distortions[3] * (1.0 - noise_level * 0.3), // p2
            model.distortions[4] * (1.0 + noise_level * 0.25), // k3
        ],
    }
}

fn run_radtan_tiny_solver(
    initial_model: &RadTanModel,
    points_3d: &nalgebra::Matrix3xX<f64>,
    points_2d: &nalgebra::Matrix2xX<f64>,
    verbose: bool,
    initial_rmse: f64,
) -> Result<OptimizationResult, Box<dyn std::error::Error>> {
    println!("{}", "-".repeat(80));
    println!("🚀 Running tiny-solver (automatic differentiation)...");
    println!("{}", "-".repeat(80));
    println!();

    let mut optimizer =
        RadTanOptimizationCost::new(initial_model.clone(), points_3d.clone(), points_2d.clone());

    let start = Instant::now();
    optimizer.optimize(verbose)?;
    let elapsed = start.elapsed();

    let mut final_model = RadTanModel::new(&nalgebra::DVector::from_vec(vec![
        optimizer.get_intrinsics().fx,
        optimizer.get_intrinsics().fy,
        optimizer.get_intrinsics().cx,
        optimizer.get_intrinsics().cy,
        optimizer.get_distortion()[0], // k1
        optimizer.get_distortion()[1], // k2
        optimizer.get_distortion()[2], // p1
        optimizer.get_distortion()[3], // p2
        optimizer.get_distortion()[4], // k3
    ]))?;
    final_model.resolution = initial_model.resolution.clone();

    let final_error = util::compute_reprojection_error(Some(&final_model), points_3d, points_2d)?;
    let improvement = ((initial_rmse - final_error.rmse) / initial_rmse) * 100.0;

    println!(
        "✅ tiny-solver completed in {:.2} ms",
        elapsed.as_secs_f64() * 1000.0
    );
    println!(
        "   Final RMSE: {:.6} pixels | Improvement: {:.2}%\n",
        final_error.rmse, improvement
    );

    Ok(OptimizationResult {
        optimizer_name: "tiny-solver".to_string(),
        success: true,
        optimization_time_ms: elapsed.as_secs_f64() * 1000.0,
        initial_rmse,
        final_rmse: final_error.rmse,
        improvement_percent: improvement,
        final_params: vec![
            final_model.intrinsics.fx,
            final_model.intrinsics.fy,
            final_model.intrinsics.cx,
            final_model.intrinsics.cy,
            final_model.distortions[0],
            final_model.distortions[1],
            final_model.distortions[2],
            final_model.distortions[3],
            final_model.distortions[4],
        ],
    })
}

fn run_radtan_apex_solver(
    initial_model: &RadTanModel,
    points_3d: &nalgebra::Matrix3xX<f64>,
    points_2d: &nalgebra::Matrix2xX<f64>,
    verbose: bool,
    initial_rmse: f64,
) -> Result<OptimizationResult, Box<dyn std::error::Error>> {
    println!("{}", "-".repeat(80));
    println!("🚀 Running apex-solver (analytical Jacobians)...");
    println!("{}", "-".repeat(80));
    println!();

    let mut optimizer =
        RadTanOptimizationCost::new(initial_model.clone(), points_3d.clone(), points_2d.clone());

    let start = Instant::now();
    optimizer.optimize_with_apex(verbose)?;
    let elapsed = start.elapsed();

    let mut final_model = RadTanModel::new(&nalgebra::DVector::from_vec(vec![
        optimizer.get_intrinsics().fx,
        optimizer.get_intrinsics().fy,
        optimizer.get_intrinsics().cx,
        optimizer.get_intrinsics().cy,
        optimizer.get_distortion()[0], // k1
        optimizer.get_distortion()[1], // k2
        optimizer.get_distortion()[2], // p1
        optimizer.get_distortion()[3], // p2
        optimizer.get_distortion()[4], // k3
    ]))?;
    final_model.resolution = initial_model.resolution.clone();

    let final_error = util::compute_reprojection_error(Some(&final_model), points_3d, points_2d)?;
    let improvement = ((initial_rmse - final_error.rmse) / initial_rmse) * 100.0;

    println!(
        "✅ apex-solver completed in {:.2} ms",
        elapsed.as_secs_f64() * 1000.0
    );
    println!(
        "   Final RMSE: {:.6} pixels | Improvement: {:.2}%\n",
        final_error.rmse, improvement
    );

    Ok(OptimizationResult {
        optimizer_name: "apex-solver".to_string(),
        success: true,
        optimization_time_ms: elapsed.as_secs_f64() * 1000.0,
        initial_rmse,
        final_rmse: final_error.rmse,
        improvement_percent: improvement,
        final_params: vec![
            final_model.intrinsics.fx,
            final_model.intrinsics.fy,
            final_model.intrinsics.cx,
            final_model.intrinsics.cy,
            final_model.distortions[0],
            final_model.distortions[1],
            final_model.distortions[2],
            final_model.distortions[3],
            final_model.distortions[4],
        ],
    })
}

// ==================== DISPLAY FUNCTIONS ====================

fn display_ds_model_params(model: &DoubleSphereModel, label: &str) {
    println!("📐 {} Model Parameters:", label);
    println!(
        "  Intrinsics: fx={:.2}, fy={:.2}, cx={:.2}, cy={:.2}",
        model.intrinsics.fx, model.intrinsics.fy, model.intrinsics.cx, model.intrinsics.cy
    );
    println!("  Distortion: alpha={:.6}, xi={:.6}", model.alpha, model.xi);
    println!(
        "  Resolution: {}x{}\n",
        model.resolution.width, model.resolution.height
    );
}

fn display_kb_model_params(model: &KannalaBrandtModel, label: &str) {
    println!("📐 {} Model Parameters:", label);
    println!(
        "  Intrinsics: fx={:.2}, fy={:.2}, cx={:.2}, cy={:.2}",
        model.intrinsics.fx, model.intrinsics.fy, model.intrinsics.cx, model.intrinsics.cy
    );
    println!(
        "  Distortion: k1={:.6}, k2={:.6}, k3={:.6}, k4={:.6}",
        model.distortions[0], model.distortions[1], model.distortions[2], model.distortions[3]
    );
    println!(
        "  Resolution: {}x{}\n",
        model.resolution.width, model.resolution.height
    );
}

/*
fn display_ucm_model_params(model: &UcmModel, label: &str) {
    println!("📐 {} Model Parameters:", label);
    println!(
        "  Intrinsics: fx={:.2}, fy={:.2}, cx={:.2}, cy={:.2}",
        model.intrinsics.fx, model.intrinsics.fy, model.intrinsics.cx, model.intrinsics.cy
    );
    println!("  Distortion: alpha={:.6}", model.alpha);
    println!(
        "  Resolution: {}x{}\n",
        model.resolution.width, model.resolution.height
    );
}
*/

fn display_eucm_model_params(model: &EucmModel, label: &str) {
    println!("📐 {} Model Parameters:", label);
    println!(
        "  Intrinsics: fx={:.2}, fy={:.2}, cx={:.2}, cy={:.2}",
        model.intrinsics.fx, model.intrinsics.fy, model.intrinsics.cx, model.intrinsics.cy
    );
    println!(
        "  Distortion: alpha={:.6}, beta={:.6}",
        model.alpha, model.beta
    );
    println!(
        "  Resolution: {}x{}\n",
        model.resolution.width, model.resolution.height
    );
}

fn display_radtan_model_params(model: &RadTanModel, label: &str) {
    println!("📐 {} Model Parameters:", label);
    println!(
        "  Intrinsics: fx={:.2}, fy={:.2}, cx={:.2}, cy={:.2}",
        model.intrinsics.fx, model.intrinsics.fy, model.intrinsics.cx, model.intrinsics.cy
    );
    println!(
        "  Distortion: k1={:.6}, k2={:.6}, p1={:.6}, p2={:.6}, k3={:.6}",
        model.distortions[0],
        model.distortions[1],
        model.distortions[2],
        model.distortions[3],
        model.distortions[4]
    );
    println!(
        "  Resolution: {}x{}\n",
        model.resolution.width, model.resolution.height
    );
}

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

fn display_ds_parameter_comparison(
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

fn display_kb_parameter_comparison(
    reference: &KannalaBrandtModel,
    tiny_result: &OptimizationResult,
    apex_result: &OptimizationResult,
) {
    println!("\n📋 PARAMETER COMPARISON");
    println!("{}", "-".repeat(80));

    let param_names = ["fx", "fy", "cx", "cy", "k1", "k2", "k3", "k4"];
    let reference_params = vec![
        reference.intrinsics.fx,
        reference.intrinsics.fy,
        reference.intrinsics.cx,
        reference.intrinsics.cy,
        reference.distortions[0],
        reference.distortions[1],
        reference.distortions[2],
        reference.distortions[3],
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

/*
fn display_ucm_parameter_comparison(
    reference: &UcmModel,
    tiny_result: &OptimizationResult,
    apex_result: &OptimizationResult,
) {
    println!("\n📋 PARAMETER COMPARISON");
    println!("{}", "-".repeat(80));

    let param_names = ["fx", "fy", "cx", "cy", "alpha"];
    let reference_params = vec![
        reference.intrinsics.fx,
        reference.intrinsics.fy,
        reference.intrinsics.cx,
        reference.intrinsics.cy,
        reference.alpha,
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
*/

fn display_eucm_parameter_comparison(
    reference: &EucmModel,
    tiny_result: &OptimizationResult,
    apex_result: &OptimizationResult,
) {
    println!("\n📋 PARAMETER COMPARISON");
    println!("{}", "-".repeat(80));

    let param_names = ["fx", "fy", "cx", "cy", "alpha", "beta"];
    let reference_params = vec![
        reference.intrinsics.fx,
        reference.intrinsics.fy,
        reference.intrinsics.cx,
        reference.intrinsics.cy,
        reference.alpha,
        reference.beta,
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

fn display_radtan_parameter_comparison(
    reference: &RadTanModel,
    tiny_result: &OptimizationResult,
    apex_result: &OptimizationResult,
) {
    println!("\n📋 PARAMETER COMPARISON");
    println!("{}", "-".repeat(80));

    let param_names = ["fx", "fy", "cx", "cy", "k1", "k2", "p1", "p2", "k3"];
    let reference_params = vec![
        reference.intrinsics.fx,
        reference.intrinsics.fy,
        reference.intrinsics.cx,
        reference.intrinsics.cy,
        reference.distortions[0],
        reference.distortions[1],
        reference.distortions[2],
        reference.distortions[3],
        reference.distortions[4],
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

fn determine_winner(tiny_result: &OptimizationResult, apex_result: &OptimizationResult) {
    println!("\n{}", "=".repeat(80));
    println!("🏆 WINNER DETERMINATION");
    println!("{}", "=".repeat(80));

    let mut tiny_score = 0;
    let mut apex_score = 0;

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
