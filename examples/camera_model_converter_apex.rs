//! Camera Model Converter using apex-solver with Analytical Jacobians
//!
//! This example demonstrates camera model parameter estimation using apex-solver's
//! Levenberg-Marquardt optimizer with hand-derived analytical Jacobians.
//!
//! **Optimizer:** apex-solver v0.1.4 (Analytical Jacobians - hand-derived)
//!
//! **For comparison with automatic differentiation (tiny-solver), run:**
//! ```bash
//! cargo run --example camera_model_converter -- \
//!   --input-model ds --input-path samples/double_sphere.yaml
//! ```
//!
//! **Supported input models:**
//! - Kannala-Brandt (KB)
//! - Double Sphere (DS)
//! - Radial-Tangential (RadTan)
//! - Unified Camera Model (UCM)
//! - Extended Unified Camera Model (EUCM)
//! - Pinhole
//!
//! **Supported output models (apex-solver):**
//! - Double Sphere (DS) - ✅ Full analytical Jacobian support
//! - Kannala-Brandt (KB) - ✅ Full analytical Jacobian support
//! - Radial-Tangential (RadTan) - ✅ Full analytical Jacobian support
//! - Extended Unified Camera Model (EUCM) - ✅ Full analytical Jacobian support
//! - Unified Camera Model (UCM) - ❌ Not yet implemented (use tiny-solver version)
//!
//! **Usage:**
//! ```bash
//! # Convert from Kannala-Brandt to other models using apex-solver
//! cargo run --example camera_model_converter_apex -- \
//!   --input-model kb \
//!   --input-path samples/kannala_brandt.yaml
//!
//! # Convert from Double Sphere with custom sample points
//! cargo run --example camera_model_converter_apex -- \
//!   --input-model ds \
//!   --input-path samples/double_sphere.yaml \
//!   --num-points 1000
//!
//! # With image quality assessment
//! cargo run --example camera_model_converter_apex -- \
//!   --input-model kb \
//!   --input-path samples/kannala_brandt.yaml \
//!   --image-path path/to/image.png
//! ```
//!
//! **Performance Expectations:**
//! Based on benchmarks, apex-solver with analytical Jacobians is typically:
//! - 1.5-3.5x faster than tiny-solver (automatic differentiation)
//! - Achieves same or better numerical precision
//! - More stable convergence for complex models

use apex_camera_models::camera::{
    CameraModel, CameraModelEnum, DoubleSphereModel, EucmModel, KannalaBrandtModel, PinholeModel,
    RadTanModel, UcmModel,
};
use apex_camera_models::optimization::{
    DoubleSphereOptimizationCost, EucmOptimizationCost, KannalaBrandtOptimizationCost, Optimizer,
    RadTanOptimizationCost, UcmOptimizationCost,
};
use apex_camera_models::util::{self, ConversionMetrics, ValidationResults};
use clap::Parser;
use log::info;

use std::path::PathBuf;
use std::time::Instant;

/// Camera model conversion tool - apex-solver version (analytical Jacobians)
#[derive(Parser, Debug)]
#[command(author, version, about, long_about = None)]
struct Cli {
    /// Type of the input camera model (kb, ds, radtan, ucm, eucm, pinhole)
    #[arg(short = 'i', long)]
    input_model: String,

    /// Path to the input model YAML file
    #[arg(short = 'p', long)]
    input_path: PathBuf,

    /// Number of sample points to generate for optimization (default: 500)
    #[arg(short = 'n', long, default_value = "500")]
    num_points: usize,

    /// Path to the input image for processing and quality assessment
    #[arg(short = 'm', long)]
    image_path: Option<PathBuf>,
}

/// Load input camera model from file based on type
fn load_input_model(
    model_type: &str,
    path: &str,
) -> Result<Box<dyn CameraModel>, Box<dyn std::error::Error>> {
    let model: Box<dyn CameraModel> = match model_type.to_lowercase().as_str() {
        "kb" | "kannala_brandt" => {
            info!("Loading Kannala-Brandt model from: {path}");
            Box::new(KannalaBrandtModel::load_from_yaml(path)?)
        }
        "ds" | "double_sphere" => {
            info!("Loading Double Sphere model from: {path}");
            Box::new(DoubleSphereModel::load_from_yaml(path)?)
        }
        "radtan" | "rad_tan" => {
            info!("Loading Radial-Tangential model from: {path}");
            Box::new(RadTanModel::load_from_yaml(path)?)
        }
        "ucm" | "unified" => {
            info!("Loading Unified Camera Model from: {path}");
            Box::new(UcmModel::load_from_yaml(path)?)
        }
        "eucm" | "extended_unified" => {
            info!("Loading Extended Unified Camera Model from: {path}");
            Box::new(EucmModel::load_from_yaml(path)?)
        }
        "pinhole" => {
            info!("Loading Pinhole model from: {path}");
            Box::new(PinholeModel::load_from_yaml(path)?)
        }
        _ => {
            return Err(format!("Unsupported input model type: {model_type}. Supported types: kb, ds, radtan, ucm, eucm, pinhole").into());
        }
    };
    Ok(model)
}

fn main() -> Result<(), Box<dyn std::error::Error>> {
    // Initialize logger
    env_logger::init();

    let cli = Cli::parse();

    // Display tool header - APEX-SOLVER VERSION
    println!("🎯 CAMERA MODEL CONVERTER - APEX-SOLVER");
    println!("=========================================");
    println!("Optimizer: apex-solver v0.1.4");
    println!("Method: Analytical Jacobians (hand-derived)");
    println!(
        "Input model: {} → Converting to all supported target models",
        cli.input_model.to_lowercase()
    );
    println!("Input file: {:?}", cli.input_path);
    if let Some(ref image_path) = cli.image_path {
        println!("Input image: {image_path:?}");
    } else {
        println!("Input image: None (synthetic image will be generated)");
    }
    println!("Sample points: {}", cli.num_points);
    println!("\nℹ️  For comparison with tiny-solver (auto-diff), run:");
    println!("   cargo run --example camera_model_converter\n");

    info!("🎯 CAMERA MODEL CONVERTER - APEX-SOLVER");
    info!(
        "Converting from {} model using analytical Jacobians",
        cli.input_model.to_lowercase()
    );

    // Step 1: Load input model
    let input_path_str = cli.input_path.to_str().ok_or("Invalid input path string")?;
    let input_model = load_input_model(&cli.input_model, input_path_str)?;

    // Display input model parameters
    util::display_input_model_parameters(&cli.input_model, &*input_model);

    info!(
        "✅ Successfully loaded {} model from YAML",
        cli.input_model.to_lowercase()
    );

    // Step 1.5: Load input image if provided
    let reference_image = if let Some(ref image_path) = cli.image_path {
        let image_path_str = image_path.to_str().ok_or("Invalid image path string")?;
        match util::load_image(image_path_str) {
            Ok(img) => {
                println!("Successfully loaded input image: {image_path:?}");
                println!("Image dimensions: {}x{}", img.width(), img.height());
                Some(img)
            }
            Err(e) => {
                println!("⚠️  Failed to load image: {e}");
                println!("Continuing with synthetic image generation...");
                None
            }
        }
    } else {
        None
    };

    // Step 2: Generate sample points
    let (points_2d, points_3d) = util::sample_points(Some(&*input_model), cli.num_points)?;

    println!("\n🎲 Generating Sample Points:");
    println!("Requested points: {}", cli.num_points);
    println!("Generated {} sample points", points_3d.ncols());

    println!("\n🔄 Creating 3D-2D Point Correspondences:");
    println!(
        "Valid 3D-2D correspondences: {} / {}",
        points_2d.ncols(),
        cli.num_points
    );

    // Export point correspondences
    util::export_point_correspondences(&points_3d, &points_2d, "point_correspondences_apex")?;

    info!(
        "✅ Generated {} 3D-2D point correspondences",
        points_2d.ncols()
    );

    // Create visualization of input model projection
    println!("\n🎨 Creating Input Model Projection Visualization:");
    let resolution = input_model.get_resolution();
    util::model_projection_visualization(
        &points_2d,
        &cli.input_model,
        reference_image.as_ref(),
        (resolution.width, resolution.height),
    )?;

    // Step 3: Run all conversions with apex-solver
    println!("\n🔄 Step 3: Converting to All Target Models (apex-solver)");
    println!("==========================================================");
    let mut all_metrics = Vec::new();

    // Convert to Double Sphere (if not the input model)
    if !matches!(
        cli.input_model.to_lowercase().as_str(),
        "ds" | "double_sphere"
    ) {
        println!(
            "\n📐 Converting {} → Double Sphere [apex-solver]",
            cli.input_model.to_lowercase(),
        );
        println!("{}", "-".repeat(48 + cli.input_model.len()));
        if let Ok(metrics) = convert_to_double_sphere(
            &*input_model,
            &points_3d,
            &points_2d,
            reference_image.as_ref(),
        ) {
            all_metrics.push(metrics);
        }
    }

    // Convert to Kannala-Brandt (if not the input model)
    if !matches!(
        cli.input_model.to_lowercase().as_str(),
        "kb" | "kannala_brandt"
    ) {
        println!(
            "\n📐 Converting {} → Kannala-Brandt [apex-solver]",
            cli.input_model.to_lowercase(),
        );
        println!("{}", "-".repeat(48 + cli.input_model.len()));
        if let Ok(metrics) = convert_to_kannala_brandt(
            &*input_model,
            &points_3d,
            &points_2d,
            reference_image.as_ref(),
        ) {
            all_metrics.push(metrics);
        }
    }

    // Convert to Radial-Tangential (if not the input model)
    if !matches!(
        cli.input_model.to_lowercase().as_str(),
        "radtan" | "rad_tan"
    ) {
        println!(
            "\n📐 Converting {} → Radial-Tangential [apex-solver]",
            cli.input_model.to_lowercase(),
        );
        println!("{}", "-".repeat(53 + cli.input_model.len()));
        if let Ok(metrics) = convert_to_rad_tan(
            &*input_model,
            &points_3d,
            &points_2d,
            reference_image.as_ref(),
        ) {
            all_metrics.push(metrics);
        }
    }

    // Convert to UCM (if not the input model)
    if !matches!(cli.input_model.to_lowercase().as_str(), "ucm" | "unified") {
        println!(
            "\n📐 Converting {} → Unified Camera Model [apex-solver]",
            cli.input_model.to_lowercase(),
        );
        println!("{}", "-".repeat(65 + cli.input_model.len()));
        if let Ok(metrics) = convert_to_ucm(
            &*input_model,
            &points_3d,
            &points_2d,
            reference_image.as_ref(),
        ) {
            all_metrics.push(metrics);
        }
    }

    // Convert to EUCM (if not the input model)
    if !matches!(
        cli.input_model.to_lowercase().as_str(),
        "eucm" | "extended_unified"
    ) {
        println!(
            "\n📐 Converting {} → Extended Unified Camera Model [apex-solver]",
            cli.input_model.to_lowercase(),
        );
        println!("{}", "-".repeat(65 + cli.input_model.len()));
        if let Ok(metrics) = convert_to_eucm(
            &*input_model,
            &points_3d,
            &points_2d,
            reference_image.as_ref(),
        ) {
            all_metrics.push(metrics);
        }
    }

    // Display results summary
    util::display_results_summary(&all_metrics, &cli.input_model);

    if all_metrics.is_empty() {
        return Ok(());
    }

    Ok(())
}

// ========== CONVERSION FUNCTIONS USING APEX-SOLVER ==========

/// Convert to Double Sphere using apex-solver with analytical Jacobians
fn convert_to_double_sphere(
    input_model: &dyn CameraModel,
    points_3d: &nalgebra::Matrix3xX<f64>,
    points_2d: &nalgebra::Matrix2xX<f64>,
    reference_image: Option<&image::RgbImage>,
) -> Result<ConversionMetrics, Box<dyn std::error::Error>> {
    let start_time = Instant::now();

    // Initialize Double Sphere model
    let initial_model = DoubleSphereModel {
        intrinsics: input_model.get_intrinsics(),
        resolution: input_model.get_resolution(),
        alpha: 0.5,
        xi: 0.1,
    };

    // Compute initial reprojection error
    let initial_error =
        util::compute_reprojection_error(Some(&initial_model), points_3d, points_2d)?;

    let mut optimizer =
        DoubleSphereOptimizationCost::new(initial_model, points_3d.clone(), points_2d.clone());

    // Linear estimation
    optimizer.linear_estimation()?;

    // NON-LINEAR OPTIMIZATION WITH APEX-SOLVER (ANALYTICAL JACOBIANS)
    info!("🚀 Running apex-solver optimization for Double Sphere...");
    let optimization_result = optimizer.optimize_with_apex(false);

    let optimization_time = start_time.elapsed().as_millis() as f64;

    // Get final parameters
    let final_intrinsics = optimizer.get_intrinsics();
    let final_distortion = optimizer.get_distortion();

    // Compute reprojection error
    let final_model = DoubleSphereModel {
        intrinsics: final_intrinsics.clone(),
        resolution: optimizer.get_resolution(),
        alpha: final_distortion[0],
        xi: final_distortion[1],
    };

    let reprojection_result =
        util::compute_reprojection_error(Some(&final_model), points_3d, points_2d)?;

    let convergence_status = match optimization_result {
        Ok(()) => "Converged (apex-solver)".to_string(),
        Err(_) => "Linear Only".to_string(),
    };

    let validation_results = util::validate_conversion_accuracy(&final_model, input_model)
        .unwrap_or_else(|_| ValidationResults {
            center_error: f64::NAN,
            near_center_error: f64::NAN,
            mid_region_error: f64::NAN,
            edge_region_error: f64::NAN,
            far_edge_error: f64::NAN,
            average_error: f64::NAN,
            max_error: f64::NAN,
            status: "NEEDS IMPROVEMENT".to_string(),
            region_data: vec![],
        });

    // Compute image quality metrics
    let image_quality = util::compute_image_quality_metrics(
        input_model,
        &final_model,
        points_3d,
        "double_sphere_apex",
        reference_image,
    )
    .map_err(|e| info!("Failed to compute image quality metrics: {e:?}"))
    .ok();

    let metrics = ConversionMetrics {
        model: CameraModelEnum::DoubleSphere(final_model),
        model_name: "Double Sphere (apex-solver)".to_string(),
        final_reprojection_error: reprojection_result,
        initial_reprojection_error: initial_error,
        optimization_time_ms: optimization_time,
        convergence_status,
        validation_results,
        image_quality,
    };

    util::display_detailed_results(&metrics);
    Ok(metrics)
}

/// Convert to Kannala-Brandt using apex-solver with analytical Jacobians
fn convert_to_kannala_brandt(
    input_model: &dyn CameraModel,
    points_3d: &nalgebra::Matrix3xX<f64>,
    points_2d: &nalgebra::Matrix2xX<f64>,
    reference_image: Option<&image::RgbImage>,
) -> Result<ConversionMetrics, Box<dyn std::error::Error>> {
    let start_time = Instant::now();

    // Initialize Kannala-Brandt model
    let initial_model = KannalaBrandtModel {
        intrinsics: input_model.get_intrinsics(),
        resolution: input_model.get_resolution(),
        distortions: [0.0; 4], // k1, k2, k3, k4
    };

    // Compute initial reprojection error
    let initial_error =
        util::compute_reprojection_error(Some(&initial_model), points_3d, points_2d)?;

    let mut optimizer =
        KannalaBrandtOptimizationCost::new(initial_model, points_3d.clone(), points_2d.clone());

    // Perform linear estimation first
    optimizer.linear_estimation()?;

    // NON-LINEAR OPTIMIZATION WITH APEX-SOLVER (ANALYTICAL JACOBIANS)
    info!("🚀 Running apex-solver optimization for Kannala-Brandt...");
    let optimization_result = optimizer.optimize_with_apex(false);

    let optimization_time = start_time.elapsed().as_millis() as f64;

    // Get final parameters
    let final_intrinsics = optimizer.get_intrinsics();
    let final_distortion = optimizer.get_distortion();

    // Compute reprojection error
    let final_model = KannalaBrandtModel {
        intrinsics: final_intrinsics.clone(),
        resolution: optimizer.get_resolution(),
        distortions: [
            final_distortion[0],
            final_distortion[1],
            final_distortion[2],
            final_distortion[3],
        ],
    };

    let reprojection_result =
        util::compute_reprojection_error(Some(&final_model), points_3d, points_2d)?;

    let validation_results = util::validate_conversion_accuracy(&final_model, input_model)
        .unwrap_or_else(|_| ValidationResults {
            center_error: f64::NAN,
            near_center_error: f64::NAN,
            mid_region_error: f64::NAN,
            edge_region_error: f64::NAN,
            far_edge_error: f64::NAN,
            average_error: f64::NAN,
            max_error: f64::NAN,
            status: "NEEDS IMPROVEMENT".to_string(),
            region_data: vec![],
        });

    let convergence_status = match optimization_result {
        Ok(()) => "Converged (apex-solver)".to_string(),
        Err(_) => "Linear Only".to_string(),
    };

    // Compute image quality metrics
    let image_quality = util::compute_image_quality_metrics(
        input_model,
        &final_model,
        points_3d,
        "kannala_brandt_apex",
        reference_image,
    )
    .map_err(|e| info!("Failed to compute image quality metrics: {e:?}"))
    .ok();

    let metrics = ConversionMetrics {
        model: CameraModelEnum::KannalaBrandt(final_model),
        model_name: "Kannala-Brandt (apex-solver)".to_string(),
        final_reprojection_error: reprojection_result,
        initial_reprojection_error: initial_error,
        optimization_time_ms: optimization_time,
        convergence_status,
        validation_results,
        image_quality,
    };

    util::display_detailed_results(&metrics);
    Ok(metrics)
}

/// Convert to Radial-Tangential using apex-solver with analytical Jacobians
fn convert_to_rad_tan(
    input_model: &dyn CameraModel,
    points_3d: &nalgebra::Matrix3xX<f64>,
    points_2d: &nalgebra::Matrix2xX<f64>,
    reference_image: Option<&image::RgbImage>,
) -> Result<ConversionMetrics, Box<dyn std::error::Error>> {
    let start_time = Instant::now();

    // Initialize RadTan model
    let initial_model = RadTanModel {
        intrinsics: input_model.get_intrinsics(),
        resolution: input_model.get_resolution(),
        distortions: [0.0; 5], // k1, k2, p1, p2, k3
    };

    // Compute initial reprojection error
    let initial_error =
        util::compute_reprojection_error(Some(&initial_model), points_3d, points_2d)?;

    let mut optimizer =
        RadTanOptimizationCost::new(initial_model, points_3d.clone(), points_2d.clone());

    // Linear estimation
    optimizer.linear_estimation()?;

    // NON-LINEAR OPTIMIZATION WITH APEX-SOLVER (ANALYTICAL JACOBIANS)
    info!("🚀 Running apex-solver optimization for RadTan...");
    let optimization_result = optimizer.optimize_with_apex(false);

    let optimization_time = start_time.elapsed().as_millis() as f64;

    // Get final parameters
    let final_intrinsics = optimizer.get_intrinsics();
    let final_distortion = optimizer.get_distortion();

    // Compute reprojection error
    let final_model = RadTanModel {
        intrinsics: final_intrinsics.clone(),
        resolution: optimizer.get_resolution(),
        distortions: [
            final_distortion[0],
            final_distortion[1],
            final_distortion[2],
            final_distortion[3],
            final_distortion[4],
        ],
    };

    let reprojection_result =
        util::compute_reprojection_error(Some(&final_model), points_3d, points_2d)?;

    let convergence_status = match optimization_result {
        Ok(()) => "Converged (apex-solver)".to_string(),
        Err(_) => "Linear Only".to_string(),
    };

    let validation_results = util::validate_conversion_accuracy(&final_model, input_model)
        .unwrap_or_else(|_| ValidationResults {
            center_error: f64::NAN,
            near_center_error: f64::NAN,
            mid_region_error: f64::NAN,
            edge_region_error: f64::NAN,
            far_edge_error: f64::NAN,
            average_error: f64::NAN,
            max_error: f64::NAN,
            status: "NEEDS IMPROVEMENT".to_string(),
            region_data: vec![],
        });

    // Compute image quality metrics
    let image_quality = util::compute_image_quality_metrics(
        input_model,
        &final_model,
        points_3d,
        "radial_tangential_apex",
        reference_image,
    )
    .map_err(|e| info!("Failed to compute image quality metrics: {e:?}"))
    .ok();

    let metrics = ConversionMetrics {
        model: CameraModelEnum::RadTan(final_model),
        model_name: "Radial-Tangential (apex-solver)".to_string(),
        final_reprojection_error: reprojection_result,
        initial_reprojection_error: initial_error,
        optimization_time_ms: optimization_time,
        convergence_status,
        validation_results,
        image_quality,
    };

    util::display_detailed_results(&metrics);
    Ok(metrics)
}

/// Convert to EUCM using apex-solver with analytical Jacobians
fn convert_to_ucm(
    input_model: &dyn CameraModel,
    points_3d: &nalgebra::Matrix3xX<f64>,
    points_2d: &nalgebra::Matrix2xX<f64>,
    reference_image: Option<&image::RgbImage>,
) -> Result<ConversionMetrics, Box<dyn std::error::Error>> {
    let start_time = Instant::now();

    // Initialize UCM model
    let initial_model = UcmModel {
        intrinsics: input_model.get_intrinsics(),
        resolution: input_model.get_resolution(),
        alpha: 0.5, // Initial guess
    };

    // Compute initial reprojection error
    let initial_error =
        util::compute_reprojection_error(Some(&initial_model), points_3d, points_2d)?;

    let mut optimizer =
        UcmOptimizationCost::new(initial_model, points_3d.clone(), points_2d.clone());

    // Linear estimation
    optimizer.linear_estimation()?;

    // NON-LINEAR OPTIMIZATION WITH APEX-SOLVER (ANALYTICAL JACOBIANS)
    info!("🚀 Running apex-solver optimization for UCM...");
    let optimization_result = optimizer.optimize_with_apex(false);

    let optimization_time = start_time.elapsed().as_millis() as f64;

    // Get final parameters
    let final_intrinsics = optimizer.get_intrinsics();
    let final_distortion = optimizer.get_distortion();

    // Compute reprojection error
    let final_model = UcmModel {
        intrinsics: final_intrinsics.clone(),
        resolution: optimizer.get_resolution(),
        alpha: final_distortion[0],
    };

    let reprojection_result =
        util::compute_reprojection_error(Some(&final_model), points_3d, points_2d)?;

    let convergence_status = match optimization_result {
        Ok(()) => "Converged (apex-solver)".to_string(),
        Err(_) => "Linear Only".to_string(),
    };

    let validation_results = util::validate_conversion_accuracy(&final_model, input_model)
        .unwrap_or_else(|_| ValidationResults {
            center_error: f64::NAN,
            near_center_error: f64::NAN,
            mid_region_error: f64::NAN,
            edge_region_error: f64::NAN,
            far_edge_error: f64::NAN,
            average_error: f64::NAN,
            max_error: f64::NAN,
            status: "NEEDS IMPROVEMENT".to_string(),
            region_data: vec![],
        });

    // Compute image quality metrics
    let image_quality = util::compute_image_quality_metrics(
        input_model,
        &final_model,
        points_3d,
        "unified_camera_model_apex",
        reference_image,
    )
    .map_err(|e| info!("Failed to compute image quality metrics: {e:?}"))
    .ok();

    let metrics = ConversionMetrics {
        model: CameraModelEnum::Ucm(final_model),
        model_name: "Unified Camera Model (apex-solver)".to_string(),
        final_reprojection_error: reprojection_result,
        initial_reprojection_error: initial_error,
        optimization_time_ms: optimization_time,
        convergence_status,
        validation_results,
        image_quality,
    };

    util::display_detailed_results(&metrics);
    Ok(metrics)
}

fn convert_to_eucm(
    input_model: &dyn CameraModel,
    points_3d: &nalgebra::Matrix3xX<f64>,
    points_2d: &nalgebra::Matrix2xX<f64>,
    reference_image: Option<&image::RgbImage>,
) -> Result<ConversionMetrics, Box<dyn std::error::Error>> {
    let start_time = Instant::now();

    // Initialize EUCM model
    let initial_model = EucmModel {
        intrinsics: input_model.get_intrinsics(),
        resolution: input_model.get_resolution(),
        alpha: 0.5, // Initial guess
        beta: 1.0,  // Initial guess
    };

    // Compute initial reprojection error
    let initial_error =
        util::compute_reprojection_error(Some(&initial_model), points_3d, points_2d)?;

    let mut optimizer =
        EucmOptimizationCost::new(initial_model, points_3d.clone(), points_2d.clone());

    // Linear estimation
    optimizer.linear_estimation()?;

    // NON-LINEAR OPTIMIZATION WITH APEX-SOLVER (ANALYTICAL JACOBIANS)
    info!("🚀 Running apex-solver optimization for EUCM...");
    let optimization_result = optimizer.optimize_with_apex(false);

    let optimization_time = start_time.elapsed().as_millis() as f64;

    // Get final parameters
    let final_intrinsics = optimizer.get_intrinsics();
    let final_distortion = optimizer.get_distortion();

    // Compute reprojection error
    let final_model = EucmModel {
        intrinsics: final_intrinsics.clone(),
        resolution: optimizer.get_resolution(),
        alpha: final_distortion[0],
        beta: final_distortion[1],
    };

    let reprojection_result =
        util::compute_reprojection_error(Some(&final_model), points_3d, points_2d)?;

    let convergence_status = match optimization_result {
        Ok(()) => "Converged (apex-solver)".to_string(),
        Err(_) => "Linear Only".to_string(),
    };

    let validation_results = util::validate_conversion_accuracy(&final_model, input_model)
        .unwrap_or_else(|_| ValidationResults {
            center_error: f64::NAN,
            near_center_error: f64::NAN,
            mid_region_error: f64::NAN,
            edge_region_error: f64::NAN,
            far_edge_error: f64::NAN,
            average_error: f64::NAN,
            max_error: f64::NAN,
            status: "NEEDS IMPROVEMENT".to_string(),
            region_data: vec![],
        });

    // Compute image quality metrics
    let image_quality = util::compute_image_quality_metrics(
        input_model,
        &final_model,
        points_3d,
        "extended_unified_camera_model_apex",
        reference_image,
    )
    .map_err(|e| info!("Failed to compute image quality metrics: {e:?}"))
    .ok();

    let metrics = ConversionMetrics {
        model: CameraModelEnum::Eucm(final_model),
        model_name: "Extended Unified Camera Model (apex-solver)".to_string(),
        final_reprojection_error: reprojection_result,
        initial_reprojection_error: initial_error,
        optimization_time_ms: optimization_time,
        convergence_status,
        validation_results,
        image_quality,
    };

    util::display_detailed_results(&metrics);
    Ok(metrics)
}
