//! Camera Model Converter using apex-solver with Analytical Jacobians
//!
//! This binary demonstrates camera model parameter estimation using apex-solver's
//! Levenberg-Marquardt optimizer with hand-derived analytical Jacobians.
//!
//! **Optimizer:** apex-solver v0.1.4 (Analytical Jacobians - hand-derived)
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
//! - Unified Camera Model (UCM) - ✅ Full analytical Jacobian support
//!
//! **Usage:**
//! ```bash
//! # Convert from Kannala-Brandt to all other models using apex-solver
//! cargo run --bin camera_converter -- \
//!   --input-model kb \
//!   --input-path samples/kannala_brandt.yaml
//!
//! # Convert from Double Sphere with custom sample points
//! cargo run --bin camera_converter -- \
//!   --input-model ds \
//!   --input-path samples/double_sphere.yaml \
//!   --num-points 1000
//!
//! # With image quality assessment
//! cargo run --bin camera_converter -- \
//!   --input-model kb \
//!   --input-path samples/kannala_brandt.yaml \
//!   --image-path path/to/image.png
//! ```

use apex_camera_models::camera::{
    CameraModel, CameraModelEnum, DoubleSphereModel, EucmModel, KannalaBrandtModel, PinholeModel,
    RadTanModel, UcmModel,
};
use apex_camera_models::factors::{
    DoubleSphereProjectionFactor, EucmProjectionFactor, KannalaBrandtProjectionFactor,
    RadTanProjectionFactor, UcmProjectionFactor,
};
use apex_camera_models::util::{self, ConversionMetrics, ValidationResults};
use apex_solver::core::problem::{Problem, VariableEnum};
use apex_solver::manifold::ManifoldType;
use apex_solver::optimizer::levenberg_marquardt::{LevenbergMarquardt, LevenbergMarquardtConfig};
use clap::Parser;
use log::info;
use nalgebra::DVector;
use std::collections::HashMap;

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
    println!();

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
    let mut model = DoubleSphereModel {
        intrinsics: input_model.get_intrinsics(),
        resolution: input_model.get_resolution(),
        alpha: 0.5,
        xi: 0.1,
    };

    // Compute initial reprojection error
    let initial_error = util::compute_reprojection_error(Some(&model), points_3d, points_2d)?;

    // Linear estimation
    model.linear_estimation(points_3d, points_2d)?;

    // Create projection factor
    let factor = DoubleSphereProjectionFactor::new(points_3d.clone(), points_2d.clone());

    // Set up apex-solver problem
    let mut problem = Problem::new();
    problem.add_residual_block(&["params"], Box::new(factor), None);

    // Set initial parameter values
    let initial_params = DVector::from_vec(vec![
        model.intrinsics.fx,
        model.intrinsics.fy,
        model.intrinsics.cx,
        model.intrinsics.cy,
        model.alpha,
        model.xi,
    ]);

    // Set parameter bounds
    problem.set_variable_bounds("params", 0, 1.0, 2000.0); // fx
    problem.set_variable_bounds("params", 1, 1.0, 2000.0); // fy
    problem.set_variable_bounds("params", 2, 0.0, 2000.0); // cx
    problem.set_variable_bounds("params", 3, 0.0, 2000.0); // cy
    problem.set_variable_bounds("params", 4, 1e-6, 1.0); // alpha
    problem.set_variable_bounds("params", 5, -5.0, 5.0); // xi

    // Set up initial values with RN manifold (Euclidean space)
    let mut initial_values = HashMap::new();
    initial_values.insert("params".to_string(), (ManifoldType::RN, initial_params));

    // NON-LINEAR OPTIMIZATION WITH APEX-SOLVER (ANALYTICAL JACOBIANS)
    info!("🚀 Running apex-solver optimization for Double Sphere...");

    // Configure and run optimizer
    let config = LevenbergMarquardtConfig::new()
        .with_max_iterations(100)
        .with_cost_tolerance(1e-6)
        .with_parameter_tolerance(1e-8)
        .with_gradient_tolerance(1e-6)
        .with_verbose(false);

    let mut solver = LevenbergMarquardt::with_config(config);

    // Run optimization
    let optimization_result = solver.optimize(&problem, &initial_values);

    let optimization_time = start_time.elapsed().as_millis() as f64;

    // Extract optimized parameters
    let convergence_status = match &optimization_result {
        Ok(result) => {
            let optimized_var = result
                .parameters
                .get("params")
                .ok_or("Variable not found")?;
            let optimized_params = match optimized_var {
                VariableEnum::Rn(rn_var) => rn_var.value.to_vector(),
                _ => return Err("Unexpected variable type".into()),
            };

            // Update model with optimized parameters
            model.intrinsics.fx = optimized_params[0];
            model.intrinsics.fy = optimized_params[1];
            model.intrinsics.cx = optimized_params[2];
            model.intrinsics.cy = optimized_params[3];
            model.alpha = optimized_params[4];
            model.xi = optimized_params[5];

            "Converged (apex-solver)".to_string()
        }
        Err(_) => "Linear Only".to_string(),
    };

    let reprojection_result = util::compute_reprojection_error(Some(&model), points_3d, points_2d)?;

    let validation_results = util::validate_conversion_accuracy(&model, input_model)
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
        &model,
        points_3d,
        "double_sphere_apex",
        reference_image,
    )
    .map_err(|e| info!("Failed to compute image quality metrics: {e:?}"))
    .ok();

    let metrics = ConversionMetrics {
        model: CameraModelEnum::DoubleSphere(model),
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
    let mut model = KannalaBrandtModel {
        intrinsics: input_model.get_intrinsics(),
        resolution: input_model.get_resolution(),
        distortions: [0.0; 4], // k1, k2, k3, k4
    };

    // Compute initial reprojection error
    let initial_error = util::compute_reprojection_error(Some(&model), points_3d, points_2d)?;

    // Linear estimation
    model.linear_estimation(points_3d, points_2d)?;

    // Create projection factor
    let factor = KannalaBrandtProjectionFactor::new(points_3d.clone(), points_2d.clone());

    // Set up apex-solver problem
    let mut problem = Problem::new();
    problem.add_residual_block(&["params"], Box::new(factor), None);

    // Set initial parameter values
    let initial_params = DVector::from_vec(vec![
        model.intrinsics.fx,
        model.intrinsics.fy,
        model.intrinsics.cx,
        model.intrinsics.cy,
        model.distortions[0],
        model.distortions[1],
        model.distortions[2],
        model.distortions[3],
    ]);

    // Set parameter bounds
    problem.set_variable_bounds("params", 0, 1.0, 2000.0); // fx
    problem.set_variable_bounds("params", 1, 1.0, 2000.0); // fy
    problem.set_variable_bounds("params", 2, 0.0, 2000.0); // cx
    problem.set_variable_bounds("params", 3, 0.0, 2000.0); // cy
    problem.set_variable_bounds("params", 4, -5.0, 5.0); // k1
    problem.set_variable_bounds("params", 5, -5.0, 5.0); // k2
    problem.set_variable_bounds("params", 6, -5.0, 5.0); // k3
    problem.set_variable_bounds("params", 7, -5.0, 5.0); // k4

    // Set up initial values with RN manifold (Euclidean space)
    let mut initial_values = HashMap::new();
    initial_values.insert("params".to_string(), (ManifoldType::RN, initial_params));

    // NON-LINEAR OPTIMIZATION WITH APEX-SOLVER (ANALYTICAL JACOBIANS)
    info!("🚀 Running apex-solver optimization for Kannala-Brandt...");

    // Configure and run optimizer
    let config = LevenbergMarquardtConfig::new()
        .with_max_iterations(100)
        .with_cost_tolerance(1e-6)
        .with_parameter_tolerance(1e-8)
        .with_gradient_tolerance(1e-6)
        .with_verbose(false);

    let mut solver = LevenbergMarquardt::with_config(config);
    let optimization_result = solver.optimize(&problem, &initial_values);

    let optimization_time = start_time.elapsed().as_millis() as f64;

    // Extract optimized parameters
    let convergence_status = match &optimization_result {
        Ok(result) => {
            let optimized_var = result
                .parameters
                .get("params")
                .ok_or("Variable not found")?;
            let optimized_params = match optimized_var {
                VariableEnum::Rn(rn_var) => rn_var.value.to_vector(),
                _ => return Err("Unexpected variable type".into()),
            };

            // Update model with optimized parameters
            model.intrinsics.fx = optimized_params[0];
            model.intrinsics.fy = optimized_params[1];
            model.intrinsics.cx = optimized_params[2];
            model.intrinsics.cy = optimized_params[3];
            model.distortions[0] = optimized_params[4];
            model.distortions[1] = optimized_params[5];
            model.distortions[2] = optimized_params[6];
            model.distortions[3] = optimized_params[7];

            "Converged (apex-solver)".to_string()
        }
        Err(_) => "Linear Only".to_string(),
    };

    let reprojection_result = util::compute_reprojection_error(Some(&model), points_3d, points_2d)?;

    let validation_results = util::validate_conversion_accuracy(&model, input_model)
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
        &model,
        points_3d,
        "kannala_brandt_apex",
        reference_image,
    )
    .map_err(|e| info!("Failed to compute image quality metrics: {e:?}"))
    .ok();

    let metrics = ConversionMetrics {
        model: CameraModelEnum::KannalaBrandt(model),
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
    let mut model = RadTanModel {
        intrinsics: input_model.get_intrinsics(),
        resolution: input_model.get_resolution(),
        distortions: [0.0; 5], // k1, k2, p1, p2, k3
    };

    // Compute initial reprojection error
    let initial_error = util::compute_reprojection_error(Some(&model), points_3d, points_2d)?;

    // Linear estimation
    model.linear_estimation(points_3d, points_2d)?;

    // Create projection factor
    let factor = RadTanProjectionFactor::new(points_3d.clone(), points_2d.clone());

    // Set up apex-solver problem
    let mut problem = Problem::new();
    problem.add_residual_block(&["params"], Box::new(factor), None);

    // Set initial parameter values
    let initial_params = DVector::from_vec(vec![
        model.intrinsics.fx,
        model.intrinsics.fy,
        model.intrinsics.cx,
        model.intrinsics.cy,
        model.distortions[0],
        model.distortions[1],
        model.distortions[2],
        model.distortions[3],
        model.distortions[4],
    ]);

    // Set parameter bounds
    problem.set_variable_bounds("params", 0, 1.0, 2000.0); // fx
    problem.set_variable_bounds("params", 1, 1.0, 2000.0); // fy
    problem.set_variable_bounds("params", 2, 0.0, 2000.0); // cx
    problem.set_variable_bounds("params", 3, 0.0, 2000.0); // cy
    problem.set_variable_bounds("params", 4, -5.0, 5.0); // k1
    problem.set_variable_bounds("params", 5, -5.0, 5.0); // k2
    problem.set_variable_bounds("params", 6, -1.0, 1.0); // p1
    problem.set_variable_bounds("params", 7, -1.0, 1.0); // p2
    problem.set_variable_bounds("params", 8, -5.0, 5.0); // k3

    // Set up initial values with RN manifold (Euclidean space)
    let mut initial_values = HashMap::new();
    initial_values.insert("params".to_string(), (ManifoldType::RN, initial_params));

    // NON-LINEAR OPTIMIZATION WITH APEX-SOLVER (ANALYTICAL JACOBIANS)
    info!("🚀 Running apex-solver optimization for RadTan...");

    // Configure and run optimizer
    let config = LevenbergMarquardtConfig::new()
        .with_max_iterations(100)
        .with_cost_tolerance(1e-6)
        .with_parameter_tolerance(1e-8)
        .with_gradient_tolerance(1e-6)
        .with_verbose(false);

    let mut solver = LevenbergMarquardt::with_config(config);
    let optimization_result = solver.optimize(&problem, &initial_values);

    let optimization_time = start_time.elapsed().as_millis() as f64;

    // Extract optimized parameters
    let convergence_status = match &optimization_result {
        Ok(result) => {
            let optimized_var = result
                .parameters
                .get("params")
                .ok_or("Variable not found")?;
            let optimized_params = match optimized_var {
                VariableEnum::Rn(rn_var) => rn_var.value.to_vector(),
                _ => return Err("Unexpected variable type".into()),
            };

            // Update model with optimized parameters
            model.intrinsics.fx = optimized_params[0];
            model.intrinsics.fy = optimized_params[1];
            model.intrinsics.cx = optimized_params[2];
            model.intrinsics.cy = optimized_params[3];
            model.distortions[0] = optimized_params[4];
            model.distortions[1] = optimized_params[5];
            model.distortions[2] = optimized_params[6];
            model.distortions[3] = optimized_params[7];
            model.distortions[4] = optimized_params[8];

            "Converged (apex-solver)".to_string()
        }
        Err(_) => "Linear Only".to_string(),
    };

    let reprojection_result = util::compute_reprojection_error(Some(&model), points_3d, points_2d)?;

    let validation_results = util::validate_conversion_accuracy(&model, input_model)
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
        &model,
        points_3d,
        "radial_tangential_apex",
        reference_image,
    )
    .map_err(|e| info!("Failed to compute image quality metrics: {e:?}"))
    .ok();

    let metrics = ConversionMetrics {
        model: CameraModelEnum::RadTan(model),
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

/// Convert to UCM using apex-solver with analytical Jacobians
fn convert_to_ucm(
    input_model: &dyn CameraModel,
    points_3d: &nalgebra::Matrix3xX<f64>,
    points_2d: &nalgebra::Matrix2xX<f64>,
    reference_image: Option<&image::RgbImage>,
) -> Result<ConversionMetrics, Box<dyn std::error::Error>> {
    let start_time = Instant::now();

    // Initialize UCM model
    let mut model = UcmModel {
        intrinsics: input_model.get_intrinsics(),
        resolution: input_model.get_resolution(),
        alpha: 0.5, // Initial guess
    };

    // Compute initial reprojection error
    let initial_error = util::compute_reprojection_error(Some(&model), points_3d, points_2d)?;

    // Linear estimation
    model.linear_estimation(points_3d, points_2d)?;

    // Create projection factor
    let factor = UcmProjectionFactor::new(points_3d.clone(), points_2d.clone());

    // Set up apex-solver problem
    let mut problem = Problem::new();
    problem.add_residual_block(&["params"], Box::new(factor), None);

    // Set initial parameter values
    let initial_params = DVector::from_vec(vec![
        model.intrinsics.fx,
        model.intrinsics.fy,
        model.intrinsics.cx,
        model.intrinsics.cy,
        model.alpha,
    ]);

    // Set parameter bounds
    problem.set_variable_bounds("params", 0, 1.0, 2000.0); // fx
    problem.set_variable_bounds("params", 1, 1.0, 2000.0); // fy
    problem.set_variable_bounds("params", 2, 0.0, 2000.0); // cx
    problem.set_variable_bounds("params", 3, 0.0, 2000.0); // cy
    problem.set_variable_bounds("params", 4, 1e-6, 10.0); // alpha

    // Set up initial values with RN manifold (Euclidean space)
    let mut initial_values = HashMap::new();
    initial_values.insert("params".to_string(), (ManifoldType::RN, initial_params));

    // NON-LINEAR OPTIMIZATION WITH APEX-SOLVER (ANALYTICAL JACOBIANS)
    info!("🚀 Running apex-solver optimization for UCM...");

    // Configure and run optimizer
    let config = LevenbergMarquardtConfig::new()
        .with_max_iterations(100)
        .with_cost_tolerance(1e-6)
        .with_parameter_tolerance(1e-8)
        .with_gradient_tolerance(1e-6)
        .with_verbose(false);

    let mut solver = LevenbergMarquardt::with_config(config);
    let optimization_result = solver.optimize(&problem, &initial_values);

    let optimization_time = start_time.elapsed().as_millis() as f64;

    // Extract optimized parameters
    let convergence_status = match &optimization_result {
        Ok(result) => {
            let optimized_var = result
                .parameters
                .get("params")
                .ok_or("Variable not found")?;
            let optimized_params = match optimized_var {
                VariableEnum::Rn(rn_var) => rn_var.value.to_vector(),
                _ => return Err("Unexpected variable type".into()),
            };

            // Update model with optimized parameters
            model.intrinsics.fx = optimized_params[0];
            model.intrinsics.fy = optimized_params[1];
            model.intrinsics.cx = optimized_params[2];
            model.intrinsics.cy = optimized_params[3];
            model.alpha = optimized_params[4];

            "Converged (apex-solver)".to_string()
        }
        Err(_) => "Linear Only".to_string(),
    };

    let reprojection_result = util::compute_reprojection_error(Some(&model), points_3d, points_2d)?;

    let validation_results = util::validate_conversion_accuracy(&model, input_model)
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
        &model,
        points_3d,
        "unified_camera_model_apex",
        reference_image,
    )
    .map_err(|e| info!("Failed to compute image quality metrics: {e:?}"))
    .ok();

    let metrics = ConversionMetrics {
        model: CameraModelEnum::Ucm(model),
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

/// Convert to EUCM using apex-solver with analytical Jacobians
fn convert_to_eucm(
    input_model: &dyn CameraModel,
    points_3d: &nalgebra::Matrix3xX<f64>,
    points_2d: &nalgebra::Matrix2xX<f64>,
    reference_image: Option<&image::RgbImage>,
) -> Result<ConversionMetrics, Box<dyn std::error::Error>> {
    let start_time = Instant::now();

    // Initialize EUCM model
    let mut model = EucmModel {
        intrinsics: input_model.get_intrinsics(),
        resolution: input_model.get_resolution(),
        alpha: 0.5, // Initial guess
        beta: 1.0,  // Initial guess
    };

    // Compute initial reprojection error
    let initial_error = util::compute_reprojection_error(Some(&model), points_3d, points_2d)?;

    // Linear estimation
    model.linear_estimation(points_3d, points_2d)?;

    // Create projection factor
    let factor = EucmProjectionFactor::new(points_3d.clone(), points_2d.clone());

    // Set up apex-solver problem
    let mut problem = Problem::new();
    problem.add_residual_block(&["params"], Box::new(factor), None);

    // Set initial parameter values
    let initial_params = DVector::from_vec(vec![
        model.intrinsics.fx,
        model.intrinsics.fy,
        model.intrinsics.cx,
        model.intrinsics.cy,
        model.alpha,
        model.beta,
    ]);

    // Set parameter bounds
    problem.set_variable_bounds("params", 0, 1.0, 2000.0); // fx
    problem.set_variable_bounds("params", 1, 1.0, 2000.0); // fy
    problem.set_variable_bounds("params", 2, 0.0, 2000.0); // cx
    problem.set_variable_bounds("params", 3, 0.0, 2000.0); // cy
    problem.set_variable_bounds("params", 4, 1e-6, 1.0); // alpha
    problem.set_variable_bounds("params", 5, 1e-6, 5.0); // beta

    // Set up initial values with RN manifold (Euclidean space)
    let mut initial_values = HashMap::new();
    initial_values.insert("params".to_string(), (ManifoldType::RN, initial_params));

    // NON-LINEAR OPTIMIZATION WITH APEX-SOLVER (ANALYTICAL JACOBIANS)
    info!("🚀 Running apex-solver optimization for EUCM...");

    // Configure and run optimizer
    let config = LevenbergMarquardtConfig::new()
        .with_max_iterations(100)
        .with_cost_tolerance(1e-6)
        .with_parameter_tolerance(1e-8)
        .with_gradient_tolerance(1e-6)
        .with_verbose(false);

    let mut solver = LevenbergMarquardt::with_config(config);
    let optimization_result = solver.optimize(&problem, &initial_values);

    let optimization_time = start_time.elapsed().as_millis() as f64;

    // Extract optimized parameters
    let convergence_status = match &optimization_result {
        Ok(result) => {
            let optimized_var = result
                .parameters
                .get("params")
                .ok_or("Variable not found")?;
            let optimized_params = match optimized_var {
                VariableEnum::Rn(rn_var) => rn_var.value.to_vector(),
                _ => return Err("Unexpected variable type".into()),
            };

            // Update model with optimized parameters
            model.intrinsics.fx = optimized_params[0];
            model.intrinsics.fy = optimized_params[1];
            model.intrinsics.cx = optimized_params[2];
            model.intrinsics.cy = optimized_params[3];
            model.alpha = optimized_params[4];
            model.beta = optimized_params[5];

            "Converged (apex-solver)".to_string()
        }
        Err(_) => "Linear Only".to_string(),
    };

    let reprojection_result = util::compute_reprojection_error(Some(&model), points_3d, points_2d)?;

    let validation_results = util::validate_conversion_accuracy(&model, input_model)
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
        &model,
        points_3d,
        "extended_unified_camera_model_apex",
        reference_image,
    )
    .map_err(|e| info!("Failed to compute image quality metrics: {e:?}"))
    .ok();

    let metrics = ConversionMetrics {
        model: CameraModelEnum::Eucm(model),
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
