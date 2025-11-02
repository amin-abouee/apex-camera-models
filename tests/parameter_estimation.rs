//! Integration tests for parameter estimation methods

use apex_camera_models::camera::{CameraModel, RadTanModel};
use apex_camera_models::util::sample_points;
use nalgebra::DVector;

#[test]
fn test_rad_tan_linear_estimation() {
    let model = RadTanModel::load_from_yaml("samples/rad_tan.yaml")
        .expect("Failed to load RadTan model");
    
    // Generate sample points using the model
    let (points_2d, points_3d) = sample_points(Some(&model), 50)
        .expect("Failed to generate sample points");
    
    // Create a new model with same intrinsics but zero distortion
    let params = DVector::from_vec(vec![
        model.intrinsics.fx,
        model.intrinsics.fy,
        model.intrinsics.cx,
        model.intrinsics.cy,
        0.0, 0.0, 0.0, 0.0, 0.0, // Zero distortion
    ]);
    let mut estimated_model = RadTanModel::new(&params)
        .expect("Failed to create model");
    estimated_model.resolution = model.resolution.clone();
    
    // Run linear estimation
    let result = estimated_model.linear_estimation(&points_3d, &points_2d);
    
    // Linear estimation should complete without errors
    assert!(result.is_ok(), "Linear estimation should succeed");
    
    // Distortion coefficients should be non-zero after estimation
    let has_nonzero = estimated_model.distortions.iter().any(|&d| d.abs() > 1e-10);
    assert!(has_nonzero, "Estimated distortion should be non-zero");
}

#[test]
fn test_linear_estimation_with_insufficient_points() {
    let model = RadTanModel::load_from_yaml("samples/rad_tan.yaml")
        .expect("Failed to load RadTan model");
    
    // Generate only 2 points (insufficient for estimation)
    let (points_2d, points_3d) = sample_points(Some(&model), 2)
        .expect("Failed to generate sample points");
    
    let params = DVector::from_vec(vec![
        model.intrinsics.fx,
        model.intrinsics.fy,
        model.intrinsics.cx,
        model.intrinsics.cy,
        0.0, 0.0, 0.0, 0.0, 0.0,
    ]);
    let mut estimated_model = RadTanModel::new(&params)
        .expect("Failed to create model");
    estimated_model.resolution = model.resolution.clone();
    
    let result = estimated_model.linear_estimation(&points_3d, &points_2d);
    
    // Should fail with insufficient points
    assert!(result.is_err(), "Should fail with insufficient points");
}

#[test]
fn test_linear_estimation_with_mismatched_points() {
    let model = RadTanModel::load_from_yaml("samples/rad_tan.yaml")
        .expect("Failed to load RadTan model");
    
    let (points_2d, points_3d) = sample_points(Some(&model), 10)
        .expect("Failed to generate sample points");
    
    // Take only first 5 3D points but all 10 2D points
    let mismatched_3d = points_3d.columns(0, 5).into();
    
    let params = DVector::from_vec(vec![
        model.intrinsics.fx,
        model.intrinsics.fy,
        model.intrinsics.cx,
        model.intrinsics.cy,
        0.0, 0.0, 0.0, 0.0, 0.0,
    ]);
    let mut estimated_model = RadTanModel::new(&params)
        .expect("Failed to create model");
    estimated_model.resolution = model.resolution.clone();
    
    let result = estimated_model.linear_estimation(&mismatched_3d, &points_2d);
    
    // Should fail with mismatched point counts
    assert!(result.is_err(), "Should fail with mismatched point counts");
}
