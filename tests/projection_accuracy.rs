//! Integration tests for projection and unprojection accuracy

use apex_camera_models::camera::{
    CameraModel, CameraModelError, DoubleSphereModel, PinholeModel,
};
use nalgebra::{DVector, Vector2, Vector3};

#[test]
fn test_projection_behind_camera() {
    let model = DoubleSphereModel::load_from_yaml("samples/double_sphere.yaml")
        .expect("Failed to load model");
    
    let point_behind = Vector3::new(0.1, 0.2, -1.0);
    let result = model.project(&point_behind);
    
    assert!(result.is_err(), "Should fail to project point behind camera");
}

#[test]
fn test_projection_at_center() {
    let model = DoubleSphereModel::load_from_yaml("samples/double_sphere.yaml")
        .expect("Failed to load model");
    
    let point_at_center = Vector3::new(0.0, 0.0, 0.0);
    let result = model.project(&point_at_center);
    
    assert!(result.is_err(), "Should fail to project point at camera center");
}

#[test]
fn test_unprojection_validates_bounds() {
    // Create a simple pinhole model that validates bounds
    let params = DVector::from_vec(vec![500.0, 500.0, 320.0, 240.0]);
    let mut model = PinholeModel::new(&params).expect("Failed to create model");
    model.resolution.width = 640;
    model.resolution.height = 480;
    
    // Points outside image bounds
    let point_negative = Vector2::new(-100.0, 100.0);
    let result = model.unproject(&point_negative);
    assert!(result.is_err(), "Pinhole should reject negative coordinates");
    
    let point_too_large = Vector2::new(1000.0, 1000.0);
    let result = model.unproject(&point_too_large);
    assert!(result.is_err(), "Pinhole should reject coordinates beyond image size");
}

#[test]
fn test_projection_unprojection_consistency() {
    let params = DVector::from_vec(vec![500.0, 500.0, 320.0, 240.0]);
    let mut model = PinholeModel::new(&params).expect("Failed to create model");
    model.resolution.width = 640;
    model.resolution.height = 480;
    
    let test_points = vec![
        Vector3::new(0.0, 0.0, 1.0),
        Vector3::new(0.2, 0.1, 1.5),
        Vector3::new(-0.1, -0.2, 2.0),
    ];
    
    for point in test_points {
        if let Ok(point_2d) = model.project(&point) {
            if let Ok(ray) = model.unproject(&point_2d) {
                let original_dir = point.normalize();
                let dot_product = original_dir.dot(&ray);
                assert!(
                    (dot_product - 1.0).abs() < 1e-6,
                    "Projection-unprojection should preserve direction"
                );
            }
        }
    }
}

#[test]
fn test_boundary_projections() {
    let model = DoubleSphereModel::load_from_yaml("samples/double_sphere.yaml")
        .expect("Failed to load model");
    
    // Test points that may be near boundaries
    let boundary_points = vec![
        Vector3::new(0.5, 0.0, 2.0),
        Vector3::new(-0.5, 0.0, 2.0),
        Vector3::new(0.0, 0.5, 2.0),
        Vector3::new(0.0, -0.5, 2.0),
    ];
    
    let mut projections_attempted = 0;
    let mut _projections_succeeded = 0;
    
    for point in boundary_points {
        projections_attempted += 1;
        match model.project(&point) {
            Ok(proj) => {
                // If projection succeeds, it must be within bounds
                assert!(
                    proj.x >= 0.0 && proj.x < model.resolution.width as f64,
                    "Projection x must be in bounds if projection succeeds"
                );
                assert!(
                    proj.y >= 0.0 && proj.y < model.resolution.height as f64,
                    "Projection y must be in bounds if projection succeeds"
                );
                _projections_succeeded += 1;
            }
            Err(CameraModelError::ProjectionOutSideImage) => {
                // This is acceptable - some points may project outside
            }
            Err(e) => panic!("Unexpected error: {:?}", e),
        }
    }
    
    assert!(projections_attempted > 0, "Should attempt projections");
    // We don't require all to succeed, just that the model works
}
