//! Integration tests for camera model conversions

use apex_camera_models::camera::{
    CameraModel, DoubleSphereModel, EucmModel, KannalaBrandtModel,
    PinholeModel, RadTanModel, UcmModel,
};
use nalgebra::{DVector, Vector3};

fn generate_test_points() -> Vec<Vector3<f64>> {
    vec![
        Vector3::new(0.1, 0.1, 1.0),
        Vector3::new(0.3, 0.0, 1.5),
        Vector3::new(-0.2, 0.3, 2.0),
        Vector3::new(-0.3, -0.2, 1.8),
        Vector3::new(0.15, -0.25, 2.5),
    ]
}

#[test]
fn test_double_sphere_basic_operations() {
    let model = DoubleSphereModel::load_from_yaml("samples/double_sphere.yaml")
        .expect("Failed to load DoubleSphere model");
    
    let mut successful_projections = 0;
    for point_3d in generate_test_points().iter() {
        if let Ok(point_2d) = model.project(point_3d) {
            assert!(point_2d.x >= 0.0 && point_2d.x < model.resolution.width as f64);
            assert!(point_2d.y >= 0.0 && point_2d.y < model.resolution.height as f64);
            
            if let Ok(ray) = model.unproject(&point_2d) {
                let dot = point_3d.normalize().dot(&ray);
                assert!(dot > 0.99);
                successful_projections += 1;
            }
        }
    }
    assert!(successful_projections > 0, "At least one projection should succeed");
}

#[test]
fn test_kannala_brandt_basic_operations() {
    let model = KannalaBrandtModel::load_from_yaml("samples/kannala_brandt.yaml")
        .expect("Failed to load KannalaBrandt model");
    
    let mut successful_projections = 0;
    for point_3d in generate_test_points().iter() {
        if let Ok(point_2d) = model.project(point_3d) {
            assert!(point_2d.x >= 0.0 && point_2d.x < model.resolution.width as f64);
            assert!(point_2d.y >= 0.0 && point_2d.y < model.resolution.height as f64);
            
            if let Ok(ray) = model.unproject(&point_2d) {
                let dot = point_3d.normalize().dot(&ray);
                assert!(dot > 0.99);
                successful_projections += 1;
            }
        }
    }
    assert!(successful_projections > 0, "At least one projection should succeed");
}

#[test]
fn test_rad_tan_basic_operations() {
    let model = RadTanModel::load_from_yaml("samples/rad_tan.yaml")
        .expect("Failed to load RadTan model");
    
    let mut successful_projections = 0;
    for point_3d in generate_test_points().iter() {
        if let Ok(point_2d) = model.project(point_3d) {
            assert!(point_2d.x >= 0.0 && point_2d.x < model.resolution.width as f64);
            assert!(point_2d.y >= 0.0 && point_2d.y < model.resolution.height as f64);
            
            if let Ok(ray) = model.unproject(&point_2d) {
                let dot = point_3d.normalize().dot(&ray);
                assert!(dot > 0.99);
                successful_projections += 1;
            }
        }
    }
    assert!(successful_projections > 0, "At least one projection should succeed");
}

#[test]
fn test_ucm_basic_operations() {
    let model = UcmModel::load_from_yaml("samples/ucm.yaml")
        .expect("Failed to load UCM model");
    
    // UCM model has principal point outside image, so we just verify the model loads
    // and can project points (some may be outside the image bounds)
    let mut total_projections = 0;
    let mut _valid_projections = 0;
    
    for point_3d in generate_test_points().iter() {
        if let Ok(point_2d) = model.project(point_3d) {
            total_projections += 1;
            // Check if within bounds (may not be due to principal point location)
            if point_2d.x >= 0.0 && point_2d.x < model.resolution.width as f64
                && point_2d.y >= 0.0 && point_2d.y < model.resolution.height as f64 {
                _valid_projections += 1;
                
                if let Ok(ray) = model.unproject(&point_2d) {
                    let dot = point_3d.normalize().dot(&ray);
                    assert!(dot > 0.99, "Direction should be preserved");
                }
            }
        }
    }
    assert!(total_projections > 0, "Model should be able to project points");
}

#[test]
fn test_eucm_basic_operations() {
    let model = EucmModel::load_from_yaml("samples/eucm.yaml")
        .expect("Failed to load EUCM model");
    
    // EUCM model has principal point outside image, so we just verify the model loads
    // and can project points (some may be outside the image bounds)
    let mut total_projections = 0;
    let mut _valid_projections = 0;
    
    for point_3d in generate_test_points().iter() {
        if let Ok(point_2d) = model.project(point_3d) {
            total_projections += 1;
            // Check if within bounds (may not be due to principal point location)
            if point_2d.x >= 0.0 && point_2d.x < model.resolution.width as f64
                && point_2d.y >= 0.0 && point_2d.y < model.resolution.height as f64 {
                _valid_projections += 1;
                
                if let Ok(ray) = model.unproject(&point_2d) {
                    let dot = point_3d.normalize().dot(&ray);
                    assert!(dot > 0.99, "Direction should be preserved");
                }
            }
        }
    }
    assert!(total_projections > 0, "Model should be able to project points");
}

#[test]
fn test_pinhole_basic_operations() {
    let params = DVector::from_vec(vec![500.0, 500.0, 320.0, 240.0]);
    let mut model = PinholeModel::new(&params).expect("Failed to create Pinhole model");
    model.resolution.width = 640;
    model.resolution.height = 480;
    
    let mut successful_projections = 0;
    for point_3d in generate_test_points().iter() {
        if let Ok(point_2d) = model.project(point_3d) {
            assert!(point_2d.x >= 0.0 && point_2d.x < model.resolution.width as f64);
            assert!(point_2d.y >= 0.0 && point_2d.y < model.resolution.height as f64);
            
            if let Ok(ray) = model.unproject(&point_2d) {
                let dot = point_3d.normalize().dot(&ray);
                assert!(dot > 0.9999);
                successful_projections += 1;
            }
        }
    }
    assert!(successful_projections > 0, "At least one projection should succeed");
}

#[test]
fn test_parameter_bounds_checking() {
    assert!(DoubleSphereModel::new(&DVector::from_vec(vec![500.0, 500.0])).is_err());
    assert!(KannalaBrandtModel::new(&DVector::from_vec(vec![500.0])).is_err());
    assert!(RadTanModel::new(&DVector::from_vec(vec![500.0, 500.0])).is_err());
    assert!(UcmModel::new(&DVector::from_vec(vec![500.0])).is_err());
    assert!(EucmModel::new(&DVector::from_vec(vec![500.0])).is_err());
    assert!(PinholeModel::new(&DVector::from_vec(vec![500.0])).is_err());
}

#[test]
fn test_invalid_intrinsics_rejection() {
    // Negative focal length
    assert!(PinholeModel::new(&DVector::from_vec(vec![-500.0, 500.0, 320.0, 240.0])).is_err());
    
    // Zero focal length
    assert!(PinholeModel::new(&DVector::from_vec(vec![0.0, 500.0, 320.0, 240.0])).is_err());
    
    // Infinite principal point
    assert!(PinholeModel::new(&DVector::from_vec(vec![500.0, 500.0, f64::INFINITY, 240.0])).is_err());
    
    // NaN principal point
    assert!(PinholeModel::new(&DVector::from_vec(vec![500.0, 500.0, 320.0, f64::NAN])).is_err());
}
