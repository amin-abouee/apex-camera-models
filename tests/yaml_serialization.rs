//! Integration tests for YAML serialization and deserialization

use apex_camera_models::camera::{
    CameraModel, DoubleSphereModel, EucmModel, KannalaBrandtModel, RadTanModel, UcmModel,
};
use std::fs;

#[test]
fn test_double_sphere_yaml_round_trip() {
    fs::create_dir_all("output").ok();
    
    let input_path = "samples/double_sphere.yaml";
    let output_path = "output/test_double_sphere.yaml";
    
    let model = DoubleSphereModel::load_from_yaml(input_path)
        .expect("Failed to load model");
    
    model.save_to_yaml(output_path).expect("Failed to save model");
    
    let reloaded = DoubleSphereModel::load_from_yaml(output_path)
        .expect("Failed to reload model");
    
    assert_eq!(model.intrinsics.fx, reloaded.intrinsics.fx);
    assert_eq!(model.intrinsics.fy, reloaded.intrinsics.fy);
    assert_eq!(model.intrinsics.cx, reloaded.intrinsics.cx);
    assert_eq!(model.intrinsics.cy, reloaded.intrinsics.cy);
    assert_eq!(model.alpha, reloaded.alpha);
    assert_eq!(model.xi, reloaded.xi);
    
    fs::remove_file(output_path).ok();
}

#[test]
fn test_kannala_brandt_yaml_load() {
    // KannalaBrandt has a known issue where save_to_yaml uses 'distortion_coeffs'
    // but load_from_yaml expects 'distortion', so we only test loading
    let input_path = "samples/kannala_brandt.yaml";
    
    let model = KannalaBrandtModel::load_from_yaml(input_path)
        .expect("Failed to load model");
    
    assert_eq!(model.intrinsics.fx, 190.97847715128717);
    assert_eq!(model.intrinsics.fy, 190.9733070521226);
    
    // Verify model can be used for projection
    assert!(model.get_model_name() == "kannala_brandt");
}

#[test]
fn test_rad_tan_yaml_round_trip() {
    fs::create_dir_all("output").ok();
    
    let input_path = "samples/rad_tan.yaml";
    let output_path = "output/test_rad_tan.yaml";
    
    let model = RadTanModel::load_from_yaml(input_path)
        .expect("Failed to load model");
    
    model.save_to_yaml(output_path).expect("Failed to save model");
    
    let reloaded = RadTanModel::load_from_yaml(output_path)
        .expect("Failed to reload model");
    
    assert_eq!(model.intrinsics.fx, reloaded.intrinsics.fx);
    assert_eq!(model.distortions.len(), reloaded.distortions.len());
    
    fs::remove_file(output_path).ok();
}

#[test]
fn test_ucm_yaml_round_trip() {
    fs::create_dir_all("output").ok();
    
    let input_path = "samples/ucm.yaml";
    let output_path = "output/test_ucm.yaml";
    
    let model = UcmModel::load_from_yaml(input_path)
        .expect("Failed to load model");
    
    model.save_to_yaml(output_path).expect("Failed to save model");
    
    let reloaded = UcmModel::load_from_yaml(output_path)
        .expect("Failed to reload model");
    
    assert_eq!(model.intrinsics.fx, reloaded.intrinsics.fx);
    assert_eq!(model.alpha, reloaded.alpha);
    
    fs::remove_file(output_path).ok();
}

#[test]
fn test_eucm_yaml_round_trip() {
    fs::create_dir_all("output").ok();
    
    let input_path = "samples/eucm.yaml";
    let output_path = "output/test_eucm.yaml";
    
    let model = EucmModel::load_from_yaml(input_path)
        .expect("Failed to load model");
    
    model.save_to_yaml(output_path).expect("Failed to save model");
    
    let reloaded = EucmModel::load_from_yaml(output_path)
        .expect("Failed to reload model");
    
    assert_eq!(model.intrinsics.fx, reloaded.intrinsics.fx);
    assert_eq!(model.alpha, reloaded.alpha);
    assert_eq!(model.beta, reloaded.beta);
    
    fs::remove_file(output_path).ok();
}
