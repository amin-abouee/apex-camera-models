//! Parameter Validation Example
//!
//! This example demonstrates:
//! - Creating camera models with different parameter sets
//! - Validating parameters to catch errors early
//! - Handling validation errors gracefully
//! - Examples for all 6 camera model types
//!
//! Run with: cargo run --example parameter_validation

use apex_camera_models::camera::*;
use nalgebra::DVector;

fn main() {
    println!("=== Camera Model Parameter Validation Example ===\n");

    // Example 1: Valid Pinhole Model
    println!("--- 1. Pinhole Model (Valid) ---");
    let pinhole_params = DVector::from_vec(vec![500.0, 500.0, 320.0, 240.0]);
    match PinholeModel::new(&pinhole_params) {
        Ok(mut model) => {
            model.resolution = Resolution {
                width: 640,
                height: 480,
            };
            println!("✓ Created pinhole model successfully");
            println!(
                "  Parameters: fx={}, fy={}, cx={}, cy={}",
                model.intrinsics.fx, model.intrinsics.fy, model.intrinsics.cx, model.intrinsics.cy
            );
        }
        Err(e) => println!("✗ Failed: {}", e),
    }

    // Example 2: Invalid Pinhole Model (negative focal length)
    println!("\n--- 2. Pinhole Model (Invalid - negative fx) ---");
    let bad_params = DVector::from_vec(vec![-500.0, 500.0, 320.0, 240.0]);
    match PinholeModel::new(&bad_params) {
        Ok(_) => println!("✗ Should have failed validation!"),
        Err(e) => println!("✓ Caught error as expected: {}", e),
    }

    // Example 3: Double Sphere Model
    println!("\n--- 3. Double Sphere Model (Valid) ---");
    let ds_params = DVector::from_vec(vec![350.0, 350.0, 320.0, 240.0, 0.58, -0.18]);
    match DoubleSphereModel::new(&ds_params) {
        Ok(mut model) => {
            model.resolution = Resolution {
                width: 640,
                height: 480,
            };
            println!("✓ Created double sphere model successfully");
            println!("  Distortion: alpha={}, xi={}", model.alpha, model.xi);
        }
        Err(e) => println!("✗ Failed: {}", e),
    }

    // Example 4: Invalid Double Sphere (alpha out of range)
    println!("\n--- 4. Double Sphere Model (Invalid - alpha > 1) ---");
    let bad_ds_params = DVector::from_vec(vec![350.0, 350.0, 320.0, 240.0, 1.5, -0.18]);
    match DoubleSphereModel::new(&bad_ds_params) {
        Ok(_) => println!("✗ Should have failed validation!"),
        Err(e) => println!("✓ Caught error as expected: {}", e),
    }

    // Example 5: UCM Model
    println!("\n--- 5. Unified Camera Model (Valid) ---");
    let ucm_params = DVector::from_vec(vec![350.0, 350.0, 320.0, 240.0, 0.8]);
    match UcmModel::new(&ucm_params) {
        Ok(mut model) => {
            model.resolution = Resolution {
                width: 640,
                height: 480,
            };
            println!("✓ Created UCM model successfully");
            println!("  Distortion: alpha={}", model.alpha);
        }
        Err(e) => println!("✗ Failed: {}", e),
    }

    // Example 6: EUCM Model
    println!("\n--- 6. Extended Unified Camera Model (Valid) ---");
    let eucm_params = DVector::from_vec(vec![350.0, 350.0, 320.0, 240.0, 1.0, 0.5]);
    match EucmModel::new(&eucm_params) {
        Ok(mut model) => {
            model.resolution = Resolution {
                width: 640,
                height: 480,
            };
            println!("✓ Created EUCM model successfully");
            println!("  Distortion: alpha={}, beta={}", model.alpha, model.beta);
        }
        Err(e) => println!("✗ Failed: {}", e),
    }

    // Example 7: Kannala-Brandt Model
    println!("\n--- 7. Kannala-Brandt Model (Valid) ---");
    let kb_params = DVector::from_vec(vec![460.0, 460.0, 320.0, 240.0, -0.01, 0.05, -0.08, 0.04]);
    match KannalaBrandtModel::new(&kb_params) {
        Ok(mut model) => {
            model.resolution = Resolution {
                width: 640,
                height: 480,
            };
            println!("✓ Created Kannala-Brandt model successfully");
            let dist = model.get_distortion();
            println!(
                "  Distortion: k1={}, k2={}, k3={}, k4={}",
                dist[0], dist[1], dist[2], dist[3]
            );
        }
        Err(e) => println!("✗ Failed: {}", e),
    }

    // Example 8: RadTan Model
    println!("\n--- 8. Radial-Tangential Model (Valid) ---");
    let radtan_params = DVector::from_vec(vec![
        460.0, 460.0, 320.0, 240.0, // intrinsics
        -0.28, 0.07, 0.0002, 0.00002, 0.0, // k1, k2, p1, p2, k3
    ]);
    match RadTanModel::new(&radtan_params) {
        Ok(mut model) => {
            model.resolution = Resolution {
                width: 640,
                height: 480,
            };
            println!("✓ Created RadTan model successfully");
            let dist = model.get_distortion();
            println!(
                "  Distortion: k1={}, k2={}, p1={}, p2={}, k3={}",
                dist[0], dist[1], dist[2], dist[3], dist[4]
            );
        }
        Err(e) => println!("✗ Failed: {}", e),
    }

    println!("\n✓ Parameter validation examples completed!");
}
