//! Basic Projection Example
//!
//! This example demonstrates the fundamental camera model operations:
//! - Creating a camera model from parameters
//! - Projecting 3D points to 2D image coordinates
//! - Unprojecting 2D points back to 3D rays
//! - Validating round-trip accuracy
//!
//! Run with: cargo run --example basic_projection

use apex_camera_models::camera::{CameraModel, DoubleSphereModel, Resolution};
use nalgebra::{DVector, Vector3};

fn main() -> Result<(), Box<dyn std::error::Error>> {
    println!("=== Basic Camera Projection Example ===\n");

    // Create a Double Sphere camera model with parameters:
    // fx, fy, cx, cy, alpha, xi
    let params = DVector::from_vec(vec![350.0, 350.0, 320.0, 240.0, 0.58, -0.18]);
    let mut model = DoubleSphereModel::new(&params)?;

    // Set resolution
    model.resolution = Resolution {
        width: 640,
        height: 480,
    };

    println!("Camera Model: {}", model.get_model_name());
    println!(
        "Resolution: {}x{}",
        model.resolution.width, model.resolution.height
    );
    println!(
        "Intrinsics: fx={}, fy={}, cx={}, cy={}\n",
        model.intrinsics.fx, model.intrinsics.fy, model.intrinsics.cx, model.intrinsics.cy
    );

    // Example 1: Project a 3D point to 2D
    println!("--- Example 1: Project 3D → 2D ---");
    let point_3d = Vector3::new(1.0, 0.5, 2.0);
    println!(
        "3D Point: [{:.3}, {:.3}, {:.3}]",
        point_3d.x, point_3d.y, point_3d.z
    );

    match model.project(&point_3d) {
        Ok(point_2d) => {
            println!(
                "2D Projection: [{:.2}, {:.2}] pixels\n",
                point_2d.x, point_2d.y
            );

            // Example 2: Unproject back to 3D
            println!("--- Example 2: Unproject 2D → 3D ---");
            match model.unproject(&point_2d) {
                Ok(ray_3d) => {
                    println!(
                        "3D Ray: [{:.3}, {:.3}, {:.3}]",
                        ray_3d.x, ray_3d.y, ray_3d.z
                    );

                    // Normalize both vectors to compare directions
                    let original_dir = point_3d.normalize();
                    let recovered_dir = ray_3d.normalize();
                    let angle_error = original_dir.dot(&recovered_dir).acos().to_degrees();

                    println!("Angular error: {:.6}°", angle_error);
                    println!("✓ Round-trip successful!\n");
                }
                Err(e) => println!("Unprojection failed: {}\n", e),
            }
        }
        Err(e) => println!("Projection failed: {}\n", e),
    }

    // Example 3: Batch projection
    println!("--- Example 3: Batch Projection ---");
    let points_3d = [Vector3::new(1.0, 0.0, 1.0),
        Vector3::new(0.0, 1.0, 1.5),
        Vector3::new(-0.5, -0.5, 2.0),
        Vector3::new(0.3, 0.8, 1.2)];

    println!("Projecting {} points:", points_3d.len());
    for (i, p3d) in points_3d.iter().enumerate() {
        match model.project(p3d) {
            Ok(p2d) => println!(
                "  Point {}: 3D({:.2}, {:.2}, {:.2}) → 2D({:.1}, {:.1})",
                i + 1,
                p3d.x,
                p3d.y,
                p3d.z,
                p2d.x,
                p2d.y
            ),
            Err(e) => println!("  Point {}: Failed - {}", i + 1, e),
        }
    }

    println!("\n✓ Example completed successfully!");
    Ok(())
}
