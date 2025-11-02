//! Model Conversion Example
//!
//! This example demonstrates how to convert between different camera models.
//! It shows:
//! - Converting points from one camera model to another
//! - Handling projection/unprojection workflows
//! - Working with different camera model types
//! - Error handling during conversion

use apex_camera_models::camera::{
    CameraModel, DoubleSphereModel, EucmModel, PinholeModel, Resolution, UcmModel,
};
use nalgebra::{DVector, Vector2, Vector3};

fn main() -> Result<(), Box<dyn std::error::Error>> {
    println!("=== Camera Model Conversion Example ===\n");

    // Create a DoubleSphere camera model
    // Parameters: fx, fy, cx, cy, alpha, xi
    let ds_params = DVector::from_vec(vec![600.0, 600.0, 960.0, 540.0, -0.2, 0.6]);
    let mut double_sphere = DoubleSphereModel::new(&ds_params)?;
    double_sphere.resolution = Resolution {
        width: 1920,
        height: 1080,
    };
    println!("Created DoubleSphere model:");
    println!(
        "  fx={}, fy={}, cx={}, cy={}",
        double_sphere.intrinsics.fx,
        double_sphere.intrinsics.fy,
        double_sphere.intrinsics.cx,
        double_sphere.intrinsics.cy
    );
    println!("  alpha={}, xi={}\n", double_sphere.alpha, double_sphere.xi);

    // Create a Pinhole camera model with similar intrinsics
    // Parameters: fx, fy, cx, cy
    let pinhole_params = DVector::from_vec(vec![600.0, 600.0, 960.0, 540.0]);
    let mut pinhole = PinholeModel::new(&pinhole_params)?;
    pinhole.resolution = Resolution {
        width: 1920,
        height: 1080,
    };
    println!("Created Pinhole model:");
    println!(
        "  fx={}, fy={}, cx={}, cy={}\n",
        pinhole.intrinsics.fx, pinhole.intrinsics.fy, pinhole.intrinsics.cx, pinhole.intrinsics.cy
    );

    // Create UCM and EUCM models
    // UCM Parameters: fx, fy, cx, cy, alpha
    let ucm_params = DVector::from_vec(vec![600.0, 600.0, 960.0, 540.0, 0.5]);
    let mut ucm = UcmModel::new(&ucm_params)?;
    ucm.resolution = Resolution {
        width: 1920,
        height: 1080,
    };

    // EUCM Parameters: fx, fy, cx, cy, alpha, beta
    let eucm_params = DVector::from_vec(vec![600.0, 600.0, 960.0, 540.0, 0.5, 1.0]);
    let mut eucm = EucmModel::new(&eucm_params)?;
    eucm.resolution = Resolution {
        width: 1920,
        height: 1080,
    };

    println!("Created UCM model (alpha={})", ucm.alpha);
    println!(
        "Created EUCM model (alpha={}, beta={})\n",
        eucm.alpha, eucm.beta
    );

    // Example 1: Convert a 3D point through different models
    println!("--- Example 1: Same 3D point through different models ---");
    let point_3d = Vector3::new(0.5, 0.3, 2.0);
    println!("3D point: [{}, {}, {}]", point_3d.x, point_3d.y, point_3d.z);

    let ds_pixel = double_sphere.project(&point_3d)?;
    println!(
        "  DoubleSphere projection: [{:.2}, {:.2}]",
        ds_pixel.x, ds_pixel.y
    );

    let pinhole_pixel = pinhole.project(&point_3d)?;
    println!(
        "  Pinhole projection:      [{:.2}, {:.2}]",
        pinhole_pixel.x, pinhole_pixel.y
    );

    let ucm_pixel = ucm.project(&point_3d)?;
    println!(
        "  UCM projection:          [{:.2}, {:.2}]",
        ucm_pixel.x, ucm_pixel.y
    );

    let eucm_pixel = eucm.project(&point_3d)?;
    println!(
        "  EUCM projection:         [{:.2}, {:.2}]",
        eucm_pixel.x, eucm_pixel.y
    );

    println!("\nNote: Different models produce different pixel coordinates for the same 3D point");
    println!("due to their different distortion characteristics.\n");

    // Example 2: Convert pixel from DoubleSphere to Pinhole
    println!("--- Example 2: Convert pixel coordinates between models ---");
    let ds_pixel_coords = Vector2::new(800.0, 600.0);
    println!(
        "DoubleSphere pixel: [{}, {}]",
        ds_pixel_coords.x, ds_pixel_coords.y
    );

    // Step 1: Unproject from DoubleSphere to 3D (at arbitrary depth)
    let bearing = double_sphere.unproject(&ds_pixel_coords)?;
    println!(
        "  Unprojected to bearing vector: [{:.4}, {:.4}, {:.4}]",
        bearing.x, bearing.y, bearing.z
    );

    // Step 2: Scale to desired depth (e.g., 3.0 meters)
    let depth = 3.0;
    let point_3d_scaled = bearing * depth;
    println!(
        "  Scaled to depth {}: [{:.4}, {:.4}, {:.4}]",
        depth, point_3d_scaled.x, point_3d_scaled.y, point_3d_scaled.z
    );

    // Step 3: Project into Pinhole model
    let pinhole_converted = pinhole.project(&point_3d_scaled)?;
    println!(
        "  Projected to Pinhole: [{:.2}, {:.2}]",
        pinhole_converted.x, pinhole_converted.y
    );

    println!("\nThis workflow: DoubleSphere pixel -> 3D point -> Pinhole pixel");
    println!("is useful for converting images between different camera models.\n");

    // Example 3: Batch conversion of points
    println!("--- Example 3: Batch point conversion ---");
    let test_points = [Vector3::new(0.1, 0.1, 1.5),
        Vector3::new(0.2, -0.1, 2.0),
        Vector3::new(-0.3, 0.2, 2.5),
        Vector3::new(0.0, 0.0, 3.0)];

    println!(
        "Converting {} points from DoubleSphere to UCM:",
        test_points.len()
    );
    for (i, point) in test_points.iter().enumerate() {
        match double_sphere.project(point) {
            Ok(ds_px) => {
                // Unproject from DoubleSphere
                if let Ok(bearing) = double_sphere.unproject(&ds_px) {
                    // Scale to original depth
                    let depth = point.norm();
                    let scaled = bearing * depth;

                    // Project to UCM
                    match ucm.project(&scaled) {
                        Ok(ucm_px) => {
                            println!("  Point {}: 3D[{:.2}, {:.2}, {:.2}] -> DS[{:.1}, {:.1}] -> UCM[{:.1}, {:.1}]",
                                i, point.x, point.y, point.z, ds_px.x, ds_px.y, ucm_px.x, ucm_px.y);
                        }
                        Err(e) => println!("  Point {}: UCM projection failed: {}", i, e),
                    }
                }
            }
            Err(e) => println!("  Point {}: DoubleSphere projection failed: {}", i, e),
        }
    }

    // Example 4: Field of view comparison
    println!("\n--- Example 4: Field of view comparison ---");
    println!("Testing edge points to understand model coverage:");

    let edge_points = vec![
        (Vector3::new(0.0, 0.0, 1.0), "Center"),
        (Vector3::new(0.5, 0.0, 1.0), "Right"),
        (Vector3::new(0.0, 0.5, 1.0), "Up"),
        (Vector3::new(1.0, 1.0, 1.0), "Diagonal"),
        (Vector3::new(1.5, 1.5, 1.0), "Far diagonal"),
    ];

    for (point, label) in edge_points {
        print!(
            "  {:<15} 3D[{:.1}, {:.1}, {:.1}] -> ",
            label, point.x, point.y, point.z
        );

        let ds_result = double_sphere.project(&point);
        let ph_result = pinhole.project(&point);

        match (ds_result, ph_result) {
            (Ok(ds), Ok(ph)) => {
                println!("DS[{:.0}, {:.0}] PH[{:.0}, {:.0}]", ds.x, ds.y, ph.x, ph.y);
            }
            (Ok(ds), Err(_)) => {
                println!("DS[{:.0}, {:.0}] PH[out of bounds]", ds.x, ds.y);
            }
            (Err(_), Ok(ph)) => {
                println!("DS[out of bounds] PH[{:.0}, {:.0}]", ph.x, ph.y);
            }
            (Err(_), Err(_)) => {
                println!("Both out of bounds");
            }
        }
    }

    println!("\n=== Conversion complete! ===");
    Ok(())
}
