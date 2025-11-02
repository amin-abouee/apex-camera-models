//! Batch Processing Example
//!
//! This example demonstrates efficient batch processing of points using camera models.
//! It shows:
//! - Processing multiple 3D points in batch
//! - Handling errors gracefully in batch operations
//! - Performance considerations for large datasets
//! - Collecting statistics from batch operations

use apex_camera_models::camera::{CameraModel, DoubleSphereModel, PinholeModel, Resolution};
use nalgebra::{DVector, Vector2, Vector3};
use std::time::Instant;

fn main() -> Result<(), Box<dyn std::error::Error>> {
    println!("=== Batch Processing Example ===\n");

    // Setup camera model
    let params = DVector::from_vec(vec![600.0, 600.0, 960.0, 540.0, -0.2, 0.6]);
    let mut camera = DoubleSphereModel::new(&params)?;
    camera.resolution = Resolution {
        width: 1920,
        height: 1080,
    };
    println!("Created DoubleSphere camera model");
    println!(
        "Resolution: {}x{}\n",
        camera.resolution.width, camera.resolution.height
    );

    // Example 1: Generate a grid of 3D points
    println!("--- Example 1: Grid projection ---");
    let grid_points = generate_3d_grid(5, 5, 2.0);
    println!(
        "Generated {}x{} grid ({} points) at depth 2.0m",
        5,
        5,
        grid_points.len()
    );

    let start = Instant::now();
    let mut success_count = 0;
    let mut pixels = Vec::new();

    for point in &grid_points {
        match camera.project(point) {
            Ok(pixel) => {
                pixels.push(pixel);
                success_count += 1;
            }
            Err(_) => {
                // Point couldn't be projected (behind camera, etc.)
            }
        }
    }

    let duration = start.elapsed();
    println!(
        "Projected {} points successfully in {:.2}ms",
        success_count,
        duration.as_secs_f64() * 1000.0
    );
    println!(
        "Success rate: {:.1}%",
        (success_count as f64 / grid_points.len() as f64) * 100.0
    );

    // Show sample projections
    println!("\nSample projections:");
    for i in 0..5.min(pixels.len()) {
        println!(
            "  3D[{:.2}, {:.2}, {:.2}] -> 2D[{:.1}, {:.1}]",
            grid_points[i].x, grid_points[i].y, grid_points[i].z, pixels[i].x, pixels[i].y
        );
    }

    // Example 2: Batch unprojection
    println!("\n--- Example 2: Batch unprojection ---");
    let pixel_grid = generate_pixel_grid(
        camera.resolution.width as usize,
        camera.resolution.height as usize,
        10,
        10,
    );
    println!(
        "Generated {}x{} pixel grid ({} pixels)",
        10,
        10,
        pixel_grid.len()
    );

    let start = Instant::now();
    let mut bearings = Vec::new();

    for pixel in &pixel_grid {
        match camera.unproject(pixel) {
            Ok(bearing) => {
                bearings.push(bearing);
            }
            Err(_) => {
                // Pixel couldn't be unprojected (outside valid region, etc.)
            }
        }
    }

    let duration = start.elapsed();
    println!(
        "Unprojected {} pixels in {:.2}ms",
        bearings.len(),
        duration.as_secs_f64() * 1000.0
    );

    // Show sample unprojections
    println!("\nSample bearing vectors:");
    for i in 0..5.min(bearings.len()) {
        let b = &bearings[i];
        let norm = b.norm();
        println!(
            "  Pixel[{:.0}, {:.0}] -> Bearing[{:.4}, {:.4}, {:.4}] (norm={:.4})",
            pixel_grid[i].x, pixel_grid[i].y, b.x, b.y, b.z, norm
        );
    }

    // Example 3: Round-trip validation
    println!("\n--- Example 3: Round-trip validation ---");
    let test_points = generate_random_points(100, 1.0, 3.0);
    println!("Testing {} random 3D points", test_points.len());

    let start = Instant::now();
    let mut round_trip_errors = Vec::new();

    for point in &test_points {
        if let Ok(pixel) = camera.project(point) {
            if let Ok(bearing) = camera.unproject(&pixel) {
                // Scale bearing to original depth
                let depth = point.norm();
                let recovered = bearing * depth;

                // Calculate error
                let error = (point - recovered).norm();
                round_trip_errors.push(error);
            }
        }
    }

    let duration = start.elapsed();

    if !round_trip_errors.is_empty() {
        let max_error = round_trip_errors
            .iter()
            .cloned()
            .fold(f64::NEG_INFINITY, f64::max);
        let min_error = round_trip_errors
            .iter()
            .cloned()
            .fold(f64::INFINITY, f64::min);
        let avg_error = round_trip_errors.iter().sum::<f64>() / round_trip_errors.len() as f64;

        println!(
            "Round-trip validation completed in {:.2}ms",
            duration.as_secs_f64() * 1000.0
        );
        println!("Validated {} points successfully", round_trip_errors.len());
        println!("Error statistics:");
        println!("  Min error: {:.6} m", min_error);
        println!("  Max error: {:.6} m", max_error);
        println!("  Avg error: {:.6} m", avg_error);
    }

    // Example 4: Compare models on same dataset
    println!("\n--- Example 4: Model comparison ---");
    let pinhole_params = DVector::from_vec(vec![600.0, 600.0, 960.0, 540.0]);
    let mut pinhole = PinholeModel::new(&pinhole_params)?;
    pinhole.resolution = Resolution {
        width: 1920,
        height: 1080,
    };
    let comparison_points = generate_3d_grid(10, 10, 2.5);
    println!(
        "Comparing DoubleSphere vs Pinhole on {} points",
        comparison_points.len()
    );

    let start = Instant::now();
    let mut ds_pixels = Vec::new();
    for point in &comparison_points {
        if let Ok(pixel) = camera.project(point) {
            ds_pixels.push(pixel);
        }
    }
    let ds_duration = start.elapsed();

    let start = Instant::now();
    let mut ph_pixels = Vec::new();
    for point in &comparison_points {
        if let Ok(pixel) = pinhole.project(point) {
            ph_pixels.push(pixel);
        }
    }
    let ph_duration = start.elapsed();

    println!(
        "DoubleSphere: {} projections in {:.2}ms",
        ds_pixels.len(),
        ds_duration.as_secs_f64() * 1000.0
    );
    println!(
        "Pinhole:      {} projections in {:.2}ms",
        ph_pixels.len(),
        ph_duration.as_secs_f64() * 1000.0
    );

    // Calculate difference in pixel coordinates
    let mut differences = Vec::new();
    for i in 0..ds_pixels.len().min(ph_pixels.len()) {
        let diff = (ds_pixels[i] - ph_pixels[i]).norm();
        differences.push(diff);
    }

    if !differences.is_empty() {
        let max_diff = differences
            .iter()
            .cloned()
            .fold(f64::NEG_INFINITY, f64::max);
        let avg_diff = differences.iter().sum::<f64>() / differences.len() as f64;
        println!("Pixel coordinate differences:");
        println!("  Max: {:.2} pixels", max_diff);
        println!("  Avg: {:.2} pixels", avg_diff);
    }

    // Example 5: Filter points by image region
    println!("\n--- Example 5: Spatial filtering ---");
    let all_points = generate_random_points(200, 0.5, 4.0);
    println!("Processing {} random points", all_points.len());

    let mut center_region = 0;
    let mut edge_region = 0;
    let mut outside = 0;

    let center_threshold = 200.0; // pixels from center

    for point in &all_points {
        if let Ok(pixel) = camera.project(point) {
            let dx = pixel.x - camera.intrinsics.cx;
            let dy = pixel.y - camera.intrinsics.cy;
            let dist_from_center = (dx * dx + dy * dy).sqrt();

            if dist_from_center < center_threshold {
                center_region += 1;
            } else {
                edge_region += 1;
            }
        } else {
            outside += 1;
        }
    }

    println!("Spatial distribution:");
    println!(
        "  Center region (<{} px from center): {}",
        center_threshold, center_region
    );
    println!(
        "  Edge region (>{} px from center):   {}",
        center_threshold, edge_region
    );
    println!("  Outside image bounds:               {}", outside);

    println!("\n=== Batch processing complete! ===");
    Ok(())
}

// Helper function to generate a 3D grid
fn generate_3d_grid(rows: usize, cols: usize, depth: f64) -> Vec<Vector3<f64>> {
    let mut points = Vec::new();
    let x_step = 2.0 / rows as f64;
    let y_step = 2.0 / cols as f64;

    for i in 0..rows {
        for j in 0..cols {
            let x = -1.0 + i as f64 * x_step;
            let y = -1.0 + j as f64 * y_step;
            points.push(Vector3::new(x, y, depth));
        }
    }
    points
}

// Helper function to generate a pixel grid
fn generate_pixel_grid(width: usize, height: usize, rows: usize, cols: usize) -> Vec<Vector2<f64>> {
    let mut pixels = Vec::new();
    let x_step = width as f64 / (cols - 1) as f64;
    let y_step = height as f64 / (rows - 1) as f64;

    for i in 0..rows {
        for j in 0..cols {
            let x = j as f64 * x_step;
            let y = i as f64 * y_step;
            pixels.push(Vector2::new(x, y));
        }
    }
    pixels
}

// Helper function to generate random 3D points
fn generate_random_points(count: usize, min_depth: f64, max_depth: f64) -> Vec<Vector3<f64>> {
    use std::collections::hash_map::RandomState;
    use std::hash::BuildHasher;

    let mut points = Vec::new();
    let hasher = RandomState::new();

    for i in 0..count {
        // Simple pseudo-random generation using hash
        let h1_val = hasher.hash_one(i * 3);
        let h2_val = hasher.hash_one(i * 3 + 1);
        let h3_val = hasher.hash_one(i * 3 + 2);

        let x = ((h1_val % 1000) as f64 / 1000.0) * 2.0 - 1.0;
        let y = ((h2_val % 1000) as f64 / 1000.0) * 2.0 - 1.0;
        let z = min_depth + ((h3_val % 1000) as f64 / 1000.0) * (max_depth - min_depth);

        points.push(Vector3::new(x, y, z));
    }
    points
}
