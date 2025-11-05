//! Image Undistortion Tool
//!
//! Undistort images using camera calibration files.
//!
//! Usage:
//!   cargo run --bin image_undistort -- -i image.jpg -c camera.yaml -o output.jpg

use apex_camera_models::camera::*;
use apex_camera_models::util::{undistort_image, InterpolationMethod};
use clap::Parser;
use std::path::PathBuf;

#[derive(Parser)]
#[command(author, version, about = "Undistort images using camera calibration")]
struct Cli {
    /// Input image path
    #[arg(short = 'i', long)]
    input: PathBuf,

    /// Camera calibration YAML file
    #[arg(short = 'c', long)]
    calib: PathBuf,

    /// Output image path
    #[arg(short = 'o', long)]
    output: PathBuf,

    /// Camera model type (fov, ds, kb, radtan, ucm, eucm, pinhole)
    #[arg(short = 'm', long, default_value = "fov")]
    model: String,

    /// Target focal length X (optional, defaults to source fx)
    #[arg(long)]
    target_fx: Option<f64>,

    /// Target focal length Y (optional, defaults to source fy)
    #[arg(long)]
    target_fy: Option<f64>,
}

fn main() -> Result<(), Box<dyn std::error::Error>> {
    let cli = Cli::parse();

    println!("🎯 Image Undistortion Tool");
    println!("==========================");
    println!("Input: {:?}", cli.input);
    println!("Calibration: {:?}", cli.calib);
    println!("Output: {:?}", cli.output);
    println!();

    // Load camera model
    let calib_path = cli.calib.to_str().unwrap();
    let model: Box<dyn CameraModel> = match cli.model.to_lowercase().as_str() {
        "fov" => Box::new(FovModel::load_from_yaml(calib_path)?),
        "ds" | "double_sphere" => Box::new(DoubleSphereModel::load_from_yaml(calib_path)?),
        "kb" | "kannala_brandt" => Box::new(KannalaBrandtModel::load_from_yaml(calib_path)?),
        "radtan" | "rad_tan" => Box::new(RadTanModel::load_from_yaml(calib_path)?),
        "ucm" => Box::new(UcmModel::load_from_yaml(calib_path)?),
        "eucm" => Box::new(EucmModel::load_from_yaml(calib_path)?),
        "pinhole" => Box::new(PinholeModel::load_from_yaml(calib_path)?),
        _ => return Err(format!("Unsupported model: {}", cli.model).into()),
    };

    println!("✓ Loaded {} camera model", cli.model);
    let intrinsics = model.get_intrinsics();
    println!("  fx={:.2}, fy={:.2}, cx={:.2}, cy={:.2}",
             intrinsics.fx, intrinsics.fy, intrinsics.cx, intrinsics.cy);
    
    let resolution = model.get_resolution();
    println!("  Resolution: {}x{}", resolution.width, resolution.height);
    println!();

    // Load input image
    let img = image::open(&cli.input)?.to_rgb8();
    println!("✓ Loaded input image: {}x{}", img.width(), img.height());

    // Prepare target intrinsics
    let target_intrinsics = if cli.target_fx.is_some() || cli.target_fy.is_some() {
        let mut target = intrinsics.clone();
        if let Some(fx) = cli.target_fx {
            target.fx = fx;
        }
        if let Some(fy) = cli.target_fy {
            target.fy = fy;
        }
        println!("✓ Using custom target focal lengths: fx={:.2}, fy={:.2}",
                 target.fx, target.fy);
        Some(target)
    } else {
        None
    };

    // Undistort image
    println!("⏳ Undistorting image...");
    let undistorted = undistort_image(
        &img,
        &*model,
        target_intrinsics,
        InterpolationMethod::Bilinear,
    )?;

    // Save output
    undistorted.save(&cli.output)?;
    println!("✓ Saved undistorted image to: {:?}", cli.output);
    println!();
    println!("✅ Done!");

    Ok(())
}
