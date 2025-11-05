//! Apex Camera Models Library
//!
//! A comprehensive Rust library for fisheye camera model conversions and calibration.
//! This library provides implementations of various camera models including:
//! - Pinhole camera model
//! - Radial-Tangential distortion model
//! - Unified Camera Model (UCM)
//! - Extended Unified Camera Model (EUCM)
//! - Double Sphere camera model
//! - Kannala-Brandt camera model
//! - Field-of-View (FOV) camera model
//!
//! The library provides camera models for projection and unprojection operations.
//! For optimization and calibration factors, see the apex-solver crate which now
//! contains all projection factors for camera calibration.

pub mod camera;
pub mod util;

// Re-export commonly used types
pub use camera::{
    CameraModel, CameraModelError, DoubleSphereModel, EucmModel, FovModel, Intrinsics,
    KannalaBrandtModel, PinholeModel, RadTanModel, Resolution, UcmModel,
};

// Note: Camera projection factors have been moved to apex-solver crate
// Use apex_solver::factors::{DoubleSphereProjectionFactor, EucmProjectionFactor, ...}
