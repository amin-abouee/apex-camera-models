//! Projection factors for apex-solver optimization.
//!
//! This module provides factor implementations for various camera models
//! that compute reprojection errors and analytical Jacobians. These factors
//! are designed to work with the apex-solver framework's Levenberg-Marquardt
//! optimizer using hand-derived analytical derivatives.
//!
//! Each camera model has a corresponding projection factor that:
//! - Stores 3D-2D point correspondences as Matrix3xX and Matrix2xX
//! - Implements the `Factor` trait from apex-solver
//! - Computes residuals and analytical Jacobians efficiently
//!
//! # Available Factors
//!
//! - [`DoubleSphereProjectionFactor`]: Double Sphere camera model (6 parameters)
//! - [`KannalaBrandtProjectionFactor`]: Kannala-Brandt fisheye model (8 parameters)
//! - [`RadTanProjectionFactor`]: Radial-Tangential distortion model (9 parameters)
//! - [`UcmProjectionFactor`]: Unified Camera Model (5 parameters)
//! - [`EucmProjectionFactor`]: Extended Unified Camera Model (6 parameters)

pub mod double_sphere_factor;
pub mod eucm_factor;
pub mod kannala_brandt_factor;
pub mod rad_tan_factor;
pub mod ucm_factor;

pub use double_sphere_factor::DoubleSphereProjectionFactor;
pub use eucm_factor::EucmProjectionFactor;
pub use kannala_brandt_factor::KannalaBrandtProjectionFactor;
pub use rad_tan_factor::RadTanProjectionFactor;
pub use ucm_factor::UcmProjectionFactor;
