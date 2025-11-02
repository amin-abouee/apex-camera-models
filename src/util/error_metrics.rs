//! Error metrics and reprojection error computation.
//!
//! This module provides functionality for computing and representing projection errors
//! in camera models, including statistical measures like RMSE, mean, and median errors.

use crate::camera::CameraModel;
use nalgebra::{Matrix2xX, Matrix3xX};
use serde::{Deserialize, Serialize};
use std::fmt;

use super::UtilError;

/// Projection error statistics for camera model evaluation.
///
/// Contains various statistical measures of reprojection errors,
/// useful for assessing camera model accuracy and conversion quality.
#[derive(Clone, Serialize, Deserialize)]
pub struct ProjectionError {
    /// Root Mean Square Error of reprojection
    pub rmse: f64,
    /// Minimum reprojection error
    pub min: f64,
    /// Maximum reprojection error
    pub max: f64,
    /// Mean reprojection error
    pub mean: f64,
    /// Standard deviation of reprojection errors
    pub stddev: f64,
    /// Median reprojection error
    pub median: f64,
}

impl fmt::Debug for ProjectionError {
    fn fmt(&self, f: &mut fmt::Formatter<'_>) -> fmt::Result {
        write!(
            f,
            "Projection Error [ rmse: {}, min: {}, max: {}, mean: {}, stddev: {}, median: {} ]",
            self.rmse, self.min, self.max, self.mean, self.stddev, self.median
        )
    }
}

/// Compute reprojection error statistics for a camera model.
///
/// Given a set of 3D points and their corresponding 2D projections,
/// this function computes how accurately the camera model can project
/// the 3D points back to the 2D locations.
///
/// # Arguments
///
/// * `camera_model` - The camera model to evaluate
/// * `points3d` - Matrix of 3D points (3×N)
/// * `points2d` - Matrix of corresponding 2D points (2×N)
///
/// # Returns
///
/// * `Result<ProjectionError, UtilError>` - Statistical measures of projection errors
///
/// # Errors
///
/// * `UtilError::ZeroProjectionPoints` - If no valid projections could be computed
pub fn compute_reprojection_error<T>(
    camera_model: Option<&T>,
    points3d: &Matrix3xX<f64>,
    points2d: &Matrix2xX<f64>,
) -> Result<ProjectionError, UtilError>
where
    T: ?Sized + CameraModel,
{
    let camera_model = camera_model.unwrap();
    let mut errors = vec![];
    for i in 0..points3d.ncols() {
        let point3d = points3d.column(i).into_owned();
        let point2d = points2d.column(i).into_owned();

        if let Ok(point2d_projected) = camera_model.project(&point3d) {
            let reprojection_error = (point2d_projected - point2d).norm();
            errors.push(reprojection_error);
        }
    }

    if errors.is_empty() {
        return Err(UtilError::ZeroProjectionPoints);
    }

    // Calculate statistics
    let n = errors.len() as f64;
    let sum: f64 = errors.iter().sum::<f64>();
    let mean = sum / n;

    // Calculate variance and standard deviation
    let variance: f64 = errors.iter().map(|x| (x - mean).powi(2)).sum::<f64>() / n;
    let stddev = variance.sqrt();

    // Calculate RMSE
    let sum_squared: f64 = errors.iter().map(|x| x.powi(2)).sum::<f64>();
    let rmse = (sum_squared / n).sqrt();

    // Find min and max
    let min = errors.iter().fold(f64::INFINITY, |a, &b| a.min(b));
    let max = errors.iter().fold(f64::NEG_INFINITY, |a, &b| a.max(b));

    // Calculate median
    let mut sorted_errors = errors.clone();
    sorted_errors.sort_by(|a, b| a.partial_cmp(b).unwrap());
    let median = if sorted_errors.len() % 2 == 0 {
        let mid = sorted_errors.len() / 2;
        (sorted_errors[mid - 1] + sorted_errors[mid]) / 2.0
    } else {
        sorted_errors[sorted_errors.len() / 2]
    };

    Ok(ProjectionError {
        rmse,
        min,
        max,
        mean,
        stddev,
        median,
    })
}
