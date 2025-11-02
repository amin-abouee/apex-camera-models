//! Camera model conversion validation utilities.
//!
//! This module provides functionality for validating the accuracy of camera model
//! conversions by testing projection consistency across different image regions.

use crate::camera::CameraModel;
use nalgebra::Vector2;
use serde::{Deserialize, Serialize};

use super::UtilError;

/// Validation results for conversion accuracy testing.
///
/// Contains error measurements for different regions of the image
/// (center, edges, etc.) to assess how well a converted camera model
/// matches the original across the entire field of view.
#[derive(Debug, Clone, Serialize, Deserialize)]
pub struct ValidationResults {
    /// Projection error at the image center
    pub center_error: f64,
    /// Projection error near the center
    pub near_center_error: f64,
    /// Projection error in the mid region
    pub mid_region_error: f64,
    /// Projection error at the edge region
    pub edge_region_error: f64,
    /// Projection error at the far edge
    pub far_edge_error: f64,
    /// Average error across all valid regions
    pub average_error: f64,
    /// Maximum error across all regions
    pub max_error: f64,
    /// Overall validation status (EXCELLENT/GOOD/NEEDS IMPROVEMENT)
    pub status: String,
    /// Detailed data for each region tested
    pub region_data: Vec<RegionValidation>,
}

/// Validation data for a single image region.
///
/// Contains the input and output projections for a test point,
/// along with the error between them.
#[derive(Debug, Clone, Serialize, Deserialize)]
pub struct RegionValidation {
    /// Name of the region (e.g., "Center", "Edge Region")
    pub name: String,
    /// Projection from the input model (u, v)
    pub input_projection: Option<(f64, f64)>,
    /// Projection from the output model (u, v)
    pub output_projection: Option<(f64, f64)>,
    /// Euclidean distance between input and output projections
    pub error: f64,
}

/// Validate conversion accuracy between concrete and trait object camera models.
///
/// This function tests how well a converted camera model (`output_model`) matches
/// the original model (`input_model`) by comparing projections at various test points
/// across the image. Test points are selected at strategic locations (center, edges, etc.)
/// to assess accuracy across the entire field of view.
///
/// # Arguments
///
/// * `output_model` - Target camera model (concrete type)
/// * `input_model` - Source camera model (trait object)
///
/// # Returns
///
/// * `Result<ValidationResults, UtilError>` - Detailed validation results with per-region errors
///
/// # Algorithm
///
/// 1. Define test points at: Center, Near Center, Mid Region, Edge Region, Far Edge
/// 2. For each test point:
///    - Unproject using input model to get 3D ray
///    - Project the 3D point using both input and output models
///    - Compute Euclidean distance between the two projections
/// 3. Aggregate errors and assign status (EXCELLENT < 0.001px, GOOD < 0.1px)
///
/// # Examples
///
/// ```rust,ignore
/// use apex_camera_models::camera::{KannalaBrandtModel, DoubleSphereModel};
/// use apex_camera_models::util::validate_conversion_accuracy;
///
/// let kb_model = KannalaBrandtModel::load_from_yaml("kb.yaml")?;
/// let ds_model = DoubleSphereModel::load_from_yaml("ds.yaml")?;
///
/// let results = validate_conversion_accuracy(&ds_model, &kb_model)?;
/// println!("Validation status: {}", results.status);
/// println!("Average error: {:.4}px", results.average_error);
/// ```
pub fn validate_conversion_accuracy<T>(
    output_model: &T,
    input_model: &dyn CameraModel,
) -> Result<ValidationResults, UtilError>
where
    T: CameraModel,
{
    // Get image resolution from input model
    let resolution = input_model.get_resolution();
    let width = resolution.width as f64;
    let height = resolution.height as f64;

    // Define test points based on image resolution as fractions
    let test_regions = [
        ("Center", Vector2::new(width * 0.5, height * 0.5)),
        ("Near Center", Vector2::new(width * 0.55, height * 0.55)),
        ("Mid Region", Vector2::new(width * 0.65, height * 0.65)),
        ("Edge Region", Vector2::new(width * 0.8, height * 0.8)),
        ("Far Edge", Vector2::new(width * 0.95, height * 0.95)),
    ];

    let mut total_error = 0.0;
    let mut max_error = 0.0;
    let mut valid_projections = 0;
    let mut region_errors = [f64::NAN; 5];
    let mut region_data = Vec::new();

    for (i, (name, test_point_2d)) in test_regions.iter().enumerate() {
        // First unproject the test point using input model to get 3D point
        match input_model.unproject(test_point_2d) {
            Ok(point_3d) => {
                // Now project the 3D point using both models
                match (
                    input_model.project(&point_3d),
                    output_model.project(&point_3d),
                ) {
                    (Ok(input_proj), Ok(output_proj)) => {
                        let error = (input_proj - output_proj).norm();
                        total_error += error;
                        max_error = f64::max(max_error, error);
                        valid_projections += 1;
                        region_errors[i] = error;

                        // Store the projection data
                        region_data.push(RegionValidation {
                            name: name.to_string(),
                            input_projection: Some((input_proj.x, input_proj.y)),
                            output_projection: Some((output_proj.x, output_proj.y)),
                            error,
                        });
                    }
                    (Ok(_), Err(_)) => {
                        region_errors[i] = f64::NAN;
                        region_data.push(RegionValidation {
                            name: name.to_string(),
                            input_projection: None,
                            output_projection: None,
                            error: f64::NAN,
                        });
                    }
                    (Err(_), Ok(_)) => {
                        region_errors[i] = f64::NAN;
                        region_data.push(RegionValidation {
                            name: name.to_string(),
                            input_projection: None,
                            output_projection: None,
                            error: f64::NAN,
                        });
                    }
                    (Err(_), Err(_)) => {
                        region_errors[i] = f64::NAN;
                        region_data.push(RegionValidation {
                            name: name.to_string(),
                            input_projection: None,
                            output_projection: None,
                            error: f64::NAN,
                        });
                    }
                }
            }
            Err(_) => {
                // If unprojection fails, skip this region
                region_errors[i] = f64::NAN;
                region_data.push(RegionValidation {
                    name: name.to_string(),
                    input_projection: None,
                    output_projection: None,
                    error: f64::NAN,
                });
            }
        }
    }

    let average_error = if valid_projections > 0 {
        total_error / valid_projections as f64
    } else {
        f64::NAN
    };

    let status = if average_error.is_nan() {
        "NEEDS IMPROVEMENT".to_string()
    } else if average_error < 0.001 {
        "EXCELLENT".to_string()
    } else if average_error < 0.1 {
        "GOOD".to_string()
    } else {
        "NEEDS IMPROVEMENT".to_string()
    };

    Ok(ValidationResults {
        center_error: region_errors[0],
        near_center_error: region_errors[1],
        mid_region_error: region_errors[2],
        edge_region_error: region_errors[3],
        far_edge_error: region_errors[4],
        average_error,
        max_error,
        status,
        region_data,
    })
}
