//! Field-of-View (FOV) Camera Model Implementation
//!
//! This module implements the Field-of-View camera model, which is particularly useful
//! for wide-angle and fisheye cameras. The model uses a field-of-view parameter to handle
//! the distortion characteristics of such cameras. It adheres to the [`CameraModel`]
//! trait defined in the parent `camera` module ([`crate::camera`]).
//!
//! # References
//!
//! The FOV model is based on:
//! "Simultaneous Localization and Mapping with Fisheye Cameras" by Zichao Zhang et al.
//! Available at: https://arxiv.org/pdf/1807.08957

use crate::camera::{validation, CameraModel, CameraModelError, Intrinsics, Resolution};
use log::info;
use nalgebra::{DVector, Vector2, Vector3};
use serde::{Deserialize, Serialize};
use std::fmt;

/// Implements the Field-of-View camera model for wide-angle/fisheye lenses.
///
/// The FOV model is designed for cameras with significant distortion,
/// common in wide-angle or fisheye lenses. It represents the camera using
/// standard intrinsic parameters ([`Intrinsics`]: fx, fy, cx, cy), image [`Resolution`],
/// and one distortion parameter: `w`.
/// `w` controls the field-of-view distortion characteristic.
///
/// # Fields
///
/// *   `intrinsics`: [`Intrinsics`] - Holds the focal lengths (fx, fy) and principal point (cx, cy).
/// *   `resolution`: [`Resolution`] - The width and height of the camera image in pixels.
/// *   `w`: `f64` - The field-of-view distortion parameter.
///     It must be in the range (ε, 3.0) where ε is a small positive value.
///
/// # References
///
/// *   Zhang, Z., et al. (2018). Simultaneous Localization and Mapping with Fisheye Cameras.
///
/// # Examples
///
/// ```rust
/// use nalgebra::DVector;
/// use apex_camera_models::camera::fov::FovModel;
/// use apex_camera_models::camera::{Intrinsics, Resolution, CameraModel, CameraModelError};
///
/// // Parameters: fx, fy, cx, cy, w
/// let params = DVector::from_vec(vec![379.045, 379.008, 505.512, 509.969, 0.9259487501905697]);
/// let mut fov_model = FovModel::new(&params).unwrap();
/// fov_model.resolution = Resolution { width: 752, height: 480 };
///
/// println!("Created FOV model: {:?}", fov_model);
/// assert_eq!(fov_model.intrinsics.fx, 379.045);
/// assert_eq!(fov_model.w, 0.9259487501905697);
/// ```
#[derive(Clone, Serialize, Deserialize)]
pub struct FovModel {
    /// Camera intrinsic parameters: `fx`, `fy`, `cx`, `cy`.
    pub intrinsics: Intrinsics,
    /// Image resolution as width and height in pixels.
    pub resolution: Resolution,
    /// Field-of-view distortion parameter.
    /// Must be in the range (ε, 3.0) where ε is a small positive value.
    pub w: f64,
}

impl FovModel {
    /// Creates a new [`FovModel`] from a DVector of parameters.
    ///
    /// This constructor initializes the model with the provided parameters.
    /// The image resolution is initialized to 0x0 and should be set explicitly
    /// or by loading from a configuration file like YAML.
    ///
    /// # Arguments
    ///
    /// * `parameters`: A `&DVector<f64>` containing the camera parameters in the following order:
    ///   1.  `fx`: Focal length along the x-axis.
    ///   2.  `fy`: Focal length along the y-axis.
    ///   3.  `cx`: Principal point x-coordinate.
    ///   4.  `cy`: Principal point y-coordinate.
    ///   5.  `w`: The field-of-view distortion parameter.
    ///
    /// # Return Value
    ///
    /// Returns a `Result<Self, CameraModelError>`. In the current implementation, this
    /// always returns `Ok(Self)` as no validation that can fail is performed within `new` itself.
    ///
    /// # Panics
    ///
    /// This function will panic if `parameters.len()` is less than 5, due to direct
    /// indexing (`parameters[0]` through `parameters[4]`).
    ///
    /// # Examples
    ///
    /// ```rust
    /// use nalgebra::DVector;
    /// use apex_camera_models::camera::fov::FovModel;
    /// use apex_camera_models::camera::Resolution;
    ///
    /// let params_vec = DVector::from_vec(vec![
    ///     379.045,             // fx
    ///     379.008,             // fy
    ///     505.512,             // cx
    ///     509.969,             // cy
    ///     0.9259487501905697   // w
    /// ]);
    /// let mut model = FovModel::new(&params_vec).unwrap();
    /// model.resolution = Resolution { width: 752, height: 480 }; // Set resolution manually
    ///
    /// assert_eq!(model.intrinsics.fx, 379.045);
    /// assert_eq!(model.w, 0.9259487501905697);
    /// assert_eq!(model.resolution.width, 752);
    /// ```
    pub fn new(parameters: &DVector<f64>) -> Result<Self, CameraModelError> {
        if parameters.len() != 5 {
            return Err(CameraModelError::InvalidParams(format!(
                "Expected 5 parameters (fx, fy, cx, cy, w), got {}",
                parameters.len()
            )));
        }

        let model = FovModel {
            intrinsics: Intrinsics {
                fx: parameters[0],
                fy: parameters[1],
                cx: parameters[2],
                cy: parameters[3],
            },
            resolution: Resolution {
                width: 0, // Resolution is typically set after creation or by loading.
                height: 0,
            },
            w: parameters[4],
        };

        info!("new FOV model is: {model:?}");
        Ok(model)
    }

    /// Performs linear estimation to initialize the w parameter from point correspondences.
    ///
    /// This method estimates the `w` parameter using a linear least squares approach
    /// given 3D-2D point correspondences. It assumes the intrinsic parameters (fx, fy, cx, cy)
    /// are already set.
    ///
    /// # Arguments
    ///
    /// * `points_3d`: Matrix3xX<f64> - 3D points in camera coordinates (each column is a point)
    /// * `points_2d`: Matrix2xX<f64> - Corresponding 2D points in image coordinates
    ///
    /// # Returns
    ///
    /// Returns `Ok(())` on success or a `CameraModelError` if the estimation fails.
    pub fn linear_estimation(
        &mut self,
        points_3d: &nalgebra::Matrix3xX<f64>,
        points_2d: &nalgebra::Matrix2xX<f64>,
    ) -> Result<(), CameraModelError> {
        // Check if the number of 2D and 3D points match
        if points_2d.ncols() != points_3d.ncols() {
            return Err(CameraModelError::InvalidParams(
                "Number of 2D and 3D points must match".to_string(),
            ));
        }

        let num_points = points_2d.ncols();

        // Need at least 2 points for linear estimation
        if num_points < 2 {
            return Err(CameraModelError::InvalidParams(
                "Need at least 2 point correspondences for linear estimation".to_string(),
            ));
        }

        // Set up the linear system to solve for w
        // We'll use a simplified approach: estimate w that minimizes reprojection error
        // Start with a reasonable initial value
        let mut best_w = 1.0;
        let mut best_error = f64::INFINITY;

        // Grid search over reasonable w values
        for w_test in (10..300).map(|i| i as f64 / 100.0) {
            let mut error_sum = 0.0;
            let mut valid_count = 0;

            for i in 0..num_points {
                let x = points_3d[(0, i)];
                let y = points_3d[(1, i)];
                let z = points_3d[(2, i)];
                let u_observed = points_2d[(0, i)];
                let v_observed = points_2d[(1, i)];

                // Try projection with this w value
                let r2 = x * x + y * y;
                let r = r2.sqrt();

                let tan_w_half = (w_test / 2.0).tan();
                let atan_wrd = (2.0 * tan_w_half * r).atan2(z);

                let eps_sqrt = f64::EPSILON.sqrt();
                let rd = if r2 < eps_sqrt {
                    2.0 * tan_w_half / w_test
                } else {
                    atan_wrd / (r * w_test)
                };

                let mx = x * rd;
                let my = y * rd;

                let u_predicted = self.intrinsics.fx * mx + self.intrinsics.cx;
                let v_predicted = self.intrinsics.fy * my + self.intrinsics.cy;

                let error = ((u_predicted - u_observed).powi(2)
                    + (v_predicted - v_observed).powi(2))
                .sqrt();

                if error.is_finite() {
                    error_sum += error;
                    valid_count += 1;
                }
            }

            if valid_count > 0 {
                let avg_error = error_sum / valid_count as f64;
                if avg_error < best_error {
                    best_error = avg_error;
                    best_w = w_test;
                }
            }
        }

        self.w = best_w;

        info!(
            "FOV linear estimation results: w = {}, avg_error = {}",
            self.w, best_error
        );

        // Clamp w to valid range
        if self.w <= f64::EPSILON {
            info!("w {} is too small, clamping to 0.01", self.w);
            self.w = 0.01;
        } else if self.w > 3.0 {
            info!("w {} is too large, clamping to 3.0", self.w);
            self.w = 3.0;
        }

        // Validate parameters
        self.validate_params()?;

        Ok(())
    }
}

/// Provides a debug string representation for [`FovModel`].
impl fmt::Debug for FovModel {
    fn fmt(&self, f: &mut fmt::Formatter<'_>) -> fmt::Result {
        write!(
            f,
            "FOV [fx: {} fy: {} cx: {} cy: {} w: {}]",
            self.intrinsics.fx, self.intrinsics.fy, self.intrinsics.cx, self.intrinsics.cy, self.w
        )
    }
}

impl CameraModel for FovModel {
    /// Projects a 3D point from camera coordinates to 2D image coordinates.
    ///
    /// This method applies the FOV projection equations. It computes the
    /// radial distance, applies the FOV distortion function using the `w` parameter,
    /// and then scales by focal lengths and adds principal point offsets.
    ///
    /// # Arguments
    ///
    /// * `point_3d`: A `&Vector3<f64>` representing the 3D point (X, Y, Z) in camera coordinates.
    ///
    /// # Return Value
    ///
    /// Returns a `Result<Vector2<f64>, CameraModelError>`.
    /// On success, it provides the projected 2D point (`Vector2<f64>`) in pixel coordinates (u, v).
    ///
    /// # Errors
    ///
    /// * [`CameraModelError::PointAtCameraCenter`]: If the point's z-coordinate is too close to zero.
    fn project(&self, point_3d: &Vector3<f64>) -> Result<Vector2<f64>, CameraModelError> {
        let x = point_3d.x;
        let y = point_3d.y;
        let z = point_3d.z;

        // Check if z is valid
        if z < f64::EPSILON.sqrt() {
            return Err(CameraModelError::PointAtCameraCenter);
        }

        let r2 = x * x + y * y;
        let r = r2.sqrt();

        let tan_w_half = (self.w / 2.0).tan();
        let atan_wrd = (2.0 * tan_w_half * r).atan2(z);

        let eps_sqrt = f64::EPSILON.sqrt();

        let rd = if r2 < eps_sqrt {
            // For points very close to the optical axis, use Taylor expansion
            2.0 * tan_w_half / self.w
        } else {
            atan_wrd / (r * self.w)
        };

        let mx = x * rd;
        let my = y * rd;

        let projected_x = self.intrinsics.fx * mx + self.intrinsics.cx;
        let projected_y = self.intrinsics.fy * my + self.intrinsics.cy;

        Ok(Vector2::new(projected_x, projected_y))
    }

    /// Unprojects a 2D image point to a 3D ray in camera coordinates.
    ///
    /// This method applies the inverse FOV model equations to convert
    /// a 2D pixel coordinate back into a 3D direction vector (ray) originating
    /// from the camera center. The resulting vector is normalized.
    ///
    /// # Arguments
    ///
    /// * `point_2d`: A `&Vector2<f64>` representing the 2D point (u, v) in pixel coordinates.
    ///
    /// # Return Value
    ///
    /// Returns a `Result<Vector3<f64>, CameraModelError>`.
    /// On success, it provides the normalized 3D ray (`Vector3<f64>`) corresponding to the 2D point.
    ///
    /// # Errors
    ///
    /// * [`CameraModelError::NumericalError`]: If the unprojection encounters numerical issues.
    fn unproject(&self, point_2d: &Vector2<f64>) -> Result<Vector3<f64>, CameraModelError> {
        let u = point_2d.x;
        let v = point_2d.y;

        let tan_w_2 = (self.w / 2.0).tan();
        let mul2tanwby2 = tan_w_2 * 2.0;

        let mx = (u - self.intrinsics.cx) / self.intrinsics.fx;
        let my = (v - self.intrinsics.cy) / self.intrinsics.fy;

        let r2 = mx * mx + my * my;
        let rd = r2.sqrt();

        let eps_sqrt = f64::EPSILON.sqrt();

        let (x, y, z) = if mul2tanwby2 > eps_sqrt && rd > eps_sqrt {
            let sin_rd_w = (rd * self.w).sin();
            let cos_rd_w = (rd * self.w).cos();
            let ru = sin_rd_w / (rd * mul2tanwby2);

            (mx * ru / cos_rd_w, my * ru / cos_rd_w, 1.0)
        } else {
            (mx, my, 1.0)
        };

        let point3d = Vector3::new(x, y, z);
        Ok(point3d.normalize())
    }

    /// Loads [`FovModel`] parameters from a YAML file.
    ///
    /// The YAML file is expected to follow a structure where camera parameters are nested
    /// under `cam0`. The intrinsic parameters (`fx`, `fy`, `cx`, `cy`) and the FOV
    /// specific distortion parameter (`w`) are typically grouped
    /// together in an `intrinsics` array in the YAML file: `[fx, fy, cx, cy, w]`.
    /// The `resolution` (width, height) is also expected under `cam0`.
    ///
    /// # Arguments
    ///
    /// * `path`: A string slice representing the path to the YAML file.
    ///
    /// # Return Value
    ///
    /// Returns a `Result<Self, CameraModelError>`. On success, it provides an instance
    /// of [`FovModel`] populated with parameters from the file.
    ///
    /// # Errors
    ///
    /// This function can return:
    /// * [`CameraModelError::IOError`]: If there's an issue reading the file.
    /// * [`CameraModelError::YamlError`]: If the YAML content is malformed or cannot be parsed.
    /// * [`CameraModelError::InvalidParams`]: If the YAML structure is missing expected fields
    ///   or if parameter values are of incorrect types or counts.
    fn load_from_yaml(path: &str) -> Result<Self, CameraModelError> {
        use crate::camera::yaml_io;

        // Use helper function to parse YAML (FOV has 5 params: fx, fy, cx, cy, w)
        let (intrinsics, resolution, extra_params) = yaml_io::parse_yaml_camera(path, 5)?;

        // Extract w from extra_params
        if extra_params.len() != 1 {
            return Err(CameraModelError::InvalidParams(format!(
                "FOV model expects exactly 5 parameters (fx, fy, cx, cy, w), got {}",
                4 + extra_params.len()
            )));
        }

        let w = extra_params[0];

        let model = FovModel {
            intrinsics,
            resolution,
            w,
        };

        // Validate parameters
        model.validate_params()?;
        Ok(model)
    }

    /// Saves the [`FovModel`] parameters to a YAML file.
    ///
    /// The parameters are saved under the `cam0` key. The `intrinsics` YAML array
    /// will contain `fx, fy, cx, cy, w` in that order.
    ///
    /// # Arguments
    ///
    /// * `path`: A string slice representing the path to the YAML file where parameters will be saved.
    ///
    /// # Return Value
    ///
    /// Returns `Ok(())` on successful save.
    ///
    /// # Errors
    ///
    /// This function can return:
    /// * [`CameraModelError::YamlError`]: If there's an issue serializing the data to YAML format.
    /// * [`CameraModelError::IOError`]: If there's an issue creating or writing to the file.
    fn save_to_yaml(&self, path: &str) -> Result<(), CameraModelError> {
        use crate::camera::yaml_io;

        // Use helper function to save YAML (w as extra parameter)
        yaml_io::save_yaml_camera(path, "fov", &self.intrinsics, &self.resolution, &[self.w])
    }

    /// Validates the parameters of the [`FovModel`].
    ///
    /// This method checks the validity of the core intrinsic parameters (focal lengths,
    /// principal point) using [`validation::validate_intrinsics`]. It also validates
    /// the FOV specific parameter:
    /// *   `w` must be in the range (ε, 3.0) where ε is a small positive value.
    ///
    /// # Return Value
    ///
    /// Returns `Ok(())` if all parameters are valid.
    ///
    /// # Errors
    ///
    /// Returns a [`CameraModelError`] if any parameter is invalid:
    /// * [`CameraModelError::InvalidParams`]: If `w` is not in the valid range.
    /// * Errors propagated from [`validation::validate_intrinsics`].
    fn validate_params(&self) -> Result<(), CameraModelError> {
        validation::validate_intrinsics(&self.intrinsics)?;

        if !self.w.is_finite() || self.w <= f64::EPSILON || self.w > 3.0 {
            return Err(CameraModelError::InvalidParams(format!(
                "w must be in range (epsilon, 3.0], got {}",
                self.w
            )));
        }

        Ok(())
    }

    /// Returns a clone of the camera's image resolution.
    fn get_resolution(&self) -> Resolution {
        self.resolution.clone()
    }

    /// Returns a clone of the camera's intrinsic parameters.
    fn get_intrinsics(&self) -> Intrinsics {
        self.intrinsics.clone()
    }

    /// Returns the distortion parameters of the FOV model.
    ///
    /// The parameter is returned as a vector containing `[w]`.
    ///
    /// # Return Value
    ///
    /// A `Vec<f64>` containing the distortion parameter: `[w]`.
    fn get_distortion(&self) -> Vec<f64> {
        vec![self.w]
    }

    /// Returns the name of the Field-of-View Camera Model.
    ///
    /// # Return Value
    ///
    /// A `&'static str` containing "fov".
    fn get_model_name(&self) -> &'static str {
        "fov"
    }
}

/// Unit tests for the [`FovModel`].
#[cfg(test)]
mod tests {
    use super::*;
    use approx::assert_relative_eq; // For floating point comparisons

    /// Helper function to create a sample [`FovModel`] instance for testing.
    fn get_sample_model() -> FovModel {
        FovModel {
            intrinsics: Intrinsics {
                fx: 379.045,
                fy: 379.008,
                cx: 505.512,
                cy: 509.969,
            },
            resolution: Resolution {
                width: 752,
                height: 480,
            },
            w: 0.9259487501905697,
        }
    }

    /// Tests loading [`FovModel`] parameters from "samples/fov.yaml".
    #[test]
    fn test_fov_load_from_yaml() {
        let path = "samples/fov.yaml";
        let model = FovModel::load_from_yaml(path).unwrap();

        assert_eq!(model.intrinsics.fx, 379.045);
        assert_eq!(model.intrinsics.fy, 379.008);
        assert_eq!(model.intrinsics.cx, 505.512);
        assert_eq!(model.intrinsics.cy, 509.969);
        assert_eq!(model.w, 0.9259487501905697);
        assert_eq!(model.resolution.width, 752);
        assert_eq!(model.resolution.height, 480);
    }

    /// Tests saving [`FovModel`] parameters to a YAML file and then reloading them.
    #[test]
    fn test_fov_save_to_yaml() {
        use std::fs;

        // Create output directory if it doesn't exist
        fs::create_dir_all("output").unwrap_or_else(|_| {
            info!("Output directory already exists or couldn't be created");
        });

        // Define input and output paths
        let input_path = "samples/fov.yaml";
        let output_path = "output/fov_saved.yaml";

        // Load the camera model from the original YAML
        let model = FovModel::load_from_yaml(input_path).unwrap();

        // Save the model to the output path
        model.save_to_yaml(output_path).unwrap();

        // Load the model from the saved file
        let saved_model = FovModel::load_from_yaml(output_path).unwrap();

        // Compare the original and saved models
        assert_eq!(model.intrinsics.fx, saved_model.intrinsics.fx);
        assert_eq!(model.intrinsics.fy, saved_model.intrinsics.fy);
        assert_eq!(model.intrinsics.cx, saved_model.intrinsics.cx);
        assert_eq!(model.intrinsics.cy, saved_model.intrinsics.cy);
        assert_eq!(model.w, saved_model.w);
        assert_eq!(model.resolution.width, saved_model.resolution.width);
        assert_eq!(model.resolution.height, saved_model.resolution.height);

        // Clean up the saved file
        fs::remove_file(output_path).unwrap();
    }

    /// Tests the consistency of projection and unprojection for the [`FovModel`].
    #[test]
    fn test_fov_project_unproject() {
        // Load the camera model from YAML
        let path = "samples/fov.yaml";
        let model = FovModel::load_from_yaml(path).unwrap();

        // Create a 3D point in camera coordinates
        let point_3d = Vector3::new(0.1, 0.1, 3.0);
        let norm_3d = point_3d.normalize();

        // Project the 3D point to pixel coordinates
        let point_2d = model.project(&point_3d).unwrap();

        // Debug print the projected coordinates
        println!("Projected point: ({}, {})", point_2d.x, point_2d.y);
        println!(
            "Image bounds: {}x{}",
            model.resolution.width, model.resolution.height
        );

        // Check if the pixel coordinates are finite
        assert!(point_2d.x.is_finite() && point_2d.y.is_finite());

        // Unproject the pixel point back to a 3D ray direction
        let point_3d_unprojected = model.unproject(&point_2d).unwrap();

        // Check if the unprojected point (normalized ray) is close to the original normalized point
        assert_relative_eq!(norm_3d.x, point_3d_unprojected.x, epsilon = 1e-4);
        assert_relative_eq!(norm_3d.y, point_3d_unprojected.y, epsilon = 1e-4);
        assert_relative_eq!(norm_3d.z, point_3d_unprojected.z, epsilon = 1e-4);
    }

    /// Tests projection and unprojection with a point near the image center.
    #[test]
    fn test_fov_project_unproject_center() {
        let model = get_sample_model();

        // Use a point that should project near the image center
        let point_3d = Vector3::new(0.0, 0.0, 1.0);
        let norm_3d = point_3d.normalize();

        // Project the 3D point to pixel coordinates
        let point_2d = model.project(&point_3d).unwrap();

        // Should project near the principal point
        assert_relative_eq!(point_2d.x, model.intrinsics.cx, epsilon = 1.0);
        assert_relative_eq!(point_2d.y, model.intrinsics.cy, epsilon = 1.0);

        // Unproject back
        let point_3d_unprojected = model.unproject(&point_2d).unwrap();

        // Check round-trip consistency
        assert_relative_eq!(norm_3d.x, point_3d_unprojected.x, epsilon = 1e-6);
        assert_relative_eq!(norm_3d.y, point_3d_unprojected.y, epsilon = 1e-6);
        assert_relative_eq!(norm_3d.z, point_3d_unprojected.z, epsilon = 1e-6);
    }

    /// Tests projection of points near or at the camera center.
    #[test]
    fn test_project_point_at_center() {
        let model = get_sample_model();

        // Point very close to origin on Z axis should still work
        let point_3d_on_z = Vector3::new(0.0, 0.0, 0.1);
        let result_origin = model.project(&point_3d_on_z);

        if let Ok(p) = result_origin {
            assert_relative_eq!(p.x, model.intrinsics.cx, epsilon = 1e-3);
            assert_relative_eq!(p.y, model.intrinsics.cy, epsilon = 1e-3);
        }

        // Point exactly at (0,0,0) should error
        let result_exact_origin = model.project(&Vector3::new(0.0, 0.0, 0.0));
        assert!(
            matches!(
                result_exact_origin,
                Err(CameraModelError::PointAtCameraCenter)
            ),
            "Projecting (0,0,0) should result in PointAtCameraCenter error."
        );
    }

    /// Tests projection of a point located behind the camera.
    #[test]
    fn test_project_point_behind_camera() {
        let model = get_sample_model();
        let point_3d = Vector3::new(0.1, 0.2, -1.0); // Point behind camera
        let result = model.project(&point_3d);
        // Expect an error as the point is behind the camera
        assert!(matches!(result, Err(CameraModelError::PointAtCameraCenter)));
    }

    /// Tests `validate_params` with a valid model.
    #[test]
    fn test_validate_params_valid() {
        let model = get_sample_model();
        assert!(model.validate_params().is_ok());
    }

    /// Tests `validate_params` with invalid `w` values.
    #[test]
    fn test_validate_params_invalid_w() {
        let mut model = get_sample_model();

        model.w = 0.0; // Invalid: w must be > epsilon
        assert!(matches!(
            model.validate_params(),
            Err(CameraModelError::InvalidParams(_))
        ));

        model.w = -0.5; // Invalid: w must be positive
        assert!(matches!(
            model.validate_params(),
            Err(CameraModelError::InvalidParams(_))
        ));

        model.w = 3.5; // Invalid: w must be <= 3.0
        assert!(matches!(
            model.validate_params(),
            Err(CameraModelError::InvalidParams(_))
        ));

        model.w = f64::NAN; // Invalid: w must be finite
        assert!(matches!(
            model.validate_params(),
            Err(CameraModelError::InvalidParams(_))
        ));
    }

    /// Tests `validate_params` with invalid core intrinsic parameters.
    #[test]
    fn test_validate_params_invalid_intrinsics() {
        let mut model = get_sample_model();
        model.intrinsics.fx = 0.0; // Invalid: fx must be > 0
        assert!(matches!(
            model.validate_params(),
            Err(CameraModelError::FocalLengthMustBePositive)
        ));
    }

    /// Tests the getter methods: `get_intrinsics`, `get_resolution`, and `get_distortion`.
    #[test]
    fn test_getters() {
        let model = get_sample_model();

        let intrinsics = model.get_intrinsics();
        assert_relative_eq!(intrinsics.fx, model.intrinsics.fx);
        assert_relative_eq!(intrinsics.fy, model.intrinsics.fy);
        assert_relative_eq!(intrinsics.cx, model.intrinsics.cx);
        assert_relative_eq!(intrinsics.cy, model.intrinsics.cy);

        let resolution = model.get_resolution();
        assert_eq!(resolution.width, model.resolution.width);
        assert_eq!(resolution.height, model.resolution.height);

        let distortion = model.get_distortion();
        assert_eq!(distortion.len(), 1);
        assert_relative_eq!(distortion[0], model.w);
    }

    /// Tests the `new` constructor.
    #[test]
    fn test_new() {
        let params =
            DVector::from_vec(vec![379.045, 379.008, 505.512, 509.969, 0.9259487501905697]);
        let model = FovModel::new(&params).unwrap();

        assert_eq!(model.intrinsics.fx, 379.045);
        assert_eq!(model.intrinsics.fy, 379.008);
        assert_eq!(model.intrinsics.cx, 505.512);
        assert_eq!(model.intrinsics.cy, 509.969);
        assert_eq!(model.w, 0.9259487501905697);
    }

    /// Tests the `new` constructor with invalid parameter count.
    #[test]
    fn test_new_invalid_param_count() {
        let params = DVector::from_vec(vec![379.045, 379.008, 505.512, 509.969]); // Only 4 params
        let result = FovModel::new(&params);
        assert!(matches!(result, Err(CameraModelError::InvalidParams(_))));
    }
}
