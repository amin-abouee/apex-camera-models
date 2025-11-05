//! This module defines the core functionalities for various camera models.
//!
//! It provides a unified interface for camera operations such as projecting 3D points
//! to 2D image coordinates and unprojecting 2D image coordinates to 3D rays.
//! The module also includes definitions for camera intrinsic parameters, resolution,
//! and error handling for camera operations.
//!
//! This module re-exports several specific camera model implementations from its submodules:
//! - `double_sphere`: Implements the Double Sphere camera model.
//! - `kannala_brandt`: Implements the Kannala-Brandt camera model.
//! - `pinhole`: Implements the Pinhole camera model.
//! - `rad_tan`: Implements the Radial-Tangential distortion model (often used with pinhole).
//!
//! It also contains a `validation` submodule for common parameter validation logic.

use nalgebra::{Vector2, Vector3};
use serde::{Deserialize, Serialize};

// Camera model modules
pub mod double_sphere;
pub mod eucm;
pub mod fov;
pub mod kannala_brandt;
pub mod pinhole;
pub mod rad_tan;
pub mod ucm;

// Re-export camera models
pub use double_sphere::DoubleSphereModel;
pub use eucm::EucmModel;
pub use fov::FovModel;
pub use kannala_brandt::KannalaBrandtModel;
pub use pinhole::PinholeModel;
pub use rad_tan::RadTanModel;
pub use ucm::UcmModel;

#[derive(Debug, Clone, Serialize, Deserialize)]
pub enum CameraModelEnum {
    DoubleSphere(DoubleSphereModel),
    Eucm(EucmModel),
    Fov(FovModel),
    KannalaBrandt(KannalaBrandtModel),
    Pinhole(PinholeModel),
    RadTan(RadTanModel),
    Ucm(UcmModel),
}

/// Represents the intrinsic parameters of a camera.
///
/// These parameters define the internal geometry of the camera,
/// including focal length and principal point.
#[derive(Debug, Clone, Serialize, Deserialize)]
pub struct Intrinsics {
    /// The focal length along the x-axis, in pixels.
    pub fx: f64,
    /// The focal length along the y-axis, in pixels.
    pub fy: f64,
    /// The x-coordinate of the principal point (optical center), in pixels.
    pub cx: f64,
    /// The y-coordinate of the principal point (optical center), in pixels.
    pub cy: f64,
}

/// Represents the resolution of a camera image.
///
/// This struct holds the width and height of the image sensor in pixels.
#[derive(Debug, Clone, Serialize, Deserialize)]
pub struct Resolution {
    /// The width of the image in pixels.
    pub width: u32,
    /// The height of the image in pixels.
    pub height: u32,
}

/// Defines the possible errors that can occur during camera model operations.
///
/// This enum covers errors related to projection, unprojection, parameter validation,
/// file I/O, and numerical issues.
#[derive(thiserror::Error, Debug)]
pub enum CameraModelError {
    /// Error indicating that a 3D point projects outside the valid image area.
    #[error("Projection is outside the image")]
    ProjectionOutSideImage,
    /// Error indicating that an input 2D point for unprojection is outside the valid image area.
    #[error("Input point is outside the image")]
    PointIsOutSideImage,
    /// Error indicating that a 3D point is too close to the camera center (z-coordinate is near zero),
    /// making projection or unprojection numerically unstable or undefined.
    #[error("z is close to zero, point is at camera center")]
    PointAtCameraCenter,
    /// Error indicating that a focal length parameter (fx or fy) is not positive.
    #[error("Focal length must be positive")]
    FocalLengthMustBePositive,
    /// Error indicating that a principal point coordinate (cx or cy) is not a finite number.
    #[error("Principal point must be finite")]
    PrincipalPointMustBeFinite,
    /// Error indicating that one or more camera parameters are invalid.
    /// Contains a string describing the specific parameter issue.
    #[error("Invalid camera parameters: {0}")]
    InvalidParams(String),
    /// Error indicating a failure during YAML deserialization when loading camera parameters.
    /// Contains a string describing the YAML parsing error.
    #[error("Failed to load YAML: {0}")]
    YamlError(String),
    /// Error indicating a failure during file input/output operations.
    /// Contains a string describing the I/O error.
    #[error("IO Error: {0}")]
    IOError(String),
    /// Error indicating a numerical instability or issue during calculations.
    /// Contains a string describing the numerical error.
    #[error("NumericalError: {0}")]
    NumericalError(String),
}

/// Implements the conversion from `std::io::Error` to `CameraModelError::IOError`.
///
/// This allows for seamless handling of I/O errors within the camera model context.
impl From<std::io::Error> for CameraModelError {
    fn from(err: std::io::Error) -> Self {
        CameraModelError::IOError(err.to_string())
    }
}

/// Implements the conversion from `yaml_rust::ScanError` to `CameraModelError::YamlError`.
///
/// This allows for seamless handling of YAML parsing errors when loading camera parameters.
impl From<yaml_rust::ScanError> for CameraModelError {
    fn from(err: yaml_rust::ScanError) -> Self {
        CameraModelError::YamlError(err.to_string())
    }
}

/// Validates that a projected 2D point falls within the image boundaries.
///
/// # Arguments
///
/// * `u` - The x-coordinate (horizontal) of the projected point in pixels
/// * `v` - The y-coordinate (vertical) of the projected point in pixels
/// * `resolution` - The image resolution (width and height)
///
/// # Returns
///
/// * `Ok(())` if the point is within bounds
/// * `Err(CameraModelError::ProjectionOutSideImage)` if the point is outside the image
///
/// # Examples
///
/// ```rust,ignore
/// use apex_camera_models::camera::validate_projection_bounds;
/// use apex_camera_models::camera::Resolution;
///
/// let resolution = Resolution { width: 640, height: 480 };
/// assert!(validate_projection_bounds(320.0, 240.0, &resolution).is_ok());
/// assert!(validate_projection_bounds(-10.0, 240.0, &resolution).is_err());
/// assert!(validate_projection_bounds(320.0, 500.0, &resolution).is_err());
/// ```
pub fn validate_projection_bounds(
    u: f64,
    v: f64,
    resolution: &Resolution,
) -> Result<(), CameraModelError> {
    if u < 0.0 || u >= resolution.width as f64 || v < 0.0 || v >= resolution.height as f64 {
        return Err(CameraModelError::ProjectionOutSideImage);
    }
    Ok(())
}

/// Validates that a 2D image point falls within the image boundaries for unprojection.
///
/// # Arguments
///
/// * `point_2d` - The 2D point in pixel coordinates (u, v)
/// * `resolution` - The image resolution (width and height)
///
/// # Returns
///
/// * `Ok(())` if the point is within bounds
/// * `Err(CameraModelError::PointIsOutSideImage)` if the point is outside the image
///
/// # Examples
///
/// ```rust,ignore
/// use apex_camera_models::camera::validate_unprojection_bounds;
/// use apex_camera_models::camera::Resolution;
/// use nalgebra::Vector2;
///
/// let resolution = Resolution { width: 640, height: 480 };
/// let valid_point = Vector2::new(320.0, 240.0);
/// let invalid_point = Vector2::new(-10.0, 240.0);
///
/// assert!(validate_unprojection_bounds(&valid_point, &resolution).is_ok());
/// assert!(validate_unprojection_bounds(&invalid_point, &resolution).is_err());
/// ```
pub fn validate_unprojection_bounds(
    point_2d: &Vector2<f64>,
    resolution: &Resolution,
) -> Result<(), CameraModelError> {
    if point_2d.x < 0.0
        || point_2d.x >= resolution.width as f64
        || point_2d.y < 0.0
        || point_2d.y >= resolution.height as f64
    {
        return Err(CameraModelError::PointIsOutSideImage);
    }
    Ok(())
}

/// Validates that a 3D point's z-coordinate is positive (in front of camera).
///
/// # Arguments
///
/// * `z` - The z-coordinate of the 3D point in camera space
///
/// # Returns
///
/// * `Ok(())` if z is sufficiently positive
/// * `Err(CameraModelError::PointAtCameraCenter)` if z is too close to zero or negative
///
/// # Examples
///
/// ```rust,ignore
/// use apex_camera_models::camera::validate_point_in_front;
///
/// assert!(validate_point_in_front(1.0).is_ok());
/// assert!(validate_point_in_front(0.001).is_ok());
/// assert!(validate_point_in_front(0.0).is_err());
/// assert!(validate_point_in_front(-1.0).is_err());
/// ```
pub fn validate_point_in_front(z: f64) -> Result<(), CameraModelError> {
    if z < f64::EPSILON.sqrt() {
        return Err(CameraModelError::PointAtCameraCenter);
    }
    Ok(())
}

/// Defines the core functionality and interface for all camera models.
///
/// This trait provides a common set of methods that any camera model implementation
/// must provide, such as projection, unprojection, parameter loading/saving,
/// validation, and retrieval of intrinsic and distortion parameters.
pub trait CameraModel {
    /// Projects a 3D point from the camera's coordinate system to 2D image coordinates.
    ///
    /// # Arguments
    /// * `point_3d` - A reference to a `Vector3<f64>` representing the 3D point (X, Y, Z) in camera coordinates.
    /// * `compute_jacobian` - A boolean flag indicating whether to compute the Jacobian of the projection function.
    ///
    /// # Returns
    /// A `Result` containing:
    /// * `Ok((Vector2<f64>, Option<DMatrix<f64>>))`: A tuple where the first element is the
    ///   projected 2D point (u, v) in pixel coordinates, and the second element is an `Option`
    ///   containing the Jacobian matrix (2x3) if `compute_jacobian` was true, otherwise `None`.
    /// * `Err(CameraModelError)`: An error if the projection fails (e.g., point is behind the camera,
    ///   projects outside the image, or numerical issues). Possible errors include
    ///   `ProjectionOutSideImage` and `PointAtCameraCenter`.
    fn project(&self, point_3d: &Vector3<f64>) -> Result<Vector2<f64>, CameraModelError>;

    /// Unprojects a 2D point from image coordinates to a 3D ray in the camera's coordinate system.
    ///
    /// The resulting 3D vector is a direction ray originating from the camera center.
    /// Its Z component is typically normalized to 1, but this can vary by model.
    ///
    /// # Arguments
    /// * `point_2d` - A reference to a `Vector2<f64>` representing the 2D point (u, v) in pixel coordinates.
    ///
    /// # Returns
    /// A `Result` containing:
    /// * `Ok(Vector3<f64>)`: The 3D ray (direction vector) corresponding to the 2D point.
    /// * `Err(CameraModelError)`: An error if the unprojection fails (e.g., point is outside the image,
    ///   or numerical issues). Possible errors include `PointIsOutSideImage`.
    fn unproject(&self, point_2d: &Vector2<f64>) -> Result<Vector3<f64>, CameraModelError>;

    /// Loads camera parameters from a YAML file.
    ///
    /// This method should parse a YAML file specified by `path` and populate the
    /// camera model's parameters.
    ///
    /// # Arguments
    /// * `path` - A string slice representing the path to the YAML file.
    ///
    /// # Returns
    /// A `Result` containing:
    /// * `Ok(Self)`: An instance of the camera model with parameters loaded from the file.
    /// * `Err(CameraModelError)`: An error if loading fails (e.g., file not found, YAML parsing error,
    ///   invalid parameters). Possible errors include `IOError`, `YamlError`, and `InvalidParams`.
    fn load_from_yaml(path: &str) -> Result<Self, CameraModelError>
    where
        Self: Sized;

    /// Saves the camera model's parameters to a YAML file.
    ///
    /// # Arguments
    /// * `path` - A string slice representing the path to the YAML file where parameters will be saved.
    ///
    /// # Returns
    /// A `Result` containing:
    /// * `Ok(())`: If saving was successful.
    /// * `Err(CameraModelError)`: An error if saving fails (e.g., I/O error). Possible errors include `IOError`.
    fn save_to_yaml(&self, path: &str) -> Result<(), CameraModelError>;

    /// Validates the current camera parameters.
    ///
    /// This method checks if the intrinsic parameters, distortion coefficients (if any),
    /// and other model-specific parameters are valid.
    ///
    /// # Returns
    /// A `Result` containing:
    /// * `Ok(())`: If all parameters are valid.
    /// * `Err(CameraModelError)`: An error describing the validation failure. Possible errors include
    ///   `FocalLengthMustBePositive`, `PrincipalPointMustBeFinite`, and `InvalidParams`.
    fn validate_params(&self) -> Result<(), CameraModelError>;

    /// Returns the resolution of the camera.
    ///
    /// # Returns
    /// A `Resolution` struct containing the width and height of the camera image.
    fn get_resolution(&self) -> Resolution;

    /// Returns the intrinsic parameters of the camera.
    ///
    /// # Returns
    /// An `Intrinsics` struct containing the focal lengths (fx, fy) and principal point (cx, cy).
    fn get_intrinsics(&self) -> Intrinsics;

    /// Returns the distortion parameters of the camera.
    ///
    /// The specific meaning and number of distortion parameters depend on the camera model.
    ///
    /// # Returns
    /// A `Vec<f64>` containing the distortion coefficients.
    fn get_distortion(&self) -> Vec<f64>;

    /// Returns the name of the camera model.
    ///
    /// This method returns a string identifier for the specific camera model type.
    ///
    /// # Returns
    /// A `&'static str` containing the name of the camera model (e.g., "double sphere", "eucm", etc.).
    fn get_model_name(&self) -> &'static str;
}

/// Provides common validation functions for camera parameters.
///
/// This module groups utility functions used to validate parts of camera models,
/// such as intrinsic parameters, ensuring they meet common criteria (e.g., positive focal length).
pub mod validation {
    use super::*;

    /// Validates the intrinsic camera parameters.
    ///
    /// Checks if the focal lengths (fx, fy) are positive and if the principal
    /// point coordinates (cx, cy) are finite numbers.
    ///
    /// # Arguments
    /// * `intrinsics` - A reference to an `Intrinsics` struct containing the parameters to validate.
    ///
    /// # Returns
    /// A `Result` containing:
    /// * `Ok(())`: If the intrinsic parameters are valid.
    /// * `Err(CameraModelError)`: An error if validation fails. Possible errors include
    ///   `FocalLengthMustBePositive` and `PrincipalPointMustBeFinite`.
    pub fn validate_intrinsics(intrinsics: &Intrinsics) -> Result<(), CameraModelError> {
        if intrinsics.fx <= 0.0 || intrinsics.fy <= 0.0 {
            return Err(CameraModelError::FocalLengthMustBePositive);
        }
        if !intrinsics.cx.is_finite() || !intrinsics.cy.is_finite() {
            return Err(CameraModelError::PrincipalPointMustBeFinite);
        }
        Ok(())
    }
}

/// Provides common YAML I/O helper functions to reduce code duplication across camera models.
///
/// This module contains utility functions for parsing and saving YAML files in the
/// standard camera calibration format used by all camera models.
pub mod yaml_io {
    use super::*;
    use std::fs;
    use std::io::Write;
    use yaml_rust::YamlLoader;

    /// Parses intrinsics, resolution, and optional extra parameters from a YAML file.
    ///
    /// This helper function extracts the common structure present in all camera model YAML files:
    /// - Loads and parses the YAML document
    /// - Extracts the `cam0` node
    /// - Parses the `intrinsics` array (fx, fy, cx, cy, ...extra params)
    /// - Parses the `resolution` array (width, height)
    ///
    /// # Arguments
    ///
    /// * `path` - Path to the YAML file
    /// * `min_intrinsics_len` - Minimum number of intrinsic parameters expected
    ///   (4 for pinhole, 5 for UCM, 6 for double sphere, etc.)
    ///
    /// # Returns
    ///
    /// Returns a tuple containing:
    /// * `Intrinsics` - The parsed intrinsic parameters (fx, fy, cx, cy)
    /// * `Resolution` - The parsed image resolution (width, height)
    /// * `Vec<f64>` - Any extra parameters beyond the first 4 (distortion params, etc.)
    ///
    /// # Errors
    ///
    /// Returns `CameraModelError` if:
    /// * File cannot be read
    /// * YAML parsing fails
    /// * Required nodes are missing
    /// * Array lengths are insufficient
    /// * Values cannot be parsed as expected types
    pub fn parse_yaml_camera(
        path: &str,
        min_intrinsics_len: usize,
    ) -> Result<(Intrinsics, Resolution, Vec<f64>), CameraModelError> {
        let contents = fs::read_to_string(path)?;
        let docs = YamlLoader::load_from_str(&contents)?;

        if docs.is_empty() {
            return Err(CameraModelError::InvalidParams(
                "Empty YAML document".to_string(),
            ));
        }

        let doc = &docs[0];
        let cam_node = &doc["cam0"];

        if cam_node.is_badvalue() {
            return Err(CameraModelError::InvalidParams(
                "Missing 'cam0' node in YAML".to_string(),
            ));
        }

        // Parse intrinsics array
        let intrinsics_yaml = cam_node["intrinsics"].as_vec().ok_or_else(|| {
            CameraModelError::InvalidParams(
                "YAML missing 'intrinsics' array under 'cam0'".to_string(),
            )
        })?;

        if intrinsics_yaml.len() < min_intrinsics_len {
            return Err(CameraModelError::InvalidParams(format!(
                "Intrinsics array must have at least {} elements, got {}",
                min_intrinsics_len,
                intrinsics_yaml.len()
            )));
        }

        // Parse resolution array
        let resolution_yaml = cam_node["resolution"].as_vec().ok_or_else(|| {
            CameraModelError::InvalidParams(
                "YAML missing 'resolution' array under 'cam0'".to_string(),
            )
        })?;

        if resolution_yaml.len() < 2 {
            return Err(CameraModelError::InvalidParams(
                "Resolution array must have at least 2 elements (width, height)".to_string(),
            ));
        }

        // Extract intrinsics (first 4 elements)
        let intrinsics = Intrinsics {
            fx: intrinsics_yaml[0].as_f64().ok_or_else(|| {
                CameraModelError::InvalidParams("Invalid fx: not a float".to_string())
            })?,
            fy: intrinsics_yaml[1].as_f64().ok_or_else(|| {
                CameraModelError::InvalidParams("Invalid fy: not a float".to_string())
            })?,
            cx: intrinsics_yaml[2].as_f64().ok_or_else(|| {
                CameraModelError::InvalidParams("Invalid cx: not a float".to_string())
            })?,
            cy: intrinsics_yaml[3].as_f64().ok_or_else(|| {
                CameraModelError::InvalidParams("Invalid cy: not a float".to_string())
            })?,
        };

        // Extract resolution
        let resolution = Resolution {
            width: resolution_yaml[0].as_i64().ok_or_else(|| {
                CameraModelError::InvalidParams("Invalid width: not an integer".to_string())
            })? as u32,
            height: resolution_yaml[1].as_i64().ok_or_else(|| {
                CameraModelError::InvalidParams("Invalid height: not an integer".to_string())
            })? as u32,
        };

        // Extract extra parameters (beyond first 4)
        let mut extra_params = Vec::new();
        for (i, param_yaml) in intrinsics_yaml.iter().enumerate().skip(4) {
            let param = param_yaml.as_f64().ok_or_else(|| {
                CameraModelError::InvalidParams(format!(
                    "Invalid parameter at index {}: not a float",
                    i
                ))
            })?;
            extra_params.push(param);
        }

        Ok((intrinsics, resolution, extra_params))
    }

    /// Saves camera model parameters to a YAML file in standard format.
    ///
    /// This helper creates a YAML file with the structure:
    /// ```yaml
    /// cam0:
    ///   camera_model: <model_name>
    ///   intrinsics: [fx, fy, cx, cy, ...extra_params]
    ///   resolution: [width, height]
    /// ```
    ///
    /// # Arguments
    ///
    /// * `path` - Path where the YAML file will be saved
    /// * `model_name` - Name of the camera model (e.g., "pinhole", "double_sphere")
    /// * `intrinsics` - The intrinsic parameters
    /// * `resolution` - The image resolution
    /// * `extra_params` - Additional parameters to append to intrinsics array
    ///
    /// # Errors
    ///
    /// Returns `CameraModelError` if:
    /// * YAML serialization fails
    /// * File creation/writing fails
    pub fn save_yaml_camera(
        path: &str,
        model_name: &str,
        intrinsics: &Intrinsics,
        resolution: &Resolution,
        extra_params: &[f64],
    ) -> Result<(), CameraModelError> {
        use serde_yaml;

        // Build intrinsics array: [fx, fy, cx, cy, ...extra]
        let mut intrinsics_vec = vec![intrinsics.fx, intrinsics.fy, intrinsics.cx, intrinsics.cy];
        intrinsics_vec.extend_from_slice(extra_params);

        // Build YAML structure
        let yaml = serde_yaml::to_value(serde_yaml::Mapping::from_iter([(
            serde_yaml::Value::String("cam0".to_string()),
            serde_yaml::to_value(serde_yaml::Mapping::from_iter([
                (
                    serde_yaml::Value::String("camera_model".to_string()),
                    serde_yaml::Value::String(model_name.to_string()),
                ),
                (
                    serde_yaml::Value::String("intrinsics".to_string()),
                    serde_yaml::to_value(intrinsics_vec)
                        .map_err(|e| CameraModelError::YamlError(e.to_string()))?,
                ),
                (
                    serde_yaml::Value::String("resolution".to_string()),
                    serde_yaml::to_value(vec![resolution.width, resolution.height])
                        .map_err(|e| CameraModelError::YamlError(e.to_string()))?,
                ),
            ]))
            .map_err(|e| CameraModelError::YamlError(e.to_string()))?,
        )]))
        .map_err(|e| CameraModelError::YamlError(e.to_string()))?;

        // Convert to string and write to file
        let yaml_string =
            serde_yaml::to_string(&yaml).map_err(|e| CameraModelError::YamlError(e.to_string()))?;

        // Create parent directory if it doesn't exist
        if let Some(parent) = std::path::Path::new(path).parent() {
            fs::create_dir_all(parent).map_err(|e| CameraModelError::IOError(e.to_string()))?;
        }

        let mut file =
            fs::File::create(path).map_err(|e| CameraModelError::IOError(e.to_string()))?;

        file.write_all(yaml_string.as_bytes())
            .map_err(|e| CameraModelError::IOError(e.to_string()))?;

        Ok(())
    }
}

/// Contains unit tests for the camera module.
#[cfg(test)]
mod tests {
    use super::*;
    use nalgebra::DVector;

    #[test]
    fn test_camera_model_names() {
        // Test Double Sphere model name
        let ds_params = DVector::from_vec(vec![350.0, 350.0, 320.0, 240.0, 0.58, -0.18]);
        let ds_model = double_sphere::DoubleSphereModel::new(&ds_params).unwrap();
        assert_eq!(ds_model.get_model_name(), "double_sphere");

        // Test EUCM model name
        let eucm_params = DVector::from_vec(vec![350.0, 350.0, 320.0, 240.0, 1.0, 0.5]);
        let eucm_model = eucm::EucmModel::new(&eucm_params).unwrap();
        assert_eq!(eucm_model.get_model_name(), "eucm");

        // Test FOV model name
        let fov_params = DVector::from_vec(vec![379.045, 379.008, 505.512, 509.969, 0.9259487501905697]);
        let fov_model = fov::FovModel::new(&fov_params).unwrap();
        assert_eq!(fov_model.get_model_name(), "fov");

        // Test Kannala-Brandt model name
        let kb_params =
            DVector::from_vec(vec![460.0, 460.0, 320.0, 240.0, -0.01, 0.05, -0.08, 0.04]);
        let kb_model = kannala_brandt::KannalaBrandtModel::new(&kb_params).unwrap();
        assert_eq!(kb_model.get_model_name(), "kannala_brandt");

        // Test Pinhole model name
        let pinhole_params = DVector::from_vec(vec![460.0, 460.0, 320.0, 240.0]);
        let pinhole_model = pinhole::PinholeModel::new(&pinhole_params).unwrap();
        assert_eq!(pinhole_model.get_model_name(), "pinhole");

        // Test RadTan model name
        let radtan_params = DVector::from_vec(vec![
            460.0, 460.0, 320.0, 240.0, -0.28, 0.07, 0.0002, 0.00002, 0.0,
        ]);
        let radtan_model = rad_tan::RadTanModel::new(&radtan_params).unwrap();
        assert_eq!(radtan_model.get_model_name(), "rad_tan");

        // Test UCM model name
        let ucm_params = DVector::from_vec(vec![350.0, 350.0, 320.0, 240.0, 0.8]);
        let ucm_model = ucm::UcmModel::new(&ucm_params).unwrap();
        assert_eq!(ucm_model.get_model_name(), "ucm");
    }

    #[test]
    fn test_validate_projection_bounds() {
        let resolution = Resolution {
            width: 640,
            height: 480,
        };

        // Valid points
        assert!(validate_projection_bounds(0.0, 0.0, &resolution).is_ok());
        assert!(validate_projection_bounds(320.0, 240.0, &resolution).is_ok());
        assert!(validate_projection_bounds(639.0, 479.0, &resolution).is_ok());

        // Invalid points
        assert!(validate_projection_bounds(-1.0, 240.0, &resolution).is_err());
        assert!(validate_projection_bounds(640.0, 240.0, &resolution).is_err());
        assert!(validate_projection_bounds(320.0, -1.0, &resolution).is_err());
        assert!(validate_projection_bounds(320.0, 480.0, &resolution).is_err());
        assert!(validate_projection_bounds(1000.0, 1000.0, &resolution).is_err());
    }

    #[test]
    fn test_validate_unprojection_bounds() {
        let resolution = Resolution {
            width: 640,
            height: 480,
        };

        // Valid points
        assert!(validate_unprojection_bounds(&Vector2::new(0.0, 0.0), &resolution).is_ok());
        assert!(validate_unprojection_bounds(&Vector2::new(320.0, 240.0), &resolution).is_ok());
        assert!(validate_unprojection_bounds(&Vector2::new(639.0, 479.0), &resolution).is_ok());

        // Invalid points
        assert!(validate_unprojection_bounds(&Vector2::new(-1.0, 240.0), &resolution).is_err());
        assert!(validate_unprojection_bounds(&Vector2::new(640.0, 240.0), &resolution).is_err());
        assert!(validate_unprojection_bounds(&Vector2::new(320.0, -1.0), &resolution).is_err());
        assert!(validate_unprojection_bounds(&Vector2::new(320.0, 480.0), &resolution).is_err());
    }

    #[test]
    fn test_validate_point_in_front() {
        // Valid z values
        assert!(validate_point_in_front(1.0).is_ok());
        assert!(validate_point_in_front(0.1).is_ok());
        assert!(validate_point_in_front(0.001).is_ok());

        // Invalid z values
        assert!(validate_point_in_front(0.0).is_err());
        assert!(validate_point_in_front(-1.0).is_err());
        assert!(validate_point_in_front(-0.001).is_err());
        assert!(validate_point_in_front(1e-10).is_err()); // Too close to zero
    }
}
