//! Double Sphere projection factor for apex-solver optimization.
//!
//! This module provides a factor implementation for the apex-solver framework
//! that computes reprojection errors and analytical Jacobians for the Double Sphere
//! camera model. This allows using apex-solver's Levenberg-Marquardt optimizer
//! with hand-derived analytical derivatives.

use apex_solver::core::factors::Factor;
use nalgebra::{DMatrix, DVector, Vector2, Vector3};

/// Projection factor for Double Sphere camera model optimization with apex-solver.
///
/// This factor computes the reprojection error between observed 2D points and
/// the projection of 3D points using the Double Sphere camera model. It provides
/// analytical Jacobians for efficient optimization.
///
/// # Residual Formulation
///
/// For each 3D-2D point correspondence, the residual is computed as:
/// ```text
/// residual_x = fx * x - (u - cx) * denom
/// residual_y = fy * y - (v - cy) * denom
/// ```
///
/// where `denom = alpha * d2 + (1 - alpha) * gamma` from the Double Sphere model.
///
/// # Parameters
///
/// The factor optimizes 6 camera parameters: `[fx, fy, cx, cy, alpha, xi]`
#[derive(Debug, Clone)]
pub struct DoubleSphereProjectionFactor {
    /// 3D points in camera coordinate system
    pub points_3d: Vec<Vector3<f64>>,
    /// Corresponding observed 2D points in image coordinates
    pub points_2d: Vec<Vector2<f64>>,
}

impl DoubleSphereProjectionFactor {
    /// Creates a new Double Sphere projection factor.
    ///
    /// # Arguments
    ///
    /// * `points_3d` - Vector of 3D points in camera coordinates
    /// * `points_2d` - Vector of corresponding 2D observed points
    ///
    /// # Panics
    ///
    /// Panics if the number of 3D and 2D points don't match.
    pub fn new(points_3d: Vec<Vector3<f64>>, points_2d: Vec<Vector2<f64>>) -> Self {
        assert_eq!(
            points_3d.len(),
            points_2d.len(),
            "Number of 3D and 2D points must match"
        );
        Self {
            points_3d,
            points_2d,
        }
    }

    /// Compute residual and analytical Jacobian for a single point.
    ///
    /// # Arguments
    ///
    /// * `point_3d` - 3D point in camera coordinates
    /// * `point_2d` - Observed 2D point
    /// * `fx, fy, cx, cy` - Intrinsic parameters
    /// * `alpha, xi` - Double Sphere distortion parameters
    /// * `compute_jacobian` - Whether to compute the Jacobian
    ///
    /// # Returns
    ///
    /// Tuple of (residual_vector, optional_jacobian_matrix)
    #[inline]
    fn compute_point_residual_jacobian(
        point_3d: &Vector3<f64>,
        point_2d: &Vector2<f64>,
        fx: f64,
        fy: f64,
        cx: f64,
        cy: f64,
        alpha: f64,
        xi: f64,
        compute_jacobian: bool,
    ) -> (Vector2<f64>, Option<nalgebra::Matrix2x6<f64>>) {
        const PRECISION: f64 = 1e-3;

        let x = point_3d.x;
        let y = point_3d.y;
        let z = point_3d.z;

        let r_squared = x * x + y * y;
        let d1 = (r_squared + z * z).sqrt();
        let gamma = xi * d1 + z;
        let d2 = (r_squared + gamma * gamma).sqrt();
        let m_alpha = 1.0 - alpha;

        let denom = alpha * d2 + m_alpha * gamma;

        // Check projection validity
        let w1 = if alpha <= 0.5 {
            alpha / m_alpha
        } else {
            m_alpha / alpha
        };
        let w2 = (w1 + xi) / (2.0 * w1 * xi + xi * xi + 1.0).sqrt();
        let check_projection = z > -w2 * d1;

        if denom < PRECISION || !check_projection {
            // Invalid projection - return large residual
            let residual = Vector2::new(1e6, 1e6);
            let jacobian = if compute_jacobian {
                Some(nalgebra::Matrix2x6::zeros())
            } else {
                None
            };
            return (residual, jacobian);
        }

        // Compute residual using C++ formulation
        let u_cx = point_2d.x - cx;
        let v_cy = point_2d.y - cy;

        let residual = Vector2::new(fx * x - u_cx * denom, fy * y - v_cy * denom);

        // Compute analytical Jacobian if requested
        let jacobian = if compute_jacobian {
            let mut jac = nalgebra::Matrix2x6::zeros();

            // ∂residual / ∂fx
            jac[(0, 0)] = x;
            jac[(1, 0)] = 0.0;

            // ∂residual / ∂fy
            jac[(0, 1)] = 0.0;
            jac[(1, 1)] = y;

            // ∂residual / ∂cx
            jac[(0, 2)] = denom;
            jac[(1, 2)] = 0.0;

            // ∂residual / ∂cy
            jac[(0, 3)] = 0.0;
            jac[(1, 3)] = denom;

            // ∂residual / ∂alpha
            jac[(0, 4)] = (gamma - d2) * u_cx;
            jac[(1, 4)] = (gamma - d2) * v_cy;

            // ∂residual / ∂xi
            let coeff = (alpha * d1 * gamma) / d2 + (m_alpha * d1);
            jac[(0, 5)] = -u_cx * coeff;
            jac[(1, 5)] = -v_cy * coeff;

            Some(jac)
        } else {
            None
        };

        (residual, jacobian)
    }
}

impl Factor for DoubleSphereProjectionFactor {
    /// Compute residuals and analytical Jacobians for all point correspondences.
    ///
    /// # Arguments
    ///
    /// * `params` - Slice containing camera parameters as a single DVector:
    ///              `params[0] = [fx, fy, cx, cy, alpha, xi]`
    /// * `compute_jacobian` - Whether to compute the Jacobian matrix
    ///
    /// # Returns
    ///
    /// Tuple of (residual_vector, optional_jacobian_matrix) where:
    /// - `residual_vector` has dimension `2 * num_points`
    /// - `jacobian_matrix` has dimension `(2 * num_points) × 6`
    fn linearize(
        &self,
        params: &[DVector<f64>],
        compute_jacobian: bool,
    ) -> (DVector<f64>, Option<DMatrix<f64>>) {
        // Extract camera parameters
        let cam_params = &params[0];
        let fx = cam_params[0];
        let fy = cam_params[1];
        let cx = cam_params[2];
        let cy = cam_params[3];
        let alpha = cam_params[4];
        let xi = cam_params[5];

        let num_points = self.points_2d.len();
        let residual_dim = num_points * 2;

        // Initialize residual vector
        let mut residuals = DVector::zeros(residual_dim);

        // Initialize Jacobian if needed
        let mut jacobian_matrix = if compute_jacobian {
            Some(DMatrix::zeros(residual_dim, 6))
        } else {
            None
        };

        // Process each point
        for i in 0..num_points {
            let (point_residual, point_jacobian) = Self::compute_point_residual_jacobian(
                &self.points_3d[i],
                &self.points_2d[i],
                fx,
                fy,
                cx,
                cy,
                alpha,
                xi,
                compute_jacobian,
            );

            // Fill residual vector
            residuals[i * 2] = point_residual[0];
            residuals[i * 2 + 1] = point_residual[1];

            // Fill Jacobian matrix if computed
            if let (Some(ref mut jac_matrix), Some(point_jac)) =
                (jacobian_matrix.as_mut(), point_jacobian)
            {
                jac_matrix
                    .view_mut((i * 2, 0), (2, 6))
                    .copy_from(&point_jac);
            }
        }

        (residuals, jacobian_matrix)
    }

    /// Returns the dimension of the residual vector.
    ///
    /// For N point correspondences, the residual dimension is 2N
    /// (2 residuals per point: x and y).
    fn get_dimension(&self) -> usize {
        self.points_2d.len() * 2
    }
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn test_factor_creation() {
        let points_3d = vec![
            Vector3::new(0.0, 0.0, 1.0),
            Vector3::new(0.1, 0.0, 1.0),
            Vector3::new(0.0, 0.1, 1.0),
        ];
        let points_2d = vec![
            Vector2::new(320.0, 240.0),
            Vector2::new(350.0, 240.0),
            Vector2::new(320.0, 270.0),
        ];

        let factor = DoubleSphereProjectionFactor::new(points_3d, points_2d);
        assert_eq!(factor.get_dimension(), 6); // 3 points × 2 residuals
    }

    #[test]
    fn test_linearize_dimensions() {
        let points_3d = vec![Vector3::new(0.0, 0.0, 1.0), Vector3::new(0.1, 0.0, 1.0)];
        let points_2d = vec![Vector2::new(320.0, 240.0), Vector2::new(350.0, 240.0)];

        let factor = DoubleSphereProjectionFactor::new(points_3d, points_2d);

        // Camera parameters: [fx, fy, cx, cy, alpha, xi]
        let params = vec![DVector::from_vec(vec![
            300.0, 300.0, 320.0, 240.0, 0.5, 0.1,
        ])];

        let (residual, jacobian) = factor.linearize(&params, true);

        assert_eq!(residual.len(), 4); // 2 points × 2 residuals
        assert!(jacobian.is_some());
        let jac = jacobian.unwrap();
        assert_eq!(jac.nrows(), 4); // 2 points × 2 residuals
        assert_eq!(jac.ncols(), 6); // 6 camera parameters
    }

    #[test]
    fn test_residual_computation() {
        // Test with a simple case where 3D point at (0,0,1) should project to (cx,cy)
        let points_3d = vec![Vector3::new(0.0, 0.0, 1.0)];
        let points_2d = vec![Vector2::new(320.0, 240.0)];

        let factor = DoubleSphereProjectionFactor::new(points_3d, points_2d);

        // Parameters with perfect projection at center
        let params = vec![DVector::from_vec(vec![
            300.0, 300.0, 320.0, 240.0, 0.5, 0.1,
        ])];

        let (residual, _) = factor.linearize(&params, false);

        // For point at (0,0,1) with the given parameters, residual should be small
        assert!(residual[0].abs() < 1.0);
        assert!(residual[1].abs() < 1.0);
    }

    #[test]
    fn test_jacobian_non_zero() {
        let points_3d = vec![Vector3::new(0.1, 0.1, 1.0)];
        let points_2d = vec![Vector2::new(330.0, 250.0)];

        let factor = DoubleSphereProjectionFactor::new(points_3d, points_2d);

        let params = vec![DVector::from_vec(vec![
            300.0, 300.0, 320.0, 240.0, 0.5, 0.1,
        ])];

        let (_, jacobian) = factor.linearize(&params, true);

        assert!(jacobian.is_some());
        let jac = jacobian.unwrap();

        // Check that Jacobian has non-zero entries
        let has_nonzero = jac.iter().any(|&x| x.abs() > 1e-10);
        assert!(has_nonzero, "Jacobian should have non-zero entries");
    }

    #[test]
    #[should_panic(expected = "Number of 3D and 2D points must match")]
    fn test_mismatched_points_panic() {
        let points_3d = vec![Vector3::new(0.0, 0.0, 1.0)];
        let points_2d = vec![Vector2::new(320.0, 240.0), Vector2::new(330.0, 250.0)];

        DoubleSphereProjectionFactor::new(points_3d, points_2d);
    }
}
