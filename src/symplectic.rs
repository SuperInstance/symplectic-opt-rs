//! Symplectic matrix operations.
//!
//! A matrix S ∈ R^{2n×2n} is symplectic if S^T J S = J, where J is the canonical
//! symplectic form. Symplectic matrices preserve the symplectic 2-form and therefore
//! phase space volume (Liouville's theorem).

use nalgebra::DMatrix;
use std::fmt;

/// Wrapper around a 2n×2n matrix that is known (or assumed) to be symplectic.
#[derive(Clone, Debug)]
pub struct SymplecticMatrix {
    matrix: DMatrix<f64>,
    n: usize, // half-dimension
}

impl SymplecticMatrix {
    /// Create a new symplectic matrix wrapper.
    ///
    /// Does **not** verify symplecticity — use [`is_symplectic`] to check.
    pub fn new(matrix: DMatrix<f64>) -> Self {
        let rows = matrix.nrows();
        assert_eq!(rows, matrix.ncols(), "Symplectic matrices must be square");
        assert_eq!(rows % 2, 0, "Symplectic matrices must have even dimension");
        Self {
            matrix,
            n: rows / 2,
        }
    }

    /// The half-dimension n (matrix is 2n × 2n).
    pub fn n(&self) -> usize {
        self.n
    }

    /// The full dimension 2n.
    pub fn dim(&self) -> usize {
        2 * self.n
    }

    /// Access the underlying matrix.
    pub fn as_matrix(&self) -> &DMatrix<f64> {
        &self.matrix
    }

    /// Compute the symplectic inverse (which equals -J S^T J for any symplectic S).
    pub fn inverse(&self) -> DMatrix<f64> {
        symplectic_inverse(&self.matrix)
    }

    /// Verify this matrix is symplectic to within `tol`.
    pub fn verify(&self, tol: f64) -> bool {
        is_symplectic(&self.matrix, tol)
    }
}

impl fmt::Display for SymplecticMatrix {
    fn fmt(&self, f: &mut fmt::Formatter<'_>) -> fmt::Result {
        write!(f, "SymplecticMatrix({}x{}):\n{}", self.dim(), self.dim(), self.matrix)
    }
}

/// The canonical symplectic form J = [0, I; -I, 0] of dimension 2n.
///
/// # Panics
/// Panics if `n` is zero.
pub fn canonical_form(n: usize) -> DMatrix<f64> {
    assert!(n > 0, "dimension n must be positive");
    let dim = 2 * n;
    let mut j = DMatrix::zeros(dim, dim);
    // Top-right block: I_n
    for i in 0..n {
        j[(i, n + i)] = 1.0;
    }
    // Bottom-left block: -I_n
    for i in 0..n {
        j[(n + i, i)] = -1.0;
    }
    j
}

/// Check whether a 2n×2n matrix is symplectic: S^T J S ≈ J.
///
/// Returns `true` if ‖S^T J S − J‖_max < `tol`.
pub fn is_symplectic(m: &DMatrix<f64>, tol: f64) -> bool {
    let rows = m.nrows();
    if rows != m.ncols() || rows % 2 != 0 {
        return false;
    }
    let n = rows / 2;
    let j = canonical_form(n);
    let product = m.transpose() * &j * m;
    let diff = &product - &j;
    diff.iter().all(|&v| v.abs() < tol)
}

/// Compute the symplectic inverse: S^{-1} = -J S^T J.
///
/// For a symplectic matrix S, this equals the standard matrix inverse but can
/// be computed more efficiently.
pub fn symplectic_inverse(m: &DMatrix<f64>) -> DMatrix<f64> {
    let rows = m.nrows();
    assert_eq!(rows, m.ncols(), "Matrix must be square");
    assert_eq!(rows % 2, 0, "Matrix must have even dimension");
    let n = rows / 2;
    let j = canonical_form(n);
    let neg_j = &j * -1.0;
    // S^{-1} = -J S^T J = (-J)(S^T)(J)
    neg_j * m.transpose() * &j
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn test_identity_is_symplectic() {
        let id = DMatrix::identity(4, 4);
        assert!(is_symplectic(&id, 1e-10));
    }

    #[test]
    fn test_canonical_form_j_squared() {
        let j = canonical_form(2);
        let j2 = &j * &j;
        let neg_id = DMatrix::from_diagonal_element(4, 4, -1.0);
        assert!((&j2 - &neg_id).iter().all(|&v| v.abs() < 1e-12));
    }

    #[test]
    fn test_symplectic_inverse_j() {
        // J^{-1} = -J for the canonical form
        let j = canonical_form(3);
        let j_inv = symplectic_inverse(&j);
        let neg_j = -&j;
        assert!((&j_inv - &neg_j).iter().all(|&v| v.abs() < 1e-12),
            "J inverse should equal -J");
    }

    #[test]
    fn test_non_symplectic_matrix() {
        let mut m = DMatrix::identity(4, 4);
        m[(0, 0)] = 2.0; // perturbation
        assert!(!is_symplectic(&m, 1e-6));
    }
}
