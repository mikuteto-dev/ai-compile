#include "aic/tensor.hpp"

#include <algorithm>
#include <cmath>
#include <limits>
#include <random>
#include <stdexcept>
#include <utility>

#ifdef __APPLE__
#ifndef ACCELERATE_NEW_LAPACK
#define ACCELERATE_NEW_LAPACK 1
#endif
#include <Accelerate/Accelerate.h>
#endif

namespace aic {

namespace {

void check_matrix_storage(const Matrix& A) {
    if (A.data.size() != A.rows * A.cols) {
        throw std::invalid_argument("Matrix storage size does not match rows * cols");
    }
}

std::vector<SVDMode> hestenes_jacobi_svd_tall(const Matrix& A, int max_sweeps = 45) {
    const std::size_t m = A.rows;
    const std::size_t n = A.cols;
    if (m == 0 || n == 0) {
        return {};
    }
    if (m < n) {
        throw std::invalid_argument("Expected rows >= cols in hestenes_jacobi_svd_tall");
    }

    std::vector<Vector> b_cols(n, Vector(m, 0.0));
    std::vector<Vector> v_cols(n, Vector(n, 0.0));
    for (std::size_t c = 0; c < n; ++c) {
        for (std::size_t r = 0; r < m; ++r) {
            b_cols[c][r] = A(r, c);
        }
        v_cols[c][c] = 1.0;
    }

    for (int sweep = 0; sweep < max_sweeps; ++sweep) {
        bool converged = true;
        for (std::size_t p = 0; p + 1 < n; ++p) {
            Vector& bp = b_cols[p];
            Vector& vp = v_cols[p];
            for (std::size_t q = p + 1; q < n; ++q) {
                Vector& bq = b_cols[q];
                Vector& vq = v_cols[q];

                double alpha = 0.0;
                double beta = 0.0;
                double gamma = 0.0;
                for (std::size_t i = 0; i < m; ++i) {
                    alpha += bp[i] * bp[i];
                    beta += bq[i] * bq[i];
                    gamma += bp[i] * bq[i];
                }
                if (alpha == 0.0 || beta == 0.0) {
                    continue;
                }
                if (std::abs(gamma) <= 1e-14 * std::sqrt(alpha * beta)) {
                    continue;
                }
                converged = false;
                const double zeta = (beta - alpha) / (2.0 * gamma);
                const double t = std::copysign(1.0, zeta) / (std::abs(zeta) + std::sqrt(1.0 + zeta * zeta));
                const double c = 1.0 / std::sqrt(1.0 + t * t);
                const double s = t * c;

                for (std::size_t i = 0; i < m; ++i) {
                    const double bpi = bp[i];
                    const double bqi = bq[i];
                    bp[i] = c * bpi - s * bqi;
                    bq[i] = s * bpi + c * bqi;
                }
                for (std::size_t i = 0; i < n; ++i) {
                    const double vpi = vp[i];
                    const double vqi = vq[i];
                    vp[i] = c * vpi - s * vqi;
                    vq[i] = s * vpi + c * vqi;
                }
            }
        }
        if (converged) {
            break;
        }
    }

    std::vector<SVDMode> modes;
    modes.reserve(n);
    for (std::size_t j = 0; j < n; ++j) {
        const double sigma = norm2(b_cols[j]);
        if (sigma > 1e-12) {
            const double inv_s = 1.0 / sigma;
            Vector u(m);
            for (std::size_t i = 0; i < m; ++i) {
                u[i] = b_cols[j][i] * inv_s;
            }
            modes.push_back(SVDMode{sigma, std::move(u), v_cols[j]});
        }
    }
    std::sort(modes.begin(), modes.end(), [](const SVDMode& lhs, const SVDMode& rhs) {
        return lhs.sigma > rhs.sigma;
    });
    return modes;
}

Matrix rows_to_matrix(const std::vector<Vector>& rows) {
    if (rows.empty()) {
        return Matrix(0, 0);
    }
    const std::size_t r = rows.size();
    const std::size_t c = rows[0].size();
    Matrix out(r, c);
    for (std::size_t i = 0; i < r; ++i) {
        if (rows[i].size() != c) {
            throw std::invalid_argument("Inconsistent row sizes in rows_to_matrix");
        }
        for (std::size_t j = 0; j < c; ++j) {
            out(i, j) = rows[i][j];
        }
    }
    return out;
}

#ifdef __APPLE__
bool lapack_dgesdd(const Matrix& A_norm, std::size_t k_eff, std::vector<SVDMode>& out_modes) {
    const std::size_t m = A_norm.rows;
    const std::size_t n = A_norm.cols;
    const std::size_t r_min = std::min(m, n);
    if (r_min == 0 || k_eff == 0) {
        out_modes.clear();
        return true;
    }
    if (m > static_cast<std::size_t>(std::numeric_limits<__LAPACK_int>::max()) ||
        n > static_cast<std::size_t>(std::numeric_limits<__LAPACK_int>::max())) {
        return false;
    }

    std::vector<double> a_col(m * n);
    for (std::size_t r = 0; r < m; ++r) {
        for (std::size_t c = 0; c < n; ++c) {
            a_col[c * m + r] = A_norm(r, c);
        }
    }

    char jobz = 'S';
    __LAPACK_int m_lap = static_cast<__LAPACK_int>(m);
    __LAPACK_int n_lap = static_cast<__LAPACK_int>(n);
    __LAPACK_int lda = m_lap;
    __LAPACK_int ldu = m_lap;
    __LAPACK_int ldvt = static_cast<__LAPACK_int>(r_min);
    __LAPACK_int lwork = -1;
    __LAPACK_int info = 0;
    double work_opt = 0.0;

    std::vector<double> s_vals(r_min, 0.0);
    std::vector<double> u_col(m * r_min, 0.0);
    std::vector<double> vt_col(r_min * n, 0.0);
    std::vector<__LAPACK_int> iwork(8 * r_min, 0);

    dgesdd_(
        &jobz,
        &m_lap,
        &n_lap,
        a_col.data(),
        &lda,
        s_vals.data(),
        u_col.data(),
        &ldu,
        vt_col.data(),
        &ldvt,
        &work_opt,
        &lwork,
        iwork.data(),
        &info);
    if (info != 0 || work_opt <= 0.0) {
        return false;
    }

    lwork = static_cast<__LAPACK_int>(work_opt);
    std::vector<double> work(static_cast<std::size_t>(lwork), 0.0);
    dgesdd_(
        &jobz,
        &m_lap,
        &n_lap,
        a_col.data(),
        &lda,
        s_vals.data(),
        u_col.data(),
        &ldu,
        vt_col.data(),
        &ldvt,
        work.data(),
        &lwork,
        iwork.data(),
        &info);
    if (info != 0) {
        return false;
    }

    out_modes.clear();
    out_modes.reserve(k_eff);
    for (std::size_t i = 0; i < k_eff; ++i) {
        if (s_vals[i] <= 1e-12) {
            break;
        }
        Vector u_vec(m);
        for (std::size_t r = 0; r < m; ++r) {
            u_vec[r] = u_col[i * m + r];
        }
        Vector v_vec(n);
        for (std::size_t c = 0; c < n; ++c) {
            v_vec[c] = vt_col[c * r_min + i];
        }
        out_modes.push_back(SVDMode{s_vals[i], std::move(u_vec), std::move(v_vec)});
    }
    return true;
}
#endif

}

Matrix::Matrix(std::size_t r, std::size_t c, double fill)
    : rows(r), cols(c), data(r * c, fill) {}

double& Matrix::operator()(std::size_t r, std::size_t c) {
    if (r >= rows || c >= cols) {
        throw std::out_of_range("Matrix index out of range");
    }
    return data[r * cols + c];
}

double Matrix::operator()(std::size_t r, std::size_t c) const {
    if (r >= rows || c >= cols) {
        throw std::out_of_range("Matrix index out of range");
    }
    return data[r * cols + c];
}

Vector Matrix::row(std::size_t r) const {
    if (r >= rows) {
        throw std::out_of_range("Matrix row index out of range");
    }
    Vector out(cols);
    const std::size_t base = r * cols;
    for (std::size_t c = 0; c < cols; ++c) {
        out[c] = data[base + c];
    }
    return out;
}

void Matrix::set_row(std::size_t r, const Vector& v) {
    if (r >= rows) {
        throw std::out_of_range("Matrix row index out of range");
    }
    if (v.size() != cols) {
        throw std::invalid_argument("Row vector length mismatch in set_row");
    }
    const std::size_t base = r * cols;
    for (std::size_t c = 0; c < cols; ++c) {
        data[base + c] = v[c];
    }
}

Vector Matrix::col(std::size_t c) const {
    if (c >= cols) {
        throw std::out_of_range("Matrix col index out of range");
    }
    Vector out(rows);
    for (std::size_t r = 0; r < rows; ++r) {
        out[r] = data[r * cols + c];
    }
    return out;
}

void Matrix::set_col(std::size_t c, const Vector& v) {
    if (c >= cols) {
        throw std::out_of_range("Matrix col index out of range");
    }
    if (v.size() != rows) {
        throw std::invalid_argument("Column vector length mismatch in set_col");
    }
    for (std::size_t r = 0; r < rows; ++r) {
        data[r * cols + c] = v[r];
    }
}

Matrix zeros(std::size_t r, std::size_t c) {
    return Matrix(r, c, 0.0);
}

Matrix eye(std::size_t n) {
    Matrix out(n, n, 0.0);
    for (std::size_t i = 0; i < n; ++i) {
        out(i, i) = 1.0;
    }
    return out;
}

Vector vec_zeros(std::size_t n) {
    return Vector(n, 0.0);
}

double dot(const Vector& a, const Vector& b) {
    if (a.size() != b.size()) {
        throw std::invalid_argument("Vector length mismatch in dot");
    }
    if (a.empty()) {
        return 0.0;
    }
#ifdef __APPLE__
    if (a.size() <= static_cast<std::size_t>(std::numeric_limits<int>::max())) {
        return cblas_ddot(static_cast<int>(a.size()), a.data(), 1, b.data(), 1);
    }
#endif
    long double acc = 0.0;
    for (std::size_t i = 0; i < a.size(); ++i) {
        acc += static_cast<long double>(a[i]) * static_cast<long double>(b[i]);
    }
    return static_cast<double>(acc);
}

double norm2(const Vector& a) {
    if (a.empty()) {
        return 0.0;
    }
    double max_abs = 0.0;
    for (double x : a) {
        const double ax = std::abs(x);
        if (ax > max_abs) {
            max_abs = ax;
        }
    }
    if (max_abs == 0.0) {
        return 0.0;
    }
    const double inv = 1.0 / max_abs;
    long double sq_sum = 0.0;
    for (double x : a) {
        const long double scaled = static_cast<long double>(x * inv);
        sq_sum += scaled * scaled;
    }
    return max_abs * std::sqrt(static_cast<double>(sq_sum));
}

Vector normalize(const Vector& a, double eps) {
    const double n = norm2(a);
    if (n <= eps) {
        return Vector(a.size(), 0.0);
    }
    const double inv = 1.0 / n;
    Vector out(a.size());
    for (std::size_t i = 0; i < a.size(); ++i) {
        out[i] = a[i] * inv;
    }
    return out;
}

Vector vec_add(const Vector& a, const Vector& b) {
    if (a.size() != b.size()) {
        throw std::invalid_argument("Vector length mismatch in vec_add");
    }
    if (a.empty()) {
        return {};
    }
#ifdef __APPLE__
    if (a.size() <= static_cast<std::size_t>(std::numeric_limits<int>::max())) {
        Vector out = b;
        cblas_daxpy(static_cast<int>(a.size()), 1.0, a.data(), 1, out.data(), 1);
        return out;
    }
#endif
    Vector out(a.size());
    for (std::size_t i = 0; i < a.size(); ++i) {
        out[i] = a[i] + b[i];
    }
    return out;
}

Vector vec_sub(const Vector& a, const Vector& b) {
    if (a.size() != b.size()) {
        throw std::invalid_argument("Vector length mismatch in vec_sub");
    }
    if (a.empty()) {
        return {};
    }
#ifdef __APPLE__
    if (a.size() <= static_cast<std::size_t>(std::numeric_limits<int>::max())) {
        Vector out = a;
        cblas_daxpy(static_cast<int>(b.size()), -1.0, b.data(), 1, out.data(), 1);
        return out;
    }
#endif
    Vector out(a.size());
    for (std::size_t i = 0; i < a.size(); ++i) {
        out[i] = a[i] - b[i];
    }
    return out;
}

Vector vec_scale(const Vector& a, double s) {
    if (a.empty()) {
        return {};
    }
#ifdef __APPLE__
    if (a.size() <= static_cast<std::size_t>(std::numeric_limits<int>::max())) {
        Vector out = a;
        cblas_dscal(static_cast<int>(out.size()), s, out.data(), 1);
        return out;
    }
#endif
    Vector out(a.size());
    for (std::size_t i = 0; i < a.size(); ++i) {
        out[i] = a[i] * s;
    }
    return out;
}

Matrix outer(const Vector& u, const Vector& v) {
    Matrix out(u.size(), v.size(), 0.0);
    for (std::size_t i = 0; i < u.size(); ++i) {
        const double ui = u[i];
        for (std::size_t j = 0; j < v.size(); ++j) {
            out(i, j) = ui * v[j];
        }
    }
    return out;
}

Vector mat_vec(const Matrix& A, const Vector& x) {
    check_matrix_storage(A);
    if (A.cols != x.size()) {
        throw std::invalid_argument("Dimension mismatch in mat_vec");
    }
    if (A.rows == 0 || A.cols == 0) {
        return Vector(A.rows, 0.0);
    }
#ifdef __APPLE__
    if (A.rows <= static_cast<std::size_t>(std::numeric_limits<int>::max()) &&
        A.cols <= static_cast<std::size_t>(std::numeric_limits<int>::max())) {
        Vector out(A.rows, 0.0);
        cblas_dgemv(
            CblasRowMajor,
            CblasNoTrans,
            static_cast<int>(A.rows),
            static_cast<int>(A.cols),
            1.0,
            A.data.data(),
            static_cast<int>(A.cols),
            x.data(),
            1,
            0.0,
            out.data(),
            1);
        return out;
    }
#endif
    Vector out(A.rows, 0.0);
    for (std::size_t r = 0; r < A.rows; ++r) {
        long double acc = 0.0;
        const std::size_t base = r * A.cols;
        for (std::size_t c = 0; c < A.cols; ++c) {
            acc += static_cast<long double>(A.data[base + c]) * static_cast<long double>(x[c]);
        }
        out[r] = static_cast<double>(acc);
    }
    return out;
}

Matrix mat_mul(const Matrix& A, const Matrix& B) {
    check_matrix_storage(A);
    check_matrix_storage(B);
    if (A.cols != B.rows) {
        throw std::invalid_argument("Dimension mismatch in mat_mul");
    }
    if (A.rows == 0 || A.cols == 0 || B.cols == 0) {
        return Matrix(A.rows, B.cols, 0.0);
    }
#ifdef __APPLE__
    if (A.rows <= static_cast<std::size_t>(std::numeric_limits<int>::max()) &&
        A.cols <= static_cast<std::size_t>(std::numeric_limits<int>::max()) &&
        B.cols <= static_cast<std::size_t>(std::numeric_limits<int>::max())) {
        Matrix out(A.rows, B.cols, 0.0);
        cblas_dgemm(
            CblasRowMajor,
            CblasNoTrans,
            CblasNoTrans,
            static_cast<int>(A.rows),
            static_cast<int>(B.cols),
            static_cast<int>(A.cols),
            1.0,
            A.data.data(),
            static_cast<int>(A.cols),
            B.data.data(),
            static_cast<int>(B.cols),
            0.0,
            out.data.data(),
            static_cast<int>(B.cols));
        return out;
    }
#endif
    Matrix out(A.rows, B.cols, 0.0);
    for (std::size_t i = 0; i < A.rows; ++i) {
        const std::size_t a_base = i * A.cols;
        const std::size_t c_base = i * B.cols;
        for (std::size_t k = 0; k < A.cols; ++k) {
            const double aik = A.data[a_base + k];
            if (aik == 0.0) {
                continue;
            }
            const std::size_t b_base = k * B.cols;
            for (std::size_t j = 0; j < B.cols; ++j) {
                out.data[c_base + j] += aik * B.data[b_base + j];
            }
        }
    }
    return out;
}

Matrix transpose(const Matrix& A) {
    check_matrix_storage(A);
    Matrix out(A.cols, A.rows, 0.0);
    for (std::size_t r = 0; r < A.rows; ++r) {
        for (std::size_t c = 0; c < A.cols; ++c) {
            out(c, r) = A(r, c);
        }
    }
    return out;
}

Matrix mat_add(const Matrix& A, const Matrix& B) {
    check_matrix_storage(A);
    check_matrix_storage(B);
    if (A.rows != B.rows || A.cols != B.cols) {
        throw std::invalid_argument("Shape mismatch in mat_add");
    }
    Matrix out(A.rows, A.cols);
    for (std::size_t i = 0; i < A.data.size(); ++i) {
        out.data[i] = A.data[i] + B.data[i];
    }
    return out;
}

Matrix mat_sub(const Matrix& A, const Matrix& B) {
    check_matrix_storage(A);
    check_matrix_storage(B);
    if (A.rows != B.rows || A.cols != B.cols) {
        throw std::invalid_argument("Shape mismatch in mat_sub");
    }
    Matrix out(A.rows, A.cols);
    for (std::size_t i = 0; i < A.data.size(); ++i) {
        out.data[i] = A.data[i] - B.data[i];
    }
    return out;
}

Matrix mat_scale(const Matrix& A, double s) {
    check_matrix_storage(A);
    Matrix out(A.rows, A.cols);
    for (std::size_t i = 0; i < A.data.size(); ++i) {
        out.data[i] = A.data[i] * s;
    }
    return out;
}

double frob_norm(const Matrix& A) {
    check_matrix_storage(A);
    double max_abs = 0.0;
    for (double val : A.data) {
        const double av = std::abs(val);
        if (av > max_abs) {
            max_abs = av;
        }
    }
    if (max_abs == 0.0) {
        return 0.0;
    }
    const double inv = 1.0 / max_abs;
    long double sq_sum = 0.0;
    for (double val : A.data) {
        const long double scaled = static_cast<long double>(val * inv);
        sq_sum += scaled * scaled;
    }
    return max_abs * std::sqrt(static_cast<double>(sq_sum));
}

Vector solve_linear(Matrix A, Vector b) {
    check_matrix_storage(A);
    if (A.rows != A.cols) {
        throw std::invalid_argument("Matrix A must be square in solve_linear");
    }
    const std::size_t n = A.rows;
    if (b.size() != n) {
        throw std::invalid_argument("Vector length mismatch in solve_linear");
    }
    if (n == 0) {
        return {};
    }

    double max_abs = 0.0;
    for (double v : A.data) {
        max_abs = std::max(max_abs, std::abs(v));
    }
    if (max_abs == 0.0) {
        throw std::runtime_error("Matrix is singular in solve_linear");
    }
    const double piv_tol = 1e-14 * max_abs;

#ifdef __APPLE__
    if (n <= static_cast<std::size_t>(std::numeric_limits<__LAPACK_int>::max())) {
        __LAPACK_int n_lap = static_cast<__LAPACK_int>(n);
        __LAPACK_int nrhs = 1;
        __LAPACK_int lda = n_lap;
        __LAPACK_int ldb = n_lap;
        __LAPACK_int info = 0;
        std::vector<__LAPACK_int> ipiv(n);
        std::vector<double> a_col(n * n);
        for (std::size_t r = 0; r < n; ++r) {
            for (std::size_t c = 0; c < n; ++c) {
                a_col[c * n + r] = A(r, c);
            }
        }
        dgesv_(&n_lap, &nrhs, a_col.data(), &lda, ipiv.data(), b.data(), &ldb, &info);
        if (info != 0) {
            throw std::runtime_error("Matrix is singular in solve_linear");
        }
        for (std::size_t i = 0; i < n; ++i) {
            if (std::abs(a_col[i * n + i]) <= piv_tol) {
                throw std::runtime_error("Matrix is singular or nearly singular in solve_linear");
            }
        }
        return b;
    }
#endif

    std::vector<std::size_t> perm(n);
    for (std::size_t i = 0; i < n; ++i) {
        perm[i] = i;
    }

    for (std::size_t col = 0; col < n; ++col) {
        std::size_t piv = col;
        double best = std::abs(A(perm[col], col));
        for (std::size_t r = col + 1; r < n; ++r) {
            const double cand = std::abs(A(perm[r], col));
            if (cand > best) {
                best = cand;
                piv = r;
            }
        }
        if (best <= piv_tol) {
            throw std::runtime_error("Matrix is singular or nearly singular in solve_linear");
        }
        if (piv != col) {
            std::swap(perm[col], perm[piv]);
        }

        const std::size_t pr = perm[col];
        const double piv_val = A(pr, col);
        for (std::size_t r = col + 1; r < n; ++r) {
            const std::size_t rr = perm[r];
            const double factor = A(rr, col) / piv_val;
            if (factor != 0.0) {
                A(rr, col) = 0.0;
                for (std::size_t c = col + 1; c < n; ++c) {
                    A(rr, c) -= factor * A(pr, c);
                }
                b[rr] -= factor * b[pr];
            }
        }
    }

    Vector x(n, 0.0);
    for (std::size_t idx = n; idx > 0; --idx) {
        const std::size_t r = idx - 1;
        const std::size_t rr = perm[r];
        long double acc = static_cast<long double>(b[rr]);
        for (std::size_t c = r + 1; c < n; ++c) {
            acc -= static_cast<long double>(A(rr, c)) * static_cast<long double>(x[c]);
        }
        x[r] = static_cast<double>(acc / static_cast<long double>(A(rr, r)));
    }
    return x;
}

Matrix solve_linear_multi(Matrix A, Matrix B) {
    check_matrix_storage(A);
    check_matrix_storage(B);
    if (A.rows != A.cols) {
        throw std::invalid_argument("Matrix A must be square in solve_linear_multi");
    }
    const std::size_t n = A.rows;
    const std::size_t m = B.cols;
    if (B.rows != n) {
        throw std::invalid_argument("Matrix B row mismatch in solve_linear_multi");
    }
    if (n == 0 || m == 0) {
        return Matrix(n, m, 0.0);
    }

#ifdef __APPLE__
    if (n <= static_cast<std::size_t>(std::numeric_limits<__LAPACK_int>::max()) &&
        m <= static_cast<std::size_t>(std::numeric_limits<__LAPACK_int>::max())) {
        __LAPACK_int n_lap = static_cast<__LAPACK_int>(n);
        __LAPACK_int nrhs = static_cast<__LAPACK_int>(m);
        __LAPACK_int lda = n_lap;
        __LAPACK_int ldb = n_lap;
        __LAPACK_int info = 0;
        std::vector<__LAPACK_int> ipiv(n);
        std::vector<double> a_col(n * n);
        std::vector<double> b_col(n * m);

        auto pack_ab = [&](double diag_damp) {
            for (std::size_t r = 0; r < n; ++r) {
                for (std::size_t c = 0; c < n; ++c) {
                    a_col[c * n + r] = A(r, c) + ((r == c) ? diag_damp : 0.0);
                }
            }
            for (std::size_t r = 0; r < n; ++r) {
                for (std::size_t c = 0; c < m; ++c) {
                    b_col[c * n + r] = B(r, c);
                }
            }
        };

        pack_ab(0.0);
        dgesv_(&n_lap, &nrhs, a_col.data(), &lda, ipiv.data(), b_col.data(), &ldb, &info);
        if (info > 0) {
            pack_ab(1e-8);
            dgesv_(&n_lap, &nrhs, a_col.data(), &lda, ipiv.data(), b_col.data(), &ldb, &info);
        }
        if (info != 0) {
            throw std::runtime_error("Matrix is singular in solve_linear_multi");
        }
        Matrix X(n, m, 0.0);
        for (std::size_t r = 0; r < n; ++r) {
            for (std::size_t c = 0; c < m; ++c) {
                X(r, c) = b_col[c * n + r];
            }
        }
        return X;
    }
#endif

    auto solve_dense_multi = [&](double diag_damp, Matrix& X_out) -> bool {
        Matrix mat = A;
        for (std::size_t i = 0; i < n; ++i) {
            mat(i, i) += diag_damp;
        }
        Matrix rhs = B;
        double max_abs = 0.0;
        for (double v : mat.data) {
            max_abs = std::max(max_abs, std::abs(v));
        }
        if (max_abs == 0.0) {
            return false;
        }
        const double piv_tol = 1e-14 * max_abs;
        std::vector<std::size_t> perm(n);
        for (std::size_t i = 0; i < n; ++i) {
            perm[i] = i;
        }
        for (std::size_t col = 0; col < n; ++col) {
            std::size_t piv = col;
            double best = std::abs(mat(perm[col], col));
            for (std::size_t r = col + 1; r < n; ++r) {
                const double cand = std::abs(mat(perm[r], col));
                if (cand > best) {
                    best = cand;
                    piv = r;
                }
            }
            if (best <= piv_tol) {
                return false;
            }
            if (piv != col) {
                std::swap(perm[col], perm[piv]);
            }
            const std::size_t pr = perm[col];
            const double piv_val = mat(pr, col);
            for (std::size_t r = col + 1; r < n; ++r) {
                const std::size_t rr = perm[r];
                const double factor = mat(rr, col) / piv_val;
                if (factor != 0.0) {
                    mat(rr, col) = 0.0;
                    for (std::size_t c = col + 1; c < n; ++c) {
                        mat(rr, c) -= factor * mat(pr, c);
                    }
                    for (std::size_t k = 0; k < m; ++k) {
                        rhs(rr, k) -= factor * rhs(pr, k);
                    }
                }
            }
        }
        X_out = Matrix(n, m, 0.0);
        for (std::size_t k = 0; k < m; ++k) {
            for (std::size_t idx = n; idx > 0; --idx) {
                const std::size_t r = idx - 1;
                const std::size_t rr = perm[r];
                long double acc = static_cast<long double>(rhs(rr, k));
                for (std::size_t c = r + 1; c < n; ++c) {
                    acc -= static_cast<long double>(mat(rr, c)) * static_cast<long double>(X_out(c, k));
                }
                X_out(r, k) = static_cast<double>(acc / static_cast<long double>(mat(rr, r)));
            }
        }
        return true;
    };

    Matrix X;
    if (solve_dense_multi(0.0, X)) {
        return X;
    }
    if (solve_dense_multi(1e-8, X)) {
        return X;
    }
    throw std::runtime_error("Matrix is singular in solve_linear_multi");
}

Matrix mat_inv(const Matrix& A) {
    check_matrix_storage(A);
    if (A.rows != A.cols) {
        throw std::invalid_argument("Matrix A must be square in mat_inv");
    }
    const std::size_t n = A.rows;
    if (n == 0) {
        return Matrix(0, 0);
    }

    double max_abs = 0.0;
    for (double v : A.data) {
        max_abs = std::max(max_abs, std::abs(v));
    }
    if (max_abs == 0.0) {
        throw std::runtime_error("Matrix is singular in mat_inv");
    }
    const double piv_tol = 1e-14 * max_abs;

    Matrix aug(n, 2 * n, 0.0);
    for (std::size_t r = 0; r < n; ++r) {
        for (std::size_t c = 0; c < n; ++c) {
            aug(r, c) = A(r, c);
        }
        aug(r, n + r) = 1.0;
    }

    for (std::size_t col = 0; col < n; ++col) {
        std::size_t piv = col;
        double best = std::abs(aug(col, col));
        for (std::size_t r = col + 1; r < n; ++r) {
            const double cand = std::abs(aug(r, col));
            if (cand > best) {
                best = cand;
                piv = r;
            }
        }
        if (best <= piv_tol) {
            throw std::runtime_error("Matrix is singular or nearly singular in mat_inv");
        }
        if (piv != col) {
            for (std::size_t c = 0; c < 2 * n; ++c) {
                std::swap(aug(col, c), aug(piv, c));
            }
        }

        const double inv_piv = 1.0 / aug(col, col);
        for (std::size_t c = 0; c < 2 * n; ++c) {
            aug(col, c) *= inv_piv;
        }
        aug(col, col) = 1.0;

        for (std::size_t r = 0; r < n; ++r) {
            if (r == col) {
                continue;
            }
            const double factor = aug(r, col);
            if (factor != 0.0) {
                for (std::size_t c = 0; c < 2 * n; ++c) {
                    aug(r, c) -= factor * aug(col, c);
                }
                aug(r, col) = 0.0;
            }
        }
    }

    Matrix inv(n, n, 0.0);
    for (std::size_t r = 0; r < n; ++r) {
        for (std::size_t c = 0; c < n; ++c) {
            inv(r, c) = aug(r, n + c);
        }
    }
    return inv;
}

std::vector<Vector> gram_schmidt(const std::vector<Vector>& vecs, double tol) {
    if (vecs.empty()) {
        return {};
    }
    const std::size_t dim = vecs[0].size();
    std::vector<Vector> basis;
    basis.reserve(vecs.size());
    for (const Vector& v : vecs) {
        if (v.size() != dim) {
            throw std::invalid_argument("Inconsistent vector lengths in gram_schmidt");
        }
        Vector w = v;
        for (int pass = 0; pass < 2; ++pass) {
            for (const Vector& q : basis) {
                const double proj = dot(w, q);
                if (proj != 0.0) {
                    for (std::size_t i = 0; i < dim; ++i) {
                        w[i] -= proj * q[i];
                    }
                }
            }
        }
        const double n = norm2(w);
        if (n > tol) {
            const double inv_n = 1.0 / n;
            for (double& x : w) {
                x *= inv_n;
            }
            basis.push_back(std::move(w));
        }
    }
    return basis;
}

std::vector<SVDMode> randomized_svd(
    const Matrix& A,
    std::size_t k,
    std::size_t oversample,
    int power_iters,
    std::uint64_t seed) {
    check_matrix_storage(A);
    const std::size_t m = A.rows;
    const std::size_t n = A.cols;
    if (k == 0 || m == 0 || n == 0) {
        return {};
    }
    const double scale = frob_norm(A);
    if (scale == 0.0) {
        return {};
    }

    const std::size_t k_eff = std::min({k, m, n});
    if (m < n) {
        std::vector<SVDMode> t_modes = randomized_svd(
            transpose(A), k_eff, oversample, power_iters, seed);
        for (SVDMode& mode : t_modes) {
            std::swap(mode.u, mode.v);
        }
        return t_modes;
    }

    const Matrix A_norm = mat_scale(A, 1.0 / scale);
    const std::size_t sample_cols = std::min(n, k_eff + oversample);

    std::mt19937_64 rng(seed);
    std::normal_distribution<double> dist(0.0, 1.0);
    Matrix Omega(n, sample_cols, 0.0);
    for (double& x : Omega.data) {
        x = dist(rng);
    }

    auto orthonormalize_cols = [](const Matrix& M) -> Matrix {
        std::vector<Vector> cols(M.cols);
        for (std::size_t c = 0; c < M.cols; ++c) {
            cols[c] = M.col(c);
        }
        const std::vector<Vector> ortho = gram_schmidt(cols, 1e-13);
        if (ortho.empty()) {
            return Matrix(M.rows, 0, 0.0);
        }
        return transpose(rows_to_matrix(ortho));
    };

    Matrix Y = mat_mul(A_norm, Omega);
    Matrix Q = orthonormalize_cols(Y);
    if (Q.cols == 0) {
        return {};
    }

    if (power_iters > 0) {
        const Matrix A_T = transpose(A_norm);
        for (int it = 0; it < power_iters; ++it) {
            const Matrix Z = mat_mul(A_T, Q);
            const Matrix Q_z = orthonormalize_cols(Z);
            if (Q_z.cols == 0) {
                return {};
            }
            Y = mat_mul(A_norm, Q_z);
            Q = orthonormalize_cols(Y);
            if (Q.cols == 0) {
                return {};
            }
        }
    }

    const Matrix B = mat_mul(transpose(Q), A_norm);
    const double b_scale = frob_norm(B);
    if (b_scale == 0.0) {
        return {};
    }
    const Matrix B_norm = mat_scale(B, 1.0 / b_scale);

    std::vector<SVDMode> b_modes;
    bool got_b = false;
#ifdef __APPLE__
    if (std::min(B_norm.rows, B_norm.cols) <= 512) {
        got_b = lapack_dgesdd(B_norm, std::min(k_eff, B_norm.rows), b_modes);
    }
#endif
    if (!got_b) {
        if (B_norm.rows >= B_norm.cols) {
            b_modes = hestenes_jacobi_svd_tall(B_norm);
        } else {
            b_modes = hestenes_jacobi_svd_tall(transpose(B_norm));
            for (SVDMode& mode : b_modes) {
                std::swap(mode.u, mode.v);
            }
        }
    }

    std::vector<SVDMode> out;
    out.reserve(std::min(k_eff, b_modes.size()));
    for (std::size_t i = 0; i < b_modes.size() && out.size() < k_eff; ++i) {
        const double sigma = b_modes[i].sigma * b_scale * scale;
        if (sigma <= 1e-12 * scale) {
            continue;
        }
        Vector u_full = normalize(mat_vec(Q, b_modes[i].u));
        out.push_back(SVDMode{sigma, std::move(u_full), std::move(b_modes[i].v)});
    }
    return out;
}

std::vector<SVDMode> svd_top_k(
    const Matrix& A,
    std::size_t k,
    int iters,
    std::uint64_t seed) {
    check_matrix_storage(A);
    const std::size_t m = A.rows;
    const std::size_t n = A.cols;
    if (k == 0 || m == 0 || n == 0) {
        return {};
    }
    const double scale = frob_norm(A);
    if (scale == 0.0) {
        return {};
    }
    if (std::min(m, n) > 512 && k + 8 < std::min(m, n) / 2) {
        return randomized_svd(A, k, 8, 2, seed);
    }

#ifdef __APPLE__
    {
        const std::size_t r_min = std::min(m, n);
        const std::size_t k_eff = std::min(k, r_min);
        if (r_min <= 512 || k_eff >= r_min / 2) {
            const Matrix A_norm = mat_scale(A, 1.0 / scale);
            std::vector<SVDMode> lap_modes;
            if (lapack_dgesdd(A_norm, k_eff, lap_modes)) {
                for (SVDMode& mode : lap_modes) {
                    mode.sigma *= scale;
                }
                return lap_modes;
            }
        }
    }
#endif

    if (m < n) {
        std::vector<SVDMode> t_modes = svd_top_k(transpose(A), k, iters, seed);
        for (SVDMode& mode : t_modes) {
            std::swap(mode.u, mode.v);
        }
        return t_modes;
    }

    const Matrix A_norm = mat_scale(A, 1.0 / scale);
    const std::size_t k_eff = std::min({k, m, n});

    std::vector<SVDMode> modes;
    if (n <= 96 || k_eff >= n / 2) {
        if (m > n) {
            std::vector<Vector> cols(n);
            for (std::size_t c = 0; c < n; ++c) {
                cols[c] = A_norm.col(c);
            }
            const std::vector<Vector> q_cols = gram_schmidt(cols, 1e-13);
            if (q_cols.empty()) {
                return {};
            }
            const std::size_t r_dim = q_cols.size();
            const Matrix q_rows = rows_to_matrix(q_cols);
            const Matrix q_mat = transpose(q_rows);
            const Matrix R = mat_mul(q_rows, A_norm);

            if (r_dim < n) {
                std::vector<SVDMode> small = hestenes_jacobi_svd_tall(transpose(R));
                modes.reserve(small.size());
                for (SVDMode& sm : small) {
                    modes.push_back(SVDMode{sm.sigma, mat_vec(q_mat, sm.v), std::move(sm.u)});
                }
            } else {
                std::vector<SVDMode> small = hestenes_jacobi_svd_tall(R);
                modes.reserve(small.size());
                for (SVDMode& sm : small) {
                    modes.push_back(SVDMode{sm.sigma, mat_vec(q_mat, sm.u), std::move(sm.v)});
                }
            }
        } else {
            modes = hestenes_jacobi_svd_tall(A_norm);
        }
    } else {
        const std::size_t q = std::min(n, k_eff + std::max<std::size_t>(8, k_eff));
        const Matrix A_T = transpose(A_norm);
        std::mt19937_64 rng(seed);
        std::normal_distribution<double> dist(0.0, 1.0);

        std::vector<Vector> v_init(q, Vector(n));
        for (std::size_t j = 0; j < q; ++j) {
            for (std::size_t i = 0; i < n; ++i) {
                v_init[j][i] = dist(rng);
            }
        }
        std::vector<Vector> v_cols = gram_schmidt(v_init, 1e-13);
        while (v_cols.size() < q) {
            Vector extra(n);
            for (std::size_t i = 0; i < n; ++i) {
                extra[i] = dist(rng);
            }
            v_cols.push_back(std::move(extra));
            v_cols = gram_schmidt(v_cols, 1e-13);
        }

        const int num_iters = std::max(1, iters);
        for (int it = 0; it < num_iters; ++it) {
            std::vector<Vector> u_raw;
            u_raw.reserve(v_cols.size());
            for (const Vector& v : v_cols) {
                u_raw.push_back(mat_vec(A_norm, v));
            }
            std::vector<Vector> u_cols = gram_schmidt(u_raw, 1e-13);
            if (u_cols.empty()) {
                return {};
            }
            std::vector<Vector> v_raw;
            v_raw.reserve(u_cols.size());
            for (const Vector& u : u_cols) {
                v_raw.push_back(mat_vec(A_T, u));
            }
            v_cols = gram_schmidt(v_raw, 1e-13);
            if (v_cols.empty()) {
                return {};
            }
        }

        std::vector<Vector> av_cols;
        av_cols.reserve(v_cols.size());
        for (const Vector& v : v_cols) {
            av_cols.push_back(mat_vec(A_norm, v));
        }
        std::vector<Vector> u_cols = gram_schmidt(av_cols, 1e-13);
        if (u_cols.empty()) {
            return {};
        }

        const std::size_t r_u = u_cols.size();
        const std::size_t r_v = v_cols.size();
        Matrix B(r_u, r_v, 0.0);
        for (std::size_t i = 0; i < r_u; ++i) {
            for (std::size_t j = 0; j < r_v; ++j) {
                B(i, j) = dot(u_cols[i], av_cols[j]);
            }
        }

        std::vector<SVDMode> core_svd;
        if (r_u >= r_v) {
            core_svd = hestenes_jacobi_svd_tall(B);
        } else {
            core_svd = hestenes_jacobi_svd_tall(transpose(B));
            for (SVDMode& cm : core_svd) {
                std::swap(cm.u, cm.v);
            }
        }

        const Matrix u_basis_mat = transpose(rows_to_matrix(u_cols));
        const Matrix v_basis_mat = transpose(rows_to_matrix(v_cols));
        modes.reserve(core_svd.size());
        for (const SVDMode& cm : core_svd) {
            modes.push_back(SVDMode{
                cm.sigma,
                normalize(mat_vec(u_basis_mat, cm.u)),
                normalize(mat_vec(v_basis_mat, cm.v))});
        }
    }

    if (modes.size() > k_eff) {
        modes.resize(k_eff);
    }
    for (SVDMode& mode : modes) {
        mode.sigma *= scale;
    }
    return modes;
}

Matrix orthogonal_procrustes(const Matrix& src_points, const Matrix& tgt_points) {
    check_matrix_storage(src_points);
    check_matrix_storage(tgt_points);
    if (src_points.rows != tgt_points.rows || src_points.cols != tgt_points.cols) {
        throw std::invalid_argument("Shape mismatch in orthogonal_procrustes");
    }
    const std::size_t n = src_points.rows;
    const std::size_t k = src_points.cols;
    if (k == 0) {
        return Matrix(0, 0);
    }
    if (n == 0) {
        return eye(k);
    }

    Matrix M = mat_mul(transpose(src_points), tgt_points);
    const double m_norm = frob_norm(M);
    if (m_norm == 0.0) {
        return eye(k);
    }
    M = mat_scale(M, 1.0 / m_norm);

    const std::vector<SVDMode> modes = svd_top_k(M, k);
    std::vector<Vector> u_cols;
    std::vector<Vector> v_cols;
    u_cols.reserve(k);
    v_cols.reserve(k);
    for (const SVDMode& mode : modes) {
        u_cols.push_back(mode.u);
        v_cols.push_back(mode.v);
    }

    if (u_cols.size() < k) {
        for (std::size_t i = 0; i < k; ++i) {
            Vector e(k, 0.0);
            e[i] = 1.0;
            u_cols.push_back(std::move(e));
        }
        u_cols = gram_schmidt(u_cols, 1e-12);
        u_cols.resize(k);
    }
    if (v_cols.size() < k) {
        for (std::size_t i = 0; i < k; ++i) {
            Vector e(k, 0.0);
            e[i] = 1.0;
            v_cols.push_back(std::move(e));
        }
        v_cols = gram_schmidt(v_cols, 1e-12);
        v_cols.resize(k);
    }

    const Matrix U = transpose(rows_to_matrix(u_cols));
    const Matrix Vt = rows_to_matrix(v_cols);
    return mat_mul(U, Vt);
}

double silu(double z) {
    if (z >= 0.0) {
        return z / (1.0 + std::exp(-z));
    }
    const double ez = std::exp(z);
    return z * (ez / (1.0 + ez));
}

Vector rms_norm(const Vector& x, const Vector& weight, double eps) {
    if (x.size() != weight.size()) {
        throw std::invalid_argument("Length mismatch in rms_norm");
    }
    const std::size_t d = x.size();
    if (d == 0) {
        return {};
    }
    long double sq_sum = 0.0;
    for (double xi : x) {
        sq_sum += static_cast<long double>(xi) * static_cast<long double>(xi);
    }
    const double mean_sq = static_cast<double>(sq_sum / static_cast<long double>(d));
    const double inv_rms = 1.0 / std::sqrt(mean_sq + eps);
    Vector out(d);
    for (std::size_t i = 0; i < d; ++i) {
        out[i] = x[i] * inv_rms * weight[i];
    }
    return out;
}

Vector softmax(const Vector& xs) {
    if (xs.empty()) {
        return {};
    }
    const double m = *std::max_element(xs.begin(), xs.end());
    Vector exps(xs.size());
    long double denom = 0.0;
    for (std::size_t i = 0; i < xs.size(); ++i) {
        const double e = std::exp(xs[i] - m);
        exps[i] = e;
        denom += static_cast<long double>(e);
    }
    const double inv_denom = static_cast<double>(1.0L / denom);
    for (double& e : exps) {
        e *= inv_denom;
    }
    return exps;
}

}
