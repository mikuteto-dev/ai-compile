#pragma once

#include <cstddef>
#include <cstdint>
#include <vector>

namespace aic {

using Vector = std::vector<double>;

struct Matrix {
    std::size_t rows = 0;
    std::size_t cols = 0;
    std::vector<double> data;

    Matrix() = default;
    Matrix(std::size_t r, std::size_t c, double fill = 0.0);

    double& operator()(std::size_t r, std::size_t c);
    double operator()(std::size_t r, std::size_t c) const;
    Vector row(std::size_t r) const;
    void set_row(std::size_t r, const Vector& v);
    Vector col(std::size_t c) const;
    void set_col(std::size_t c, const Vector& v);
};

Matrix zeros(std::size_t r, std::size_t c);
Matrix eye(std::size_t n);
Vector vec_zeros(std::size_t n);

double dot(const Vector& a, const Vector& b);
double norm2(const Vector& a);
Vector normalize(const Vector& a, double eps = 1e-12);

Vector vec_add(const Vector& a, const Vector& b);
Vector vec_sub(const Vector& a, const Vector& b);
Vector vec_scale(const Vector& a, double s);

Matrix outer(const Vector& u, const Vector& v);
Vector mat_vec(const Matrix& A, const Vector& x);
Matrix mat_mul(const Matrix& A, const Matrix& B);
Matrix transpose(const Matrix& A);

Matrix mat_add(const Matrix& A, const Matrix& B);
Matrix mat_sub(const Matrix& A, const Matrix& B);
Matrix mat_scale(const Matrix& A, double s);
double frob_norm(const Matrix& A);

Vector solve_linear(Matrix A, Vector b);
Matrix solve_linear_multi(Matrix A, Matrix B);
Matrix mat_inv(const Matrix& A);
std::vector<Vector> gram_schmidt(const std::vector<Vector>& vecs, double tol = 1e-10);

struct SVDMode {
    double sigma = 0.0;
    Vector u;
    Vector v;
};

std::vector<SVDMode> svd_top_k(
    const Matrix& A,
    std::size_t k,
    int iters = 50,
    std::uint64_t seed = 0);

std::vector<SVDMode> randomized_svd(
    const Matrix& A,
    std::size_t k,
    std::size_t oversample = 8,
    int power_iters = 2,
    std::uint64_t seed = 0);

Matrix orthogonal_procrustes(const Matrix& src_points, const Matrix& tgt_points);

double silu(double z);
Vector rms_norm(const Vector& x, const Vector& weight, double eps = 1e-6);
Vector softmax(const Vector& xs);

}
