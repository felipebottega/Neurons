#include "functions_native.h"

#include <godot_cpp/core/class_db.hpp>

#include <cmath>

using namespace godot;

namespace {
constexpr float PI_VALUE = 3.141592653589793238462643383279502884f;
}

void FunctionsNative::_bind_methods() {
	ClassDB::bind_method(D_METHOD("W", "t"), &FunctionsNative::W);
	ClassDB::bind_method(D_METHOD("U", "xt"), &FunctionsNative::U);
	ClassDB::bind_method(D_METHOD("F", "beta", "x"), &FunctionsNative::F);
	ClassDB::bind_method(D_METHOD("G", "delta_1", "delta_2", "x"), &FunctionsNative::G);
	ClassDB::bind_method(D_METHOD("Beta", "x", "I", "J"), &FunctionsNative::Beta);
	ClassDB::bind_method(D_METHOD("A", "i", "j", "k", "x", "I", "J", "K"), &FunctionsNative::A);
	ClassDB::bind_method(D_METHOD("H_coefficients", "x"), &FunctionsNative::H_coefficients);
	ClassDB::bind_method(D_METHOD("H", "A_ijk", "A_uvw", "coefficients"), &FunctionsNative::H);
	ClassDB::bind_method(D_METHOD("D", "i", "j", "k", "x", "I", "J", "K"), &FunctionsNative::D);
	ClassDB::bind_method(D_METHOD("P", "x"), &FunctionsNative::P);
}

FunctionsNative::FunctionsNative() {
}

FunctionsNative::~FunctionsNative() {
}

double FunctionsNative::W(int64_t t) const {
	return w_value(static_cast<int32_t>(t));
}

double FunctionsNative::U(int64_t xt) const {
	return u_value(static_cast<int32_t>(xt));
}

double FunctionsNative::F(double beta, const Array &x) const {
	return f_value(beta, _array_to_vector(x));
}

double FunctionsNative::G(bool delta_1, bool delta_2, const Array &x) const {
	return g_value(delta_1, delta_2, _array_to_vector(x));
}

double FunctionsNative::Beta(const Array &x, int64_t I, int64_t J) const {
	return beta_value(_array_to_vector(x), static_cast<int32_t>(I), static_cast<int32_t>(J));
}

double FunctionsNative::A(int64_t i, int64_t j, int64_t k, const Array &x, int64_t I, int64_t J, int64_t K) const {
	return a_value(
			static_cast<int32_t>(i),
			static_cast<int32_t>(j),
			static_cast<int32_t>(k),
			_array_to_vector(x),
			static_cast<int32_t>(I),
			static_cast<int32_t>(J),
			static_cast<int32_t>(K));
}

Vector4 FunctionsNative::H_coefficients(const Array &x) const {
	return h_coefficients_value(_array_to_vector(x));
}

double FunctionsNative::H(double A_ijk, double A_uvw, const Vector4 &coefficients) const {
	return h_value(A_ijk, A_uvw, coefficients);
}

double FunctionsNative::D(int64_t i, int64_t j, int64_t k, const Array &x, int64_t I, int64_t J, int64_t K) const {
	return d_value(
			static_cast<int32_t>(i),
			static_cast<int32_t>(j),
			static_cast<int32_t>(k),
			_array_to_vector(x),
			static_cast<int32_t>(I),
			static_cast<int32_t>(J),
			static_cast<int32_t>(K));
}

double FunctionsNative::P(const Array &x) const {
	return p_value(_array_to_vector(x));
}

float FunctionsNative::w_value(int32_t t) {
	return std::sqrt(1.0f + static_cast<float>(t));
}

float FunctionsNative::u_value(int32_t xt) {
	return 1.0f + static_cast<float>(xt);
}

float FunctionsNative::f_value(float beta, const std::vector<int32_t> &x) {
	float y = 0.0f;

	for (int32_t t = 0; t < static_cast<int32_t>(x.size()); ++t) {
		y += w_value(t) * std::pow(beta, static_cast<float>(x[static_cast<size_t>(t)]));
	}

	return y;
}

float FunctionsNative::g_value(bool delta_1, bool delta_2, const std::vector<int32_t> &x) {
	float y = 0.0f;
	const int32_t exponent_1 = delta_1 ? 1 : 0;
	const int32_t exponent_2 = delta_2 ? 1 : 0;

	for (int32_t t = 0; t < static_cast<int32_t>(x.size()); ++t) {
		const float gene_value = 1.0f + static_cast<float>(x[static_cast<size_t>(t)]);
		y += std::sin(1.0f + static_cast<float>(t))
				* std::pow(std::sin(gene_value), static_cast<float>(exponent_1))
				* std::pow(std::cos(gene_value), static_cast<float>(exponent_2));
	}

	return y;
}

float FunctionsNative::beta_value(const std::vector<int32_t> &x, int32_t I, int32_t J) {
	float weighted_sum = 0.0f;
	float max_sum = 0.0f;

	for (int32_t t = 0; t < static_cast<int32_t>(x.size()); ++t) {
		const float weight = w_value(t);
		weighted_sum += weight * static_cast<float>(x[static_cast<size_t>(t)]);
		max_sum += 3.0f * weight;
	}

	const float scale = std::pow(static_cast<float>(I * J) / 16.0f, 1.0f / 3.0f);
	return 1.0f + weighted_sum / max_sum * scale;
}

float FunctionsNative::a_value(int32_t i, int32_t j, int32_t k, const std::vector<int32_t> &x, int32_t I, int32_t J, int32_t K) {
	const int32_t p = static_cast<int32_t>(x.size());
	const int32_t n = static_cast<int32_t>(std::llround(std::pow(static_cast<float>(p), 1.0f / 3.0f)));
	const float a = static_cast<float>(i) / static_cast<float>(I - 1);
	const float b = static_cast<float>(j) / static_cast<float>(J - 1);
	const float c = static_cast<float>(k) / static_cast<float>(K - 1);
	float y = 0.0f;

	for (int32_t r = 0; r < n; ++r) {
		for (int32_t s = 0; s < n; ++s) {
			for (int32_t q = 0; q < n; ++q) {
				const int32_t t = q + n * s + n * n * r;
				const float coefficient = 2.0f * static_cast<float>(x[static_cast<size_t>(t)]) / 3.0f - 1.0f;
				y += coefficient
						* std::cos(PI_VALUE * static_cast<float>(r) * a)
						* std::cos(PI_VALUE * static_cast<float>(s) * b)
						* std::cos(PI_VALUE * static_cast<float>(q) * c);
			}
		}
	}

	return std::tanh(y / std::sqrt(static_cast<float>(p)));
}

Vector4 FunctionsNative::h_coefficients_value(const std::vector<int32_t> &x) {
	Vector4 coefficients(0.0f, 0.0f, 0.0f, 0.0f);
	float C = 0.0f;

	for (int32_t t = 0; t < static_cast<int32_t>(x.size()); ++t) {
		const float weight = w_value(t);
		C += weight;

		switch (x[static_cast<size_t>(t)]) {
			case 0:
				coefficients.x += weight;
				break;
			case 1:
				coefficients.y += weight;
				break;
			case 2:
				coefficients.z += weight;
				break;
			case 3:
				coefficients.w += weight;
				break;
			default:
				break;
		}
	}

	return coefficients * (4.0f / C);
}

float FunctionsNative::h_value(float A_ijk, float A_uvw, const Vector4 &coefficients) {
	return std::sin(
			3.0f * PI_VALUE / 8.0f
			+ coefficients.x * (std::sin(A_ijk) + std::cos(A_uvw))
			+ coefficients.y * (std::sin(2.0f * A_ijk) + std::cos(2.0f * A_uvw))
			+ coefficients.z * (std::sin(3.0f * A_ijk) + std::cos(3.0f * A_uvw))
			+ coefficients.w * (std::sin(4.0f * A_ijk) + std::cos(4.0f * A_uvw)));
}

float FunctionsNative::d_value(int32_t i, int32_t j, int32_t k, const std::vector<int32_t> &x, int32_t I, int32_t J, int32_t K) {
	return 1.5f + a_value(i, j, k, x, I, J, K);
}

float FunctionsNative::p_value(const std::vector<int32_t> &x) {
	return 1.75f + 0.25f * std::cos(g_value(true, true, x));
}

std::vector<int32_t> FunctionsNative::_array_to_vector(const Array &x) {
	std::vector<int32_t> result(static_cast<size_t>(x.size()));
	for (int32_t i = 0; i < x.size(); ++i) {
		result[static_cast<size_t>(i)] = static_cast<int32_t>(static_cast<int64_t>(x[i]));
	}
	return result;
}
