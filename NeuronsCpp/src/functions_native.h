#ifndef FUNCTIONS_NATIVE_H
#define FUNCTIONS_NATIVE_H

#include <godot_cpp/classes/ref_counted.hpp>
#include <godot_cpp/variant/array.hpp>
#include <godot_cpp/variant/vector4.hpp>

#include <cstdint>
#include <vector>

namespace godot {

class FunctionsNative : public RefCounted {
	GDCLASS(FunctionsNative, RefCounted);

protected:
	static void _bind_methods();

public:
	FunctionsNative();
	~FunctionsNative();

	double W(int64_t t) const;
	double U(int64_t xt) const;
	double F(double beta, const Array &x) const;
	double G(bool delta_1, bool delta_2, const Array &x) const;
	double Beta(const Array &x, int64_t I, int64_t J) const;
	double A(int64_t i, int64_t j, int64_t k, const Array &x, int64_t I, int64_t J, int64_t K) const;
	Vector4 H_coefficients(const Array &x) const;
	double H(double A_ijk, double A_uvw, const Vector4 &coefficients) const;
	double D(int64_t i, int64_t j, int64_t k, const Array &x, int64_t I, int64_t J, int64_t K) const;
	double P(const Array &x) const;

	static double w_value(int32_t t);
	static double u_value(int32_t xt);
	static double f_value(double beta, const std::vector<int32_t> &x);
	static double g_value(bool delta_1, bool delta_2, const std::vector<int32_t> &x);
	static double beta_value(const std::vector<int32_t> &x, int32_t I, int32_t J);
	static double a_value(int32_t i, int32_t j, int32_t k, const std::vector<int32_t> &x, int32_t I, int32_t J, int32_t K);
	static Vector4 h_coefficients_value(const std::vector<int32_t> &x);
	static double h_value(double A_ijk, double A_uvw, const Vector4 &coefficients);
	static double d_value(int32_t i, int32_t j, int32_t k, const std::vector<int32_t> &x, int32_t I, int32_t J, int32_t K);
	static double p_value(const std::vector<int32_t> &x);

private:
	static std::vector<int32_t> _array_to_vector(const Array &x);
};

}

#endif
