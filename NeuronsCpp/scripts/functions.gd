class_name Functions
extends Node


func W(t: int) -> float:
	return sqrt(1.0 + t)

func U(xt: int) -> float:
	return 1.0 + xt

func F(beta: float, x: Array) -> float:
	var p = len(x)
	var y = 0.0
	
	for t in p:
		y += (W(t) * beta**x[t])
		
	return y
	
func G(delta_1: bool, delta_2: bool, x: Array) -> float:
	var p = len(x)
	var y = 0.0
	
	for t in p:
		y += (sin(1 + t) * (sin(1 + x[t])**int(delta_1)) * (cos(1 + x[t])**int(delta_2)))
	
	return y

func Beta(x: Array, I: int, J: int) -> float:
	var weighted_sum := 0.0
	var max_sum := 0.0

	for t in x.size():
		weighted_sum += W(t) * x[t]
		max_sum += 3.0 * W(t)

	var scale := pow(float(I * J) / 16.0, 1.0 / 3.0)

	return 1.0 + weighted_sum / max_sum * scale
	
func A(i: int, j: int, k: int, x: Array, I: int, J: int, K: int) -> float:
	var p: int = x.size()
	var n: int = roundi(pow(float(p), 1.0 / 3.0))
	var a := float(i) / float(I - 1)
	var b := float(j) / float(J - 1)
	var c := float(k) / float(K - 1)
	var y := 0.0

	for r in n:
		for s in n:
			for q in n:
				var t := q + n * s + n * n * r
				var coefficient := 2.0 * float(x[t]) / 3.0 - 1.0
				y += coefficient * cos(PI * r * a) * cos(PI * s * b) * cos(PI * q * c)

	return tanh(y / sqrt(float(p)))

func H_coefficients(x: Array) -> Vector4:
	var coefficients = Vector4.ZERO
	var C = 0.0

	for t in x.size():
		var weight = W(t)
		C += weight

		match x[t]:
			0: coefficients.x += weight
			1: coefficients.y += weight
			2: coefficients.z += weight
			3: coefficients.w += weight

	return coefficients * (3.0 * PI / (4.0 * C))

func H(A_ijk: float, A_uvw: float, coefficients: Vector4) -> float:
	return sin(PI/16.0 + coefficients.x * cos(A_ijk) * sin(A_uvw) + coefficients.y * cos(2.0 * A_ijk) * sin(2.0 * A_uvw) + coefficients.z * cos(3.0 * A_ijk) * sin(3.0 * A_uvw) + coefficients.w * cos(4.0 * A_ijk) * sin(4.0 * A_uvw))

func D(i: int, j: int, k: int, x: Array, I: int, J: int, K: int) -> float:
	return 1.5 + A(i, j, k, x, I, J, K)
	
func P(x: Array):
	return 1.75 + 0.25 * cos(G(true, true, x))
	
