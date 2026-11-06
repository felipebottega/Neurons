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

func S(x: Array) -> float:
	# Relative spatial extent for the first pass of initial connectivity.
	return clampf((1.0 - sqrt(abs(cos(G(false, true, x))))) / 10.0, 0.0, 0.1)

func Beta(x: Array, I: int, J: int) -> float:
	var weighted_sum := 0.0
	var max_sum := 0.0

	for t in x.size():
		weighted_sum += W(t) * x[t]
		max_sum += 3.0 * W(t)

	var scale := pow(float(I * J) / 16.0, 1.0 / 3.0)

	return 1.0 + weighted_sum / max_sum * scale
	
# Fixed spatial length in lattice steps; keep equal to brain_neurons.glsl.
# Brain dimensions only bound the sampled domain, not the local field scale.
const A_SPATIAL_SCALE: float = 16.0

func A(i: int, j: int, k: int, x: Array, I: int, J: int, K: int) -> float:
	# I, J, K are retained for existing callers. Equal lattice coordinates
	# produce equal A values regardless of the containing brain dimensions.
	var p: int = x.size()
	var n: int = roundi(pow(float(p), 1.0 / 3.0))
	var a := float(i) / A_SPATIAL_SCALE
	var b := float(j) / A_SPATIAL_SCALE
	var c := float(k) / A_SPATIAL_SCALE
	var basis_a: Array[float] = []
	var basis_b: Array[float] = []
	var basis_c: Array[float] = []
	for mode in n:
		var frequency := sqrt(float(mode + 1))
		basis_a.append(cos(PI * frequency * a))
		basis_b.append(cos(PI * frequency * b))
		basis_c.append(cos(PI * frequency * c))
	var y := 0.0

	for r in n:
		for s in n:
			for q in n:
				var t := q + n * s + n * n * r
				var coefficient := 2.0 * float(x[t]) / 3.0 - 1.0
				y += coefficient * basis_a[r] * basis_b[s] * basis_c[q]

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

	return coefficients * (4.0 / C)

func H(A_ijk: float, A_uvw: float, coefficients: Vector4) -> float:
	return sin(
		3.0 * PI / 8.0
		+ coefficients.x * (sin(A_ijk) + cos(A_uvw))
		+ coefficients.y * (sin(2.0 * A_ijk) + cos(2.0 * A_uvw))
		+ coefficients.z * (sin(3.0 * A_ijk) + cos(3.0 * A_uvw))
		+ coefficients.w * (sin(4.0 * A_ijk) + cos(4.0 * A_uvw))
	)

func D(i: int, j: int, k: int, x: Array, I: int, J: int, K: int) -> float:
	return 1.5 + A(i, j, k, x, I, J, K)
	
func P(x: Array):
	return 1.75 + 0.25 * cos(G(true, true, x))
	
