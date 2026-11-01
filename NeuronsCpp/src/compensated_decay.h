#ifndef NEURONS_COMPENSATED_DECAY_H
#define NEURONS_COMPENSATED_DECAY_H

#if defined(__FAST_MATH__)
#error "Compensated decay requires floating-point reassociation to be disabled."
#endif

#if defined(_MSC_VER)
#pragma float_control(precise, on, push)
#endif

namespace neurons_numeric {

// Signed rounding debt: the represented weight is approximately weight - residual.
// Keep the residual across iterations; it may be negative after rounding downward.
// Do not reassociate these operations: doing so would erase the compensation.
inline float subtract_decay(float weight, float decrement, float &residual) {
	const float adjusted = decrement + residual;
	const float next = weight - adjusted;
	const float applied = weight - next;
	residual = adjusted - applied;
	if (next <= 0.0f) {
		residual = 0.0f;
		return 0.0f;
	}
	return next;
}

} // namespace neurons_numeric

#if defined(_MSC_VER)
#pragma float_control(pop)
#endif

#endif
