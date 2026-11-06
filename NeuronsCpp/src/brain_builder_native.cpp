#include "brain_builder_native.h"
#include "functions_native.h"

#include <godot_cpp/core/class_db.hpp>
#include <godot_cpp/variant/array.hpp>
#include <godot_cpp/variant/string.hpp>
#include <godot_cpp/variant/vector4.hpp>
#include <godot_cpp/variant/utility_functions.hpp>

#include <algorithm>
#include <array>
#include <unordered_set>
#include <atomic>
#include <chrono>
#include <cmath>
#include <cstdint>
#include <limits>
#include <thread>
#include <utility>
#include <vector>

using namespace godot;

namespace {

template <typename T, typename Comparator>
class NativeSortArray {
	static constexpr int64_t INTROSORT_THRESHOLD = 16;
	Comparator compare;

	int64_t median_of_3_index(const T *array, int64_t a_index, int64_t b_index, int64_t c_index) const {
		const T &a = array[a_index];
		const T &b = array[b_index];
		const T &c = array[c_index];
		if (compare(a, b)) {
			if (compare(b, c)) {
				return b_index;
			}
			if (compare(a, c)) {
				return c_index;
			}
			return a_index;
		}
		if (compare(a, c)) {
			return a_index;
		}
		if (compare(b, c)) {
			return c_index;
		}
		return b_index;
	}

	int64_t bitlog(int64_t n) const {
		int64_t k = 0;
		while (n != 1) {
			n >>= 1;
			++k;
		}
		return k;
	}

	void push_heap(int64_t first, int64_t hole_index, int64_t top_index, T value, T *array) const {
		int64_t parent = (hole_index - 1) / 2;
		while (hole_index > top_index && compare(array[first + parent], value)) {
			array[first + hole_index] = array[first + parent];
			hole_index = parent;
			parent = (hole_index - 1) / 2;
		}
		array[first + hole_index] = std::move(value);
	}

	void adjust_heap(int64_t first, int64_t hole_index, int64_t length, T value, T *array) const {
		const int64_t top_index = hole_index;
		int64_t second_child = 2 * hole_index + 2;

		while (second_child < length) {
			if (compare(array[first + second_child], array[first + second_child - 1])) {
				--second_child;
			}
			array[first + hole_index] = array[first + second_child];
			hole_index = second_child;
			second_child = 2 * (second_child + 1);
		}

		if (second_child == length) {
			array[first + hole_index] = array[first + second_child - 1];
			hole_index = second_child - 1;
		}

		push_heap(first, hole_index, top_index, std::move(value), array);
	}

	void pop_heap(int64_t first, int64_t last, int64_t result, T value, T *array) const {
		array[result] = array[first];
		adjust_heap(first, 0, last - first, std::move(value), array);
	}

	void pop_heap(int64_t first, int64_t last, T *array) const {
		pop_heap(first, last - 1, last - 1, array[last - 1], array);
	}

	void sort_heap(int64_t first, int64_t last, T *array) const {
		while (last - first > 1) {
			pop_heap(first, last--, array);
		}
	}

	void make_heap(int64_t first, int64_t last, T *array) const {
		if (last - first < 2) {
			return;
		}
		const int64_t length = last - first;
		int64_t parent = (length - 2) / 2;
		while (true) {
			adjust_heap(first, parent, length, array[first + parent], array);
			if (parent == 0) {
				return;
			}
			--parent;
		}
	}

	void partial_sort(int64_t first, int64_t last, int64_t middle, T *array) const {
		make_heap(first, middle, array);
		for (int64_t i = middle; i < last; ++i) {
			if (compare(array[i], array[first])) {
				pop_heap(first, middle, i, array[i], array);
			}
		}
		sort_heap(first, middle, array);
	}

	int64_t partitioner(int64_t first, int64_t last, int64_t pivot, T *array) const {
		const T *pivot_element = &array[pivot];
		while (true) {
			while (first != pivot && compare(array[first], *pivot_element)) {
				++first;
			}
			--last;
			while (last != pivot && compare(*pivot_element, array[last])) {
				--last;
			}
			if (first >= last) {
				return first;
			}
			if (pivot_element == &array[first]) {
				pivot_element = &array[last];
			} else if (pivot_element == &array[last]) {
				pivot_element = &array[first];
			}
			std::swap(array[first], array[last]);
			++first;
		}
	}

	void introsort(int64_t first, int64_t last, T *array, int64_t max_depth) const {
		while (last - first > INTROSORT_THRESHOLD) {
			if (max_depth == 0) {
				partial_sort(first, last, last, array);
				return;
			}
			--max_depth;
			const int64_t cut = partitioner(
					first,
					last,
					median_of_3_index(array, first, first + (last - first) / 2, last - 1),
					array);
			introsort(cut, last, array, max_depth);
			last = cut;
		}
	}

	void unguarded_linear_insert(int64_t last, T value, T *array) const {
		int64_t next = last - 1;
		while (compare(value, array[next])) {
			array[last] = array[next];
			last = next;
			--next;
		}
		array[last] = std::move(value);
	}

	void linear_insert(int64_t first, int64_t last, T *array) const {
		T value = array[last];
		if (compare(value, array[first])) {
			for (int64_t i = last; i > first; --i) {
				array[i] = array[i - 1];
			}
			array[first] = std::move(value);
		} else {
			unguarded_linear_insert(last, std::move(value), array);
		}
	}

	void insertion_sort(int64_t first, int64_t last, T *array) const {
		if (first == last) {
			return;
		}
		for (int64_t i = first + 1; i != last; ++i) {
			linear_insert(first, i, array);
		}
	}

	void unguarded_insertion_sort(int64_t first, int64_t last, T *array) const {
		for (int64_t i = first; i != last; ++i) {
			unguarded_linear_insert(i, array[i], array);
		}
	}

	void final_insertion_sort(int64_t first, int64_t last, T *array) const {
		if (last - first > INTROSORT_THRESHOLD) {
			insertion_sort(first, first + INTROSORT_THRESHOLD, array);
			unguarded_insertion_sort(first + INTROSORT_THRESHOLD, last, array);
		} else {
			insertion_sort(first, last, array);
		}
	}

public:
	explicit NativeSortArray(Comparator comparator) : compare(std::move(comparator)) {
	}

	void sort(T *array, int64_t length) const {
		if (length <= 1) {
			return;
		}
		introsort(0, length, array, bitlog(length) * 2);
		final_insertion_sort(0, length, array);
	}
};

template <typename Function, typename ProgressFunction>
void run_parallel(int32_t count, Function function, ProgressFunction progress_function) {
	if (count <= 0) {
		return;
	}

	unsigned int hardware_threads = std::thread::hardware_concurrency();
	if (hardware_threads == 0) {
		hardware_threads = 1;
	}
	const int32_t worker_count = std::min<int32_t>(count, static_cast<int32_t>(hardware_threads));

	if (worker_count <= 1) {
		for (int32_t index = 0; index < count; ++index) {
			function(index);
			progress_function(index + 1);
		}
		return;
	}

	std::atomic<int32_t> next_index{ 0 };
	std::atomic<int32_t> completed{ 0 };
	std::vector<std::thread> workers;
	workers.reserve(static_cast<size_t>(worker_count));

	for (int32_t worker_index = 0; worker_index < worker_count; ++worker_index) {
		workers.emplace_back([&]() {
			while (true) {
				const int32_t index = next_index.fetch_add(1, std::memory_order_relaxed);
				if (index >= count) {
					break;
				}
				function(index);
				completed.fetch_add(1, std::memory_order_release);
			}
		});
	}

	int32_t last_reported = -1;
	while (completed.load(std::memory_order_acquire) < count) {
		const int32_t current_completed = completed.load(std::memory_order_acquire);
		if (current_completed != last_reported) {
			last_reported = current_completed;
			progress_function(current_completed);
		}
		std::this_thread::sleep_for(std::chrono::milliseconds(1));
	}

	for (std::thread &worker : workers) {
		worker.join();
	}

	progress_function(count);
}

struct ConnectionCandidate {
	int32_t target_spatial_index = 0;
	float value = 0.0f;
};


using Quad = std::array<int32_t, 4>;
struct Offset3 { int32_t x, y, z; };

// Match the GPU builder's GDScript scalar calculations. Keep every intermediate
// in double precision, including positional weights and beta, until floor().
// The other gene mappings retain their existing FP32 implementation.
double initial_connection_beta(const std::vector<int32_t> &gene, int32_t I, int32_t J) {
	double weighted_sum = 0.0;
	double max_sum = 0.0;
	for (size_t t = 0; t < gene.size(); ++t) {
		const double weight = std::sqrt(1.0 + static_cast<double>(t));
		weighted_sum += weight * static_cast<double>(gene[t]);
		max_sum += 3.0 * weight;
	}
	const double scale = std::pow(static_cast<double>(I) * static_cast<double>(J) / 16.0, 1.0 / 3.0);
	return 1.0 + weighted_sum / max_sum * scale;
}

double initial_connection_limit(double beta, const std::vector<int32_t> &gene) {
	double value = 0.0;
	for (size_t t = 0; t < gene.size(); ++t) {
		value += std::sqrt(1.0 + static_cast<double>(t)) * std::pow(beta, static_cast<double>(gene[t]));
	}
	return value;
}

Quad rotate_quad(const Quad &q, int32_t step) {
	Quad result;
	for (int32_t t = 0; t < 4; ++t) result[t] = (q[t] + step) % 21;
	std::sort(result.begin(), result.end());
	return result;
}

// One orbit is preferred; the other 284 are visited in lexicographic canonical order.
// There are 5985 distinct combinations, partitioned into 285 rotation orbits of 21.
std::vector<Quad> cyclic_quadruples() {
	std::vector<Quad> result;
	result.reserve(5985);
	const Quad first{0, 5, 10, 15};
	Quad first_canonical = first;
	for (int32_t h = 1; h < 21; ++h) first_canonical = std::min(first_canonical, rotate_quad(first, h));
	for (int32_t h = 0; h < 21; ++h) result.push_back(rotate_quad(first, h));
	for (int32_t a = 0; a < 18; ++a)
	for (int32_t b = a + 1; b < 19; ++b)
	for (int32_t c = b + 1; c < 20; ++c)
	for (int32_t d = c + 1; d < 21; ++d) {
		const Quad candidate{a, b, c, d};
		if (candidate == first_canonical) continue;
		bool canonical = true;
		for (int32_t h = 1; h < 21; ++h) {
			if (rotate_quad(candidate, h) < candidate) { canonical = false; break; }
		}
		if (!canonical) continue;
		for (int32_t h = 0; h < 21; ++h) result.push_back(rotate_quad(candidate, h));
	}
	return result;
}

int32_t four_decimal_digits(const std::vector<int32_t> &gene, int32_t start, const Quad &q) {
	int32_t result = 0;
	for (int32_t idx : q) result = result * 10 + gene[static_cast<size_t>(start + idx)];
	return result;
}

// Prepared once per brain, then reused for every neuron in every thread.
std::vector<Offset3> supplementary_offsets(const std::vector<int32_t> &gene,
		int32_t I, int32_t J, int32_t K, int32_t radius_i, int32_t radius_j, int32_t radius_k) {
	std::vector<Offset3> offsets;
	const int32_t Di = I - radius_i - 1;
	const int32_t Dj = J - radius_j - 1;
	const int32_t Dk = K - radius_k - 1;
	if (Di <= 0 && Dj <= 0 && Dk <= 0) return offsets;
	const auto quads = cyclic_quadruples();
	offsets.reserve(quads.size());
	for (size_t t = 0; t < quads.size(); ++t) {
		const Quad &q = quads[t];
		const int32_t a = four_decimal_digits(gene, 0, q) % I;
		const int32_t b = four_decimal_digits(gene, 21, q) % J;
		const int32_t c = four_decimal_digits(gene, 42, q) % K;
		// Preferred axis rotates i -> j -> k; skip axes with no exterior space.
		for (int32_t tries = 0; tries < 3; ++tries) {
			const int32_t axis = static_cast<int32_t>((t + tries) % 3);
			if (axis == 0 && Di > 0) {
				offsets.push_back({radius_i + 1 + a % Di, b, c}); break;
			}
			if (axis == 1 && Dj > 0) {
				offsets.push_back({a, radius_j + 1 + b % Dj, c}); break;
			}
			if (axis == 2 && Dk > 0) {
				offsets.push_back({a, b, radius_k + 1 + c % Dk}); break;
			}
		}
	}
	return offsets;
}

}

void BrainBuilderNative::_bind_methods() {
	ClassDB::bind_method(
			D_METHOD("build", "genome", "size", "num_inputs", "num_outputs", "progress_callback", "connections_density"),
			&BrainBuilderNative::build,
			DEFVAL(Callable()), DEFVAL(1.0));
}

BrainBuilderNative::BrainBuilderNative() {
}

BrainBuilderNative::~BrainBuilderNative() {
}

Dictionary BrainBuilderNative::build(
		const Ref<GenomeNative> &genome,
		const Vector3i &size,
		int64_t num_inputs,
		int64_t num_outputs,
		const Callable &progress_callback,
		double connections_density) {
	if (genome.is_null()) {
		UtilityFunctions::push_error("GenomeNative is null.");
		return Dictionary();
	}
	if (size.x < 2 || size.y < 2 || size.z < 2 || num_inputs < 0 || num_outputs < 0 ||
		num_inputs + num_outputs > static_cast<int64_t>(size.x) * size.y * size.z) {
		UtilityFunctions::push_error("Invalid brain dimensions or input/output counts.");
		return Dictionary();
	}
	if (!std::isfinite(connections_density) || connections_density <= 0.0) {
		UtilityFunctions::push_error("Connections density eta must be positive and finite.");
		return Dictionary();
	}

	_report_progress(progress_callback, 0);

	NativeBrain brain;
	brain.size = size;
	brain.beta = initial_connection_beta(genome->beta, size.x, size.y);
	brain.connections_density = connections_density;
	const double unscaled_limit = initial_connection_limit(brain.beta, genome->max_connections);
	if (connections_density * unscaled_limit > static_cast<double>(std::numeric_limits<int32_t>::max() - 2)) {
		UtilityFunctions::push_error("Connections density exceeds supported maximum degree.");
		return Dictionary();
	}
	brain.max_connections = std::max(1, static_cast<int32_t>(std::floor(connections_density * unscaled_limit)));
	brain.refractory_strength = FunctionsNative::f_value(0.4f, genome->refractory_strength);
	brain.exponential_factor = FunctionsNative::f_value(0.4f, genome->exponential_factor);
	brain.structural_plasticity = FunctionsNative::p_value(genome->structural_plasticity);

	const Vector3 modulatory_dynamics = _modulatory_dynamics(genome->modulatory_dynamics);
	brain.lambda = modulatory_dynamics.x;
	brain.nu = modulatory_dynamics.y;
	brain.mu = modulatory_dynamics.z;

	_create_neurons(brain, *genome.ptr(), progress_callback);
	// The provisional graph must be built with every neuron internal. Its degrees
	// determine the admissibility of each prime-gap input/output candidate.
	_create_connections(brain, genome->connections, genome->initial_search_region, progress_callback);
	brain.role_selection_valid = _set_input_output_neurons(
			brain, static_cast<int32_t>(num_inputs), static_cast<int32_t>(num_outputs), progress_callback);
	const bool connections_valid = _clear_invalid_connections(
			brain, genome->connections, genome->input_influence, progress_callback);
	brain.valid = brain.role_selection_valid && connections_valid;

	// Native construction is not finished until its millions of Godot Variants are created.
	Dictionary result = _to_dictionary(brain, progress_callback);
	_report_progress(progress_callback, 100);
	return result;
}

void BrainBuilderNative::_create_neurons(NativeBrain &brain, const GenomeNative &genome, const Callable &progress_callback) const {
	const int32_t I = brain.size.x;
	const int32_t J = brain.size.y;
	const int32_t K = brain.size.z;
	const int32_t total_neurons = I * J * K;
	const float first_polarity = FunctionsNative::a_value(0, 0, 0, genome.polarity_factor, I, J, K);
	const float retention_normalization = 1.0f - std::exp(-2.0f);

	brain.neurons.resize(static_cast<size_t>(total_neurons));

	run_parallel(
			total_neurons,
			[&](int32_t creation_index) {
				const int32_t plane_stride = J * K;
				const int32_t i = creation_index / plane_stride;
				const int32_t remaining = creation_index - i * plane_stride;
				const int32_t j = remaining / K;
				const int32_t k = remaining - j * K;

				NativeNeuron neuron;
				neuron.position = Vector3i(i, j, k);
				neuron.activation_threshold = (1.0f + FunctionsNative::a_value(i, j, k, genome.activation_threshold, I, J, K)) / 2.0f;
				neuron.decay_factor = FunctionsNative::d_value(i, j, k, genome.decay_factor, I, J, K);
				neuron.retention_factor = (1.0f - std::exp(-1.0f - FunctionsNative::a_value(i, j, k, genome.retention_factor, I, J, K))) / retention_normalization;
				neuron.polarity_factor = FunctionsNative::a_value(i, j, k, genome.polarity_factor, I, J, K) >= first_polarity ? 1 : -1;
				neuron.hebbian_plasticity_rate = (1.0f + FunctionsNative::a_value(i, j, k, genome.hebbian_plasticity_rate, I, J, K)) / 4.0f;
				neuron.modulatory_release_factor = (1.0f + FunctionsNative::a_value(i, j, k, genome.modulatory_release_factor, I, J, K)) / 2.0f;
				neuron.modulatory_sensitivity = FunctionsNative::a_value(i, j, k, genome.modulatory_sensitivity, I, J, K);
				neuron.modulatory_field = 0.0f;
				neuron.connections_value = FunctionsNative::a_value(i, j, k, genome.connections, I, J, K);
				brain.neurons[static_cast<size_t>(creation_index)] = std::move(neuron);
			},
			[&](int32_t completed) {
				_report_progress(progress_callback, static_cast<int32_t>(16.0f * static_cast<float>(completed) / static_cast<float>(total_neurons)));
			});
}

bool BrainBuilderNative::_set_input_output_neurons(NativeBrain &brain, int32_t num_inputs, int32_t num_outputs, const Callable &progress_callback) const {
	const int32_t I = brain.size.x;
	const int32_t IJ = I * brain.size.y;
	const int32_t total = static_cast<int32_t>(brain.neurons.size());
	const int32_t channels = num_inputs + num_outputs;
	brain.role_selection_gap = 0;
	if (channels > 0) {
		const int32_t quotient = total / channels;
		// Strictly less than floor(N/m), including when the quotient is prime.
		for (int32_t candidate = quotient - 1; candidate >= 2; --candidate) {
			bool prime = true;
			for (int32_t divisor = 2; static_cast<int64_t>(divisor) * divisor <= candidate; ++divisor) {
				if (candidate % divisor == 0) { prime = false; break; }
			}
			if (prime) { brain.role_selection_gap = candidate; break; }
		}
		if (brain.role_selection_gap == 0) {
			_report_progress(progress_callback, 75);
			return false;
		}
	}

	// Creation order (i,j,k) is not spatial linear order (k,j,i). Build
	// the spatial map explicitly and never reinterpret source indices.
	std::vector<int32_t> by_spatial(static_cast<size_t>(total), -1);
	std::vector<int32_t> incoming(static_cast<size_t>(total), 0);
	for (int32_t index = 0; index < total; ++index) {
		const NativeNeuron &neuron = brain.neurons[static_cast<size_t>(index)];
		by_spatial[static_cast<size_t>(_spatial_index(neuron.position, I, IJ))] = index;
		for (const NativeConnection &edge : neuron.connections) {
			++incoming[static_cast<size_t>(edge.target_spatial_index)];
		}
	}
	std::vector<uint8_t> occupied(static_cast<size_t>(total), 0);
	std::vector<int32_t> selected_inputs;
	std::vector<int32_t> selected_outputs;
	selected_inputs.reserve(num_inputs);
	selected_outputs.reserve(num_outputs);
	bool valid = true;
	for (int32_t channel = 0; channel < channels; ++channel) {
		const bool is_input = channel < num_inputs;
		const int64_t preferred = (static_cast<int64_t>(channel + 1) * brain.role_selection_gap - 1) % total;
		int32_t selected = -1;
		for (int32_t advance = 0; advance < total; ++advance) {
			const int32_t spatial = static_cast<int32_t>((preferred + advance) % total);
			if (occupied[static_cast<size_t>(spatial)] != 0) continue;
			const int32_t neuron_index = by_spatial[static_cast<size_t>(spatial)];
			const NativeNeuron &neuron = brain.neurons[static_cast<size_t>(neuron_index)];
			const int32_t degree = is_input ? static_cast<int32_t>(neuron.connections.size()) : incoming[static_cast<size_t>(spatial)];
			if (degree < 2) continue;
			selected = spatial;
			break;
		}
		if (selected < 0) { valid = false; break; }
		occupied[static_cast<size_t>(selected)] = 1;
		if (is_input) selected_inputs.push_back(selected);
		else selected_outputs.push_back(selected);
	}

	std::vector<NativeNeuron> ordered;
	ordered.reserve(static_cast<size_t>(total));
	for (int32_t spatial : selected_inputs) {
		NativeNeuron neuron = std::move(brain.neurons[static_cast<size_t>(by_spatial[static_cast<size_t>(spatial)])]);
		neuron.input = true;
		neuron.activation_threshold = 0.0f;
		neuron.retention_factor = 0.0f;
		neuron.polarity_factor = 1;
		ordered.push_back(std::move(neuron));
	}
	for (int32_t spatial = 0; spatial < total; ++spatial) {
		if (occupied[static_cast<size_t>(spatial)] == 0) {
			ordered.push_back(std::move(brain.neurons[static_cast<size_t>(by_spatial[static_cast<size_t>(spatial)])]));
		}
	}
	for (int32_t spatial : selected_outputs) {
		NativeNeuron neuron = std::move(brain.neurons[static_cast<size_t>(by_spatial[static_cast<size_t>(spatial)])]);
		neuron.output = true;
		ordered.push_back(std::move(neuron));
	}
	brain.neurons = std::move(ordered);
	_report_progress(progress_callback, 75);
	return valid;
}

void BrainBuilderNative::_create_connections(NativeBrain &brain, const std::vector<int32_t> &gene,
		const std::vector<int32_t> &search_gene, const Callable &progress_callback) const {
	const int32_t I = brain.size.x, J = brain.size.y, K = brain.size.z;
	const int32_t IJ = I * J;
	const int32_t total_neurons = static_cast<int32_t>(brain.neurons.size());
	// Region scalar uses double precision, like GDScript's scalar G/S in the GPU builder.
	double g_01 = 0.0;
	for (int32_t t = 0; t < static_cast<int32_t>(search_gene.size()); ++t)
		g_01 += std::sin(1.0 + static_cast<double>(t)) * std::cos(1.0 + static_cast<double>(search_gene[static_cast<size_t>(t)]));
	const double S = std::clamp((1.0 - std::sqrt(std::abs(std::cos(g_01)))) / 10.0, 0.0, 0.1);
	const int32_t radius_i = std::min(I-1, static_cast<int32_t>(std::ceil(S * static_cast<double>(I))));
	const int32_t radius_j = std::min(J-1, static_cast<int32_t>(std::ceil(S * static_cast<double>(J))));
	const int32_t radius_k = std::min(K-1, static_cast<int32_t>(std::ceil(S * static_cast<double>(K))));
	brain.initial_search_extent = static_cast<float>(S);
	brain.initial_search_radius_i = radius_i;
	brain.initial_search_radius_j = radius_j;
	brain.initial_search_radius_k = radius_k;
	const auto extra_offsets = supplementary_offsets(search_gene, I, J, K, radius_i, radius_j, radius_k);
	std::vector<float> energy(static_cast<size_t>(total_neurons));
	for (const auto &neuron : brain.neurons)
		energy[static_cast<size_t>(_spatial_index(neuron.position, I, IJ))] = neuron.connections_value;
	const Vector4 coefficients = FunctionsNative::h_coefficients_value(gene);
	std::vector<std::vector<NativeConnection>> results(static_cast<size_t>(total_neurons));

	run_parallel(total_neurons, [&](int32_t neuron_index) {
		const auto &source = brain.neurons[static_cast<size_t>(neuron_index)];
		if (brain.max_connections <= 0 || source.output) return;
		const Vector3i p = source.position;
		const float source_a = energy[static_cast<size_t>(_spatial_index(p, I, IJ))];
		std::vector<ConnectionCandidate> candidates;
		const int32_t min_i = std::max(0, p.x - radius_i), max_i = std::min(I - 1, p.x + radius_i);
		const int32_t min_j = std::max(0, p.y - radius_j), max_j = std::min(J - 1, p.y + radius_j);
		const int32_t min_k = std::max(0, p.z - radius_k), max_k = std::min(K - 1, p.z + radius_k);
		candidates.reserve(static_cast<size_t>(max_i-min_i+1)*(max_j-min_j+1)*(max_k-min_k+1));
		for (int32_t u = min_i; u <= max_i; ++u)
		for (int32_t v = min_j; v <= max_j; ++v)
		for (int32_t w = min_k; w <= max_k; ++w) {
			if (u == p.x && v == p.y && w == p.z) continue;
			const int32_t target = u + I*v + IJ*w;
			const float h = FunctionsNative::h_value(source_a, energy[static_cast<size_t>(target)], coefficients);
			if (h > 0.0f) candidates.push_back({target,h});
		}
		const int32_t local_count = std::min<int32_t>(brain.max_connections, static_cast<int32_t>(candidates.size()));
		auto greater = [](const ConnectionCandidate &a,const ConnectionCandidate &b) {
			return (a.value > b.value) || (a.value == b.value && a.target_spatial_index < b.target_spatial_index);
		};
		if (local_count > 0) {
			if (static_cast<size_t>(local_count) < candidates.size())
				std::nth_element(candidates.begin(), candidates.begin()+local_count, candidates.end(), greater);
			NativeSortArray<ConnectionCandidate, decltype(greater)> sorter(greater);
			sorter.sort(candidates.data(), local_count);
		}
		auto &out = results[static_cast<size_t>(neuron_index)];
		out.reserve(static_cast<size_t>(std::min(brain.max_connections, total_neurons - 1)));
		for (int32_t c = 0; c < local_count; ++c)
			out.push_back({candidates[static_cast<size_t>(c)].target_spatial_index, candidates[static_cast<size_t>(c)].value, 0.0f, 0});

		// Every sign tested counts against the N_max budget, even if invalid or repeated.
		// Only prepare duplicate tracking when the complementary pass is actually needed.
		if (local_count == brain.max_connections || extra_offsets.empty()) return;
		std::unordered_set<int32_t> seen;
		seen.reserve(static_cast<size_t>(std::min(brain.max_connections, total_neurons - 1)));
		for (const auto &edge : out) seen.insert(edge.target_spatial_index);
		int32_t attempts = 0;
		for (const Offset3 &delta : extra_offsets) {
			for (int32_t sign = 0; sign < 8; ++sign) {
				if (attempts >= brain.max_connections || static_cast<int32_t>(out.size()) >= brain.max_connections) return;
				++attempts;
				const int32_t u = p.x + ((sign & 4) ? -delta.x : delta.x);
				const int32_t v = p.y + ((sign & 2) ? -delta.y : delta.y);
				const int32_t w = p.z + ((sign & 1) ? -delta.z : delta.z);
				if (u < 0 || u >= I || v < 0 || v >= J || w < 0 || w >= K) continue;
				if (u == p.x && v == p.y && w == p.z) continue;
				const int32_t target = u + I*v + IJ*w;
				if (!seen.insert(target).second) continue;
				const float h = FunctionsNative::h_value(source_a, energy[static_cast<size_t>(target)], coefficients);
				if (h > 0.0f) out.push_back({target,h,0.0f,1});
			}
		}
	}, [&](int32_t completed) {
		_report_progress(progress_callback,21+static_cast<int32_t>(54.0f*static_cast<float>(completed)/static_cast<float>(total_neurons)));
	});
	for (int32_t n=0;n<total_neurons;++n) brain.neurons[static_cast<size_t>(n)].connections=std::move(results[static_cast<size_t>(n)]);
	_report_progress(progress_callback,75);
}

bool BrainBuilderNative::_clear_invalid_connections(
		NativeBrain &brain,
		const std::vector<int32_t> &gene,
		const std::vector<int32_t> &input_influence_gene,
		const Callable &progress_callback) const {
	const int32_t I = brain.size.x;
	const int32_t J = brain.size.y;
	const int32_t IJ = I * J;
	const int32_t total_neurons = static_cast<int32_t>(brain.neurons.size());
	std::vector<int32_t> neuron_by_spatial_index(static_cast<size_t>(total_neurons), -1);
	std::vector<int32_t> incoming_connections(static_cast<size_t>(total_neurons), 0);

	for (int32_t neuron_index = 0; neuron_index < total_neurons; ++neuron_index) {
		const NativeNeuron &neuron = brain.neurons[static_cast<size_t>(neuron_index)];
		neuron_by_spatial_index[static_cast<size_t>(_spatial_index(neuron.position, I, IJ))] = neuron_index;
	}

	for (int32_t neuron_index = 0; neuron_index < total_neurons; ++neuron_index) {
		NativeNeuron &source = brain.neurons[static_cast<size_t>(neuron_index)];
		std::vector<NativeConnection> valid_connections;
		valid_connections.reserve(source.connections.size());

		for (const NativeConnection &connection : source.connections) {
			const int32_t target_index = neuron_by_spatial_index[static_cast<size_t>(connection.target_spatial_index)];
			const NativeNeuron &target = brain.neurons[static_cast<size_t>(target_index)];

			if (target.input) {
				continue;
			}
			if (source.input && target.output) {
				continue;
			}
			if (source.output) {
				continue;
			}

			valid_connections.push_back(connection);
			if (connection.construction_pass == 0) ++brain.first_pass_connections;
			else if (connection.construction_pass == 1) ++brain.second_pass_connections;
			++incoming_connections[static_cast<size_t>(connection.target_spatial_index)];
		}

		source.connections = std::move(valid_connections);
		_report_progress(
				progress_callback,
				75 + static_cast<int32_t>(4.0f * static_cast<float>(neuron_index + 1) / static_cast<float>(total_neurons)));
	}

	IsolationCounts before;
	for (const NativeNeuron &neuron : brain.neurons) {
		const int32_t spatial_index = _spatial_index(neuron.position, I, IJ);
		const int32_t incoming_count = incoming_connections[static_cast<size_t>(spatial_index)];

		if (neuron.input) {
			if (neuron.connections.empty()) {
				++before.input;
			}
		} else if (neuron.output) {
			if (incoming_count == 0) {
				++before.output;
			}
		} else if (incoming_count == 0) {
			++before.internal;
		}
	}
	brain.isolated_before_rescue = before;
	// Role rescue is diagnostic, not a substitute for eligible interface roles.
	brain.role_cleanup_valid = before.input == 0 && before.output == 0;
	const bool rescue_success = _rescue_isolated_neurons(
			brain,
			gene,
			neuron_by_spatial_index,
			incoming_connections,
			progress_callback);

	IsolationCounts after;
	int64_t internal_incoming_sum = 0;
	int32_t internal_neuron_count = 0;

	for (const NativeNeuron &neuron : brain.neurons) {
		const int32_t spatial_index = _spatial_index(neuron.position, I, IJ);
		const int32_t incoming_count = incoming_connections[static_cast<size_t>(spatial_index)];

		if (neuron.input) {
			if (neuron.connections.empty()) {
				++after.input;
			}
		} else if (neuron.output) {
			if (incoming_count == 0) {
				++after.output;
			}
		} else {
			if (incoming_count == 0) {
				++after.internal;
			}
			internal_incoming_sum += incoming_count;
			++internal_neuron_count;
		}
	}
	brain.isolated_after_rescue = after;

	const float mean_internal_incoming = internal_neuron_count == 0
			? 0.0f
			: static_cast<float>(internal_incoming_sum) / static_cast<float>(internal_neuron_count);
	brain.mean_initial_incoming = mean_internal_incoming;
	brain.input_gain = (1.0f + std::sin(FunctionsNative::g_value(true, true, input_influence_gene))) * mean_internal_incoming;
	const float input_min_weight = 1.0f / (1.0f + brain.input_gain);

	// Attenuate initial synaptic weights only after top-H selection, invalid-connection
	// filtering, and developmental rescue. This also covers rescued connections.
	// The same source-dependent factor applies to every outgoing connection.
	for (NativeNeuron &neuron : brain.neurons) {
		const float source_a = neuron.connections_value;
		const float attenuation = 1.0f - source_a * source_a;

		for (NativeConnection &connection : neuron.connections) {
			connection.weight = std::max(0.1f, connection.weight * attenuation);
			if (neuron.input) {
				// Preserve the pre-existing lower bound for input synapses.
				connection.weight = std::max(connection.weight, input_min_weight);
			}
		}
	}

	_report_progress(progress_callback, 83);
	return brain.role_cleanup_valid && rescue_success && after.internal == 0 && after.output == 0 && after.input == 0;
}

bool BrainBuilderNative::_rescue_isolated_neurons(
		NativeBrain &brain,
		const std::vector<int32_t> &gene,
		const std::vector<int32_t> &neuron_by_spatial_index,
		std::vector<int32_t> &incoming_connections,
		const Callable &progress_callback) const {
	const int32_t I = brain.size.x;
	const int32_t J = brain.size.y;
	const int32_t K = brain.size.z;
	const int32_t IJ = I * J;
	const int32_t rescue_max_connections = brain.max_connections;
	const Vector4 h_coefficients = FunctionsNative::h_coefficients_value(gene);
	std::vector<int32_t> isolated_receivers;
	bool rescue_success = true;
	RescueFailureCauses failure_causes;

	for (int32_t neuron_index = 0; neuron_index < static_cast<int32_t>(brain.neurons.size()); ++neuron_index) {
		const NativeNeuron &neuron = brain.neurons[static_cast<size_t>(neuron_index)];
		if (neuron.input) {
			continue;
		}
		const int32_t spatial_index = _spatial_index(neuron.position, I, IJ);
		if (incoming_connections[static_cast<size_t>(spatial_index)] == 0) {
			isolated_receivers.push_back(neuron_index);
		}
	}

	const int32_t isolated_receiver_count = static_cast<int32_t>(isolated_receivers.size());
	for (int32_t isolated_index = 0; isolated_index < isolated_receiver_count; ++isolated_index) {
		const int32_t target_index = isolated_receivers[static_cast<size_t>(isolated_index)];
		const NativeNeuron &target = brain.neurons[static_cast<size_t>(target_index)];
		const Vector3i target_position = target.position;
		const int32_t target_spatial_index = _spatial_index(target_position, I, IJ);
		const float target_connection_value = target.connections_value;
		int32_t best_source_index = -1;
		float best_h = 0.0f;
		int32_t structural_candidates = 0;
		int32_t positive_h_candidates = 0;
		int32_t free_positive_h_candidates = 0;

		// Six orthogonal neighbors only, not entire adjacent slices.
		constexpr int32_t neighbors[6][3] = {{0,0,-1},{0,-1,0},{-1,0,0},{1,0,0},{0,1,0},{0,0,1}};
		for (const auto &shift : neighbors) {
			const int32_t u = target_position.x + shift[0];
			const int32_t v = target_position.y + shift[1];
			const int32_t w = target_position.z + shift[2];
			if (u < 0 || u >= I || v < 0 || v >= J || w < 0 || w >= K) continue;
			const int32_t spatial = u + I*v + IJ*w;
			const auto &source = brain.neurons[static_cast<size_t>(neuron_by_spatial_index[static_cast<size_t>(spatial)])];
			if (source.output || (target.output && source.input)) continue;
			++structural_candidates;
			const float h = FunctionsNative::h_value(source.connections_value, target_connection_value, h_coefficients);
			++brain.rescue_candidates_receiver;
			const float h2 = h*h;
			if (!(h2 > 0.0f)) continue;
			++positive_h_candidates;
			if (static_cast<int32_t>(source.connections.size()) >= rescue_max_connections) continue;
			++free_positive_h_candidates;
			if (h2 > best_h) { best_h = h2; best_source_index = neuron_by_spatial_index[static_cast<size_t>(spatial)]; }
		}

		if (best_source_index < 0) {
			rescue_success = false;
			if (structural_candidates == 0) {
				++failure_causes.no_structural_candidate;
			} else if (positive_h_candidates == 0) {
				++failure_causes.no_positive_h_candidate;
			} else if (free_positive_h_candidates == 0) {
				++failure_causes.positive_h_candidates_saturated;
			}
			continue;
		}

		brain.neurons[static_cast<size_t>(best_source_index)].connections.push_back({
			target_spatial_index, best_h, 0.0f, 2
		});
		incoming_connections[static_cast<size_t>(target_spatial_index)] = 1;
		_report_progress(
				progress_callback,
				79 + static_cast<int32_t>(2.0f * static_cast<float>(isolated_index + 1) / static_cast<float>(isolated_receiver_count)));
	}

	std::vector<int32_t> isolated_inputs;
	for (int32_t neuron_index = 0; neuron_index < static_cast<int32_t>(brain.neurons.size()); ++neuron_index) {
		const NativeNeuron &neuron = brain.neurons[static_cast<size_t>(neuron_index)];
		if (neuron.input && neuron.connections.empty()) {
			isolated_inputs.push_back(neuron_index);
		}
	}

	const int32_t isolated_input_count = static_cast<int32_t>(isolated_inputs.size());
	for (int32_t isolated_index = 0; isolated_index < isolated_input_count; ++isolated_index) {
		const int32_t source_index = isolated_inputs[static_cast<size_t>(isolated_index)];
		NativeNeuron &source = brain.neurons[static_cast<size_t>(source_index)];

		if (static_cast<int32_t>(source.connections.size()) >= rescue_max_connections) {
			rescue_success = false;
			++failure_causes.input_source_saturated;
			continue;
		}

		const Vector3i source_position = source.position;
		const float source_connection_value = source.connections_value;
		int32_t best_target_index = -1;
		int32_t best_target_spatial_index = -1;
		float best_input_h = 0.0f;
		int32_t structural_targets = 0;
		int32_t positive_h_targets = 0;

		constexpr int32_t neighbors[6][3] = {{0,0,-1},{0,-1,0},{-1,0,0},{1,0,0},{0,1,0},{0,0,1}};
		for (const auto &shift : neighbors) {
			const int32_t u = source_position.x + shift[0];
			const int32_t v = source_position.y + shift[1];
			const int32_t w = source_position.z + shift[2];
			if (u < 0 || u >= I || v < 0 || v >= J || w < 0 || w >= K) continue;
			const int32_t spatial = u + I*v + IJ*w;
			const int32_t target_index = neuron_by_spatial_index[static_cast<size_t>(spatial)];
			const NativeNeuron &target = brain.neurons[static_cast<size_t>(target_index)];
			if (target.input || target.output) continue;
			++structural_targets;
			const float h = FunctionsNative::h_value(source_connection_value, target.connections_value, h_coefficients);
			++brain.rescue_candidates_input;
			const float h2 = h*h;
			if (!(h2 > 0.0f)) continue;
			++positive_h_targets;
			if (h2 > best_input_h) { best_input_h = h2; best_target_index = target_index; best_target_spatial_index = spatial; }
		}

		if (best_target_index < 0) {
			rescue_success = false;
			if (structural_targets == 0) {
				++failure_causes.no_structural_candidate;
			} else if (positive_h_targets == 0) {
				++failure_causes.no_positive_h_candidate;
			}
			continue;
		}

		source.connections.push_back({
			best_target_spatial_index, best_input_h, 0.0f, 2
		});
		++incoming_connections[static_cast<size_t>(best_target_spatial_index)];
		_report_progress(
				progress_callback,
				81 + static_cast<int32_t>(2.0f * static_cast<float>(isolated_index + 1) / static_cast<float>(isolated_input_count)));
	}

	brain.rescue_failure_causes = failure_causes;
	_report_progress(progress_callback, 83);
	return rescue_success;
}

Vector3 BrainBuilderNative::_modulatory_dynamics(const std::vector<int32_t> &gene) const {
	return Vector3(
			1.0f / (1.0f + std::exp(-FunctionsNative::g_value(true, false, gene))),
			1.0f / (1.0f + std::exp(-FunctionsNative::g_value(false, true, gene))),
			1.0f / (1.0f + std::exp(-FunctionsNative::g_value(true, true, gene))));
}

Dictionary BrainBuilderNative::_to_dictionary(const NativeBrain &brain, const Callable &progress_callback) const {
	Dictionary result;
	result["size"] = brain.size;
	result["beta"] = brain.beta;
	result["max_connections"] = brain.max_connections;
	result["connections_density"] = brain.connections_density;
	result["initial_connections_first_pass"] = brain.first_pass_connections;
	result["initial_connections_second_pass"] = brain.second_pass_connections;
	result["initial_local_connections"] = brain.first_pass_connections;
	result["initial_additional_connections"] = brain.second_pass_connections;
	result["initial_search_extent"] = brain.initial_search_extent;
	result["initial_search_radius_i"] = brain.initial_search_radius_i;
	result["initial_search_radius_j"] = brain.initial_search_radius_j;
	result["initial_search_radius_k"] = brain.initial_search_radius_k;
	result["refractory_strength"] = brain.refractory_strength;
	result["exponential_factor"] = brain.exponential_factor;
	result["structural_plasticity"] = brain.structural_plasticity;
	result["lambda"] = brain.lambda;
	result["nu"] = brain.nu;
	result["mu"] = brain.mu;
	result["input_gain"] = brain.input_gain;
	result["mean_initial_incoming"] = brain.mean_initial_incoming;
	result["role_selection_gap"] = brain.role_selection_gap;
	result["role_selection_valid"] = brain.role_selection_valid;
	result["role_cleanup_valid"] = brain.role_cleanup_valid;
	result["valid"] = brain.valid;

	Dictionary isolated_before;
	isolated_before["internal"] = brain.isolated_before_rescue.internal;
	isolated_before["output"] = brain.isolated_before_rescue.output;
	isolated_before["input"] = brain.isolated_before_rescue.input;
	result["isolated_before_rescue"] = isolated_before;

	Dictionary isolated_after;
	isolated_after["internal"] = brain.isolated_after_rescue.internal;
	isolated_after["output"] = brain.isolated_after_rescue.output;
	isolated_after["input"] = brain.isolated_after_rescue.input;
	result["isolated_after_rescue"] = isolated_after;

	Dictionary failure_causes;
	failure_causes["no_structural_candidate"] = brain.rescue_failure_causes.no_structural_candidate;
	failure_causes["no_positive_h_candidate"] = brain.rescue_failure_causes.no_positive_h_candidate;
	failure_causes["positive_h_candidates_saturated"] = brain.rescue_failure_causes.positive_h_candidates_saturated;
	failure_causes["input_source_saturated"] = brain.rescue_failure_causes.input_source_saturated;
	result["rescue_failure_causes"] = failure_causes;
	result["rescue_candidates_receiver"] = brain.rescue_candidates_receiver;
	result["rescue_candidates_input"] = brain.rescue_candidates_input;
	result["rescue_candidates_total"] = brain.rescue_candidates_receiver + brain.rescue_candidates_input;

	const int64_t neuron_count = static_cast<int64_t>(brain.neurons.size());
	Array neurons;
	neurons.resize(neuron_count);
	const int32_t I = brain.size.x;
	const int32_t J = brain.size.y;

	// Cache STRING Variant keys, not StringName keys, to preserve the exact key types
	// of the previous Dictionary representation without constructing them per field.
	const Variant position_key("position");
	const Variant input_key("input");
	const Variant output_key("output");
	const Variant activation_threshold_key("activation_threshold");
	const Variant decay_factor_key("decay_factor");
	const Variant retention_factor_key("retention_factor");
	const Variant polarity_factor_key("polarity_factor");
	const Variant hebbian_plasticity_rate_key("hebbian_plasticity_rate");
	const Variant modulatory_release_factor_key("modulatory_release_factor");
	const Variant modulatory_sensitivity_key("modulatory_sensitivity");
	const Variant modulatory_field_key("modulatory_field");
	const Variant connections_value_key("connections_value");
	const Variant connections_key("connections");
	const Variant target_key("target");
	const Variant weight_key("weight");
	const Variant eligibility_trace_key("eligibility_trace");

	// Spatial indices are reused by millions of connections. Compute their
	// coordinates once, rather than repeating integer division for every edge.
	std::vector<Vector3i> positions_by_spatial_index(static_cast<size_t>(neuron_count));
	for (int32_t spatial_index = 0; spatial_index < static_cast<int32_t>(neuron_count); ++spatial_index) {
		positions_by_spatial_index[static_cast<size_t>(spatial_index)] = _position_from_spatial_index(spatial_index, I, J);
	}

	// Account for each neuron and connection object exported to Godot.
	// The percentage therefore continues to advance during serialization.
	int64_t total_objects = neuron_count;
	for (const NativeNeuron &neuron : brain.neurons) {
		total_objects += static_cast<int64_t>(neuron.connections.size());
	}
	int64_t exported_objects = 0;
	int32_t last_export_percent = 83;
	_report_progress(progress_callback, last_export_percent);

	for (int32_t neuron_index = 0; neuron_index < static_cast<int32_t>(neuron_count); ++neuron_index) {
		const NativeNeuron &native_neuron = brain.neurons[static_cast<size_t>(neuron_index)];
		Dictionary neuron;
		neuron[position_key] = native_neuron.position;
		neuron[input_key] = native_neuron.input;
		neuron[output_key] = native_neuron.output;
		neuron[activation_threshold_key] = native_neuron.activation_threshold;
		neuron[decay_factor_key] = native_neuron.decay_factor;
		neuron[retention_factor_key] = native_neuron.retention_factor;
		neuron[polarity_factor_key] = native_neuron.polarity_factor;
		neuron[hebbian_plasticity_rate_key] = native_neuron.hebbian_plasticity_rate;
		neuron[modulatory_release_factor_key] = native_neuron.modulatory_release_factor;
		neuron[modulatory_sensitivity_key] = native_neuron.modulatory_sensitivity;
		neuron[modulatory_field_key] = native_neuron.modulatory_field;
		neuron[connections_value_key] = native_neuron.connections_value;

		const int64_t connection_count = static_cast<int64_t>(native_neuron.connections.size());
		Array connections;
		connections.resize(connection_count);
		for (int32_t connection_index = 0; connection_index < static_cast<int32_t>(connection_count); ++connection_index) {
			const NativeConnection &native_connection = native_neuron.connections[static_cast<size_t>(connection_index)];
			Dictionary connection;
			connection[target_key] = positions_by_spatial_index[static_cast<size_t>(native_connection.target_spatial_index)];
			connection[weight_key] = native_connection.weight;
			connection[eligibility_trace_key] = native_connection.eligibility_trace;
			connections[connection_index] = connection;
		}

		neuron[connections_key] = connections;
		neurons[neuron_index] = neuron;

		exported_objects += 1 + connection_count;
		const int32_t percent = 83 + static_cast<int32_t>(
				16.0 * static_cast<double>(exported_objects) / static_cast<double>(std::max<int64_t>(total_objects, 1)));
		if (percent > last_export_percent) {
			last_export_percent = percent;
			_report_progress(progress_callback, percent);
		}
	}

	result["neurons"] = neurons;
	return result;
}

void BrainBuilderNative::_report_progress(const Callable &progress_callback, int32_t percent) const {
	if (!progress_callback.is_valid()) {
		return;
	}
	const int32_t clamped_percent = std::max(0, std::min(100, percent));
	progress_callback.call(clamped_percent);
}

int32_t BrainBuilderNative::_spatial_index(const Vector3i &position, int32_t I, int32_t IJ) {
	return position.x + I * position.y + IJ * position.z;
}

Vector3i BrainBuilderNative::_position_from_spatial_index(int32_t spatial_index, int32_t I, int32_t J) {
	const int32_t IJ = I * J;
	const int32_t z = spatial_index / IJ;
	const int32_t within_slice = spatial_index - z * IJ;
	const int32_t y = within_slice / I;
	const int32_t x = within_slice - y * I;
	return Vector3i(x, y, z);
}
