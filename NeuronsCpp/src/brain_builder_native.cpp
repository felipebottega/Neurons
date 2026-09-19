#include "brain_builder_native.h"
#include "functions_native.h"

#include <godot_cpp/core/class_db.hpp>
#include <godot_cpp/variant/array.hpp>
#include <godot_cpp/variant/vector4.hpp>
#include <godot_cpp/variant/utility_functions.hpp>

#include <algorithm>
#include <atomic>
#include <chrono>
#include <cmath>
#include <cstdint>
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
	double value = 0.0;
};

}

void BrainBuilderNative::_bind_methods() {
	ClassDB::bind_method(
			D_METHOD("build", "genome", "size", "num_inputs", "num_outputs", "progress_callback"),
			&BrainBuilderNative::build,
			DEFVAL(Callable()));
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
		const Callable &progress_callback) {
	if (genome.is_null()) {
		UtilityFunctions::push_error("GenomeNative is null.");
		return Dictionary();
	}
	if (size.x < 2 || size.y < 2 || size.z < 2) {
		UtilityFunctions::push_error("Brain dimensions I, J, K must all be at least 2.");
		return Dictionary();
	}

	_report_progress(progress_callback, 0);

	NativeBrain brain;
	brain.size = size;
	brain.beta = FunctionsNative::beta_value(genome->beta, size.x, size.y);
	brain.max_connections = static_cast<int32_t>(std::floor(FunctionsNative::f_value(brain.beta, genome->max_connections)));
	brain.refractory_strength = FunctionsNative::f_value(0.4, genome->refractory_strength);
	brain.exponential_factor = FunctionsNative::f_value(0.4, genome->exponential_factor);
	brain.structural_plasticity = FunctionsNative::p_value(genome->structural_plasticity);

	const Vector3 modulatory_dynamics = _modulatory_dynamics(genome->modulatory_dynamics);
	brain.lambda = modulatory_dynamics.x;
	brain.nu = modulatory_dynamics.y;
	brain.mu = modulatory_dynamics.z;

	_create_neurons(brain, *genome.ptr(), progress_callback);
	_set_input_output_neurons(brain, static_cast<int32_t>(num_inputs), static_cast<int32_t>(num_outputs), progress_callback);
	_create_connections(brain, genome->connections, progress_callback);
	brain.valid = _clear_invalid_connections(brain, genome->connections, genome->input_influence, progress_callback);
	_report_progress(progress_callback, 100);

	return _to_dictionary(brain);
}

void BrainBuilderNative::_create_neurons(NativeBrain &brain, const GenomeNative &genome, const Callable &progress_callback) const {
	const int32_t I = brain.size.x;
	const int32_t J = brain.size.y;
	const int32_t K = brain.size.z;
	const int32_t total_neurons = I * J * K;
	const double first_polarity = FunctionsNative::a_value(0, 0, 0, genome.polarity_factor, I, J, K);
	const double retention_normalization = 1.0 - std::exp(-2.0);

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
				neuron.activation_threshold = (1.0 + FunctionsNative::a_value(i, j, k, genome.activation_threshold, I, J, K)) / 2.0;
				neuron.decay_factor = FunctionsNative::d_value(i, j, k, genome.decay_factor, I, J, K);
				neuron.retention_factor = (1.0 - std::exp(-1.0 - FunctionsNative::a_value(i, j, k, genome.retention_factor, I, J, K))) / retention_normalization;
				neuron.polarity_factor = FunctionsNative::a_value(i, j, k, genome.polarity_factor, I, J, K) >= first_polarity ? 1 : -1;
				neuron.hebbian_plasticity_rate = (1.0 + FunctionsNative::a_value(i, j, k, genome.hebbian_plasticity_rate, I, J, K)) / 4.0;
				neuron.modulatory_release_factor = (1.0 + FunctionsNative::a_value(i, j, k, genome.modulatory_release_factor, I, J, K)) / 2.0;
				neuron.modulatory_sensitivity = FunctionsNative::a_value(i, j, k, genome.modulatory_sensitivity, I, J, K);
				neuron.modulatory_field = 0.0;
				neuron.connections_value = FunctionsNative::a_value(i, j, k, genome.connections, I, J, K);
				brain.neurons[static_cast<size_t>(creation_index)] = std::move(neuron);
			},
			[&](int32_t completed) {
				_report_progress(progress_callback, static_cast<int32_t>(20.0 * static_cast<double>(completed) / static_cast<double>(total_neurons)));
			});
}

void BrainBuilderNative::_set_input_output_neurons(NativeBrain &brain, int32_t num_inputs, int32_t num_outputs, const Callable &progress_callback) const {
	const int32_t K = brain.size.z;
	const int32_t neurons_per_slice = brain.size.x * brain.size.y;
	std::vector<std::vector<NativeNeuron>> slices(static_cast<size_t>(K));

	for (std::vector<NativeNeuron> &slice : slices) {
		slice.reserve(static_cast<size_t>(neurons_per_slice));
	}

	for (NativeNeuron &neuron : brain.neurons) {
		slices[static_cast<size_t>(neuron.position.z)].push_back(std::move(neuron));
	}

	for (int32_t k = 0; k < K; ++k) {
		auto comparator = [](const NativeNeuron &a, const NativeNeuron &b) {
			return a.connections_value < b.connections_value;
		};
		NativeSortArray<NativeNeuron, decltype(comparator)> sorter(comparator);
		std::vector<NativeNeuron> &slice = slices[static_cast<size_t>(k)];
		sorter.sort(slice.data(), static_cast<int64_t>(slice.size()));
	}

	std::vector<NativeNeuron> reordered;
	reordered.reserve(brain.neurons.size());
	for (int32_t rank = 0; rank < neurons_per_slice; ++rank) {
		for (int32_t k = 0; k < K; ++k) {
			reordered.push_back(std::move(slices[static_cast<size_t>(k)][static_cast<size_t>(rank)]));
		}
	}
	brain.neurons = std::move(reordered);

	for (int32_t i = 0; i < num_inputs; ++i) {
		NativeNeuron &neuron = brain.neurons[static_cast<size_t>(i)];
		neuron.input = true;
		neuron.activation_threshold = 0.0;
		neuron.retention_factor = 0.0;
		neuron.polarity_factor = 1;
	}

	for (int32_t i = 0; i < num_outputs; ++i) {
		brain.neurons[brain.neurons.size() - 1 - static_cast<size_t>(i)].output = true;
	}

	_report_progress(progress_callback, 30);
}

void BrainBuilderNative::_create_connections(NativeBrain &brain, const std::vector<int32_t> &gene, const Callable &progress_callback) const {
	const int32_t I = brain.size.x;
	const int32_t J = brain.size.y;
	const int32_t K = brain.size.z;
	const int32_t IJ = I * J;
	const int32_t total_neurons = static_cast<int32_t>(brain.neurons.size());
	std::vector<double> connection_values(static_cast<size_t>(total_neurons), 0.0);

	for (const NativeNeuron &neuron : brain.neurons) {
		connection_values[static_cast<size_t>(_spatial_index(neuron.position, I, IJ))] = neuron.connections_value;
	}

	std::vector<std::vector<NativeConnection>> connection_results(static_cast<size_t>(total_neurons));
	const Vector4 h_coefficients = FunctionsNative::h_coefficients_value(gene);

	run_parallel(
			total_neurons,
			[&](int32_t neuron_index) {
				const NativeNeuron &source_neuron = brain.neurons[static_cast<size_t>(neuron_index)];
				if (brain.max_connections <= 0 || source_neuron.output) {
					return;
				}

				const Vector3i pos = source_neuron.position;
				const int32_t source_linear_index = _spatial_index(pos, I, IJ);
				const double A_ijk = connection_values[static_cast<size_t>(source_linear_index)];
				const int32_t target_k_min = std::max(0, pos.z - 2);
				const int32_t target_k_max = std::min(K - 1, pos.z + 2);
				std::vector<ConnectionCandidate> candidates;
				candidates.reserve(static_cast<size_t>(I * J * (target_k_max - target_k_min + 1) - 1));

				for (int32_t u = 0; u < I; ++u) {
					const bool u_equals_posx = u == pos.x;
					for (int32_t v = 0; v < J; ++v) {
						const int32_t Iv = I * v;
						const bool v_equals_posy = v == pos.y;
						for (int32_t w = target_k_min; w <= target_k_max; ++w) {
							if (u_equals_posx && v_equals_posy && w == pos.z) {
								continue;
							}
							const int32_t target_linear_index = u + Iv + IJ * w;
							const double A_uvw = connection_values[static_cast<size_t>(target_linear_index)];
							const double h = FunctionsNative::h_value(A_ijk, A_uvw, h_coefficients);
							if (h > 0.0) {
								candidates.push_back({ target_linear_index, h });
							}
						}
					}
				}

				auto comparator = [](const ConnectionCandidate &a, const ConnectionCandidate &b) {
					return a.value > b.value;
				};
				NativeSortArray<ConnectionCandidate, decltype(comparator)> sorter(comparator);
				sorter.sort(candidates.data(), static_cast<int64_t>(candidates.size()));

				const int32_t connection_count = std::min<int32_t>(brain.max_connections, static_cast<int32_t>(candidates.size()));
				std::vector<NativeConnection> &connections = connection_results[static_cast<size_t>(neuron_index)];
				connections.resize(static_cast<size_t>(connection_count));
				for (int32_t index = 0; index < connection_count; ++index) {
					connections[static_cast<size_t>(index)] = {
						candidates[static_cast<size_t>(index)].target_spatial_index,
						candidates[static_cast<size_t>(index)].value,
						0.0
					};
				}
			},
			[&](int32_t completed) {
				_report_progress(progress_callback, 30 + static_cast<int32_t>(65.0 * static_cast<double>(completed) / static_cast<double>(total_neurons)));
			});

	for (int32_t neuron_index = 0; neuron_index < total_neurons; ++neuron_index) {
		brain.neurons[static_cast<size_t>(neuron_index)].connections = std::move(connection_results[static_cast<size_t>(neuron_index)]);
	}

	_report_progress(progress_callback, 95);
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
			++incoming_connections[static_cast<size_t>(connection.target_spatial_index)];
		}

		source.connections = std::move(valid_connections);
		_report_progress(
				progress_callback,
				95 + static_cast<int32_t>(3.0 * static_cast<double>(neuron_index + 1) / static_cast<double>(total_neurons)));
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

	const double mean_internal_incoming = internal_neuron_count == 0
			? 0.0
			: static_cast<double>(internal_incoming_sum) / static_cast<double>(internal_neuron_count);
	brain.input_gain = (1.0 + std::sin(FunctionsNative::g_value(true, true, input_influence_gene))) * mean_internal_incoming;
	const double input_min_weight = 1.0 / (1.0 + brain.input_gain);

	for (NativeNeuron &neuron : brain.neurons) {
		if (!neuron.input) {
			continue;
		}
		for (NativeConnection &connection : neuron.connections) {
			connection.weight = std::max(connection.weight, input_min_weight);
		}
	}

	return rescue_success && after.internal == 0 && after.output == 0 && after.input == 0;
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
	const int32_t rescue_max_connections = static_cast<int32_t>(std::ceil(1.01 * static_cast<double>(brain.max_connections)) + 1.0);
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
		const double target_connection_value = target.connections_value;
		const int32_t source_k_min = std::max(0, target_position.z - 2);
		const int32_t source_k_max = std::min(K - 1, target_position.z + 2);
		int32_t best_source_index = -1;
		double best_h = 0.0;
		int32_t structural_candidates = 0;
		int32_t positive_h_candidates = 0;
		int32_t free_positive_h_candidates = 0;

		for (int32_t source_k = source_k_min; source_k <= source_k_max; ++source_k) {
			const int32_t slice_offset = IJ * source_k;
			for (int32_t source_j = 0; source_j < J; ++source_j) {
				const int32_t row_offset = slice_offset + I * source_j;
				for (int32_t source_i = 0; source_i < I; ++source_i) {
					const int32_t source_spatial_index = row_offset + source_i;
					if (source_spatial_index == target_spatial_index) {
						continue;
					}

					const int32_t source_index = neuron_by_spatial_index[static_cast<size_t>(source_spatial_index)];
					const NativeNeuron &source = brain.neurons[static_cast<size_t>(source_index)];
					if (source.output) {
						continue;
					}
					if (target.output && source.input) {
						continue;
					}

					++structural_candidates;
					const double h = FunctionsNative::h_value(source.connections_value, target_connection_value, h_coefficients);
					if (h <= 0.0) {
						continue;
					}
					++positive_h_candidates;
					if (static_cast<int32_t>(source.connections.size()) >= rescue_max_connections) {
						continue;
					}
					++free_positive_h_candidates;
					if (h > best_h) {
						best_h = h;
						best_source_index = source_index;
					}
				}
			}
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
			target_spatial_index,
			best_h,
			0.0
		});
		incoming_connections[static_cast<size_t>(target_spatial_index)] = 1;
		_report_progress(
				progress_callback,
				98 + static_cast<int32_t>(static_cast<double>(isolated_index + 1) / static_cast<double>(isolated_receiver_count)));
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
		const double source_connection_value = source.connections_value;
		const int32_t target_k_min = std::max(0, source_position.z - 2);
		const int32_t target_k_max = std::min(K - 1, source_position.z + 2);
		int32_t best_target_index = -1;
		int32_t best_target_spatial_index = -1;
		double best_input_h = 0.0;
		int32_t structural_targets = 0;
		int32_t positive_h_targets = 0;

		for (int32_t target_k = target_k_min; target_k <= target_k_max; ++target_k) {
			const int32_t slice_offset = IJ * target_k;
			for (int32_t target_j = 0; target_j < J; ++target_j) {
				const int32_t row_offset = slice_offset + I * target_j;
				for (int32_t target_i = 0; target_i < I; ++target_i) {
					const int32_t target_spatial_index = row_offset + target_i;
					const int32_t target_index = neuron_by_spatial_index[static_cast<size_t>(target_spatial_index)];
					const NativeNeuron &target = brain.neurons[static_cast<size_t>(target_index)];

					if (target.input || target.output) {
						continue;
					}

					++structural_targets;
					const double h = FunctionsNative::h_value(source_connection_value, target.connections_value, h_coefficients);
					if (h <= 0.0) {
						continue;
					}
					++positive_h_targets;
					if (h > best_input_h) {
						best_input_h = h;
						best_target_index = target_index;
						best_target_spatial_index = target_spatial_index;
					}
				}
			}
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
			best_target_spatial_index,
			best_input_h,
			0.0
		});
		++incoming_connections[static_cast<size_t>(best_target_spatial_index)];
		_report_progress(
				progress_callback,
				99 + static_cast<int32_t>(static_cast<double>(isolated_index + 1) / static_cast<double>(isolated_input_count)));
	}

	brain.rescue_failure_causes = failure_causes;
	_report_progress(progress_callback, 100);
	return rescue_success;
}

Vector3 BrainBuilderNative::_modulatory_dynamics(const std::vector<int32_t> &gene) const {
	return Vector3(
			1.0 / (1.0 + std::exp(-FunctionsNative::g_value(true, false, gene))),
			1.0 / (1.0 + std::exp(-FunctionsNative::g_value(false, true, gene))),
			1.0 / (1.0 + std::exp(-FunctionsNative::g_value(true, true, gene))));
}

Dictionary BrainBuilderNative::_to_dictionary(const NativeBrain &brain) const {
	Dictionary result;
	result["size"] = brain.size;
	result["beta"] = brain.beta;
	result["max_connections"] = brain.max_connections;
	result["refractory_strength"] = brain.refractory_strength;
	result["exponential_factor"] = brain.exponential_factor;
	result["structural_plasticity"] = brain.structural_plasticity;
	result["lambda"] = brain.lambda;
	result["nu"] = brain.nu;
	result["mu"] = brain.mu;
	result["input_gain"] = brain.input_gain;
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

	Array neurons;
	neurons.resize(static_cast<int64_t>(brain.neurons.size()));
	const int32_t I = brain.size.x;
	const int32_t J = brain.size.y;

	for (int32_t neuron_index = 0; neuron_index < static_cast<int32_t>(brain.neurons.size()); ++neuron_index) {
		const NativeNeuron &native_neuron = brain.neurons[static_cast<size_t>(neuron_index)];
		Dictionary neuron;
		neuron["position"] = native_neuron.position;
		neuron["input"] = native_neuron.input;
		neuron["output"] = native_neuron.output;
		neuron["activation_threshold"] = native_neuron.activation_threshold;
		neuron["decay_factor"] = native_neuron.decay_factor;
		neuron["retention_factor"] = native_neuron.retention_factor;
		neuron["polarity_factor"] = native_neuron.polarity_factor;
		neuron["hebbian_plasticity_rate"] = native_neuron.hebbian_plasticity_rate;
		neuron["modulatory_release_factor"] = native_neuron.modulatory_release_factor;
		neuron["modulatory_sensitivity"] = native_neuron.modulatory_sensitivity;
		neuron["modulatory_field"] = native_neuron.modulatory_field;
		neuron["connections_value"] = native_neuron.connections_value;

		Array connections;
		connections.resize(static_cast<int64_t>(native_neuron.connections.size()));
		for (int32_t connection_index = 0; connection_index < static_cast<int32_t>(native_neuron.connections.size()); ++connection_index) {
			const NativeConnection &native_connection = native_neuron.connections[static_cast<size_t>(connection_index)];
			Dictionary connection;
			connection["target"] = _position_from_spatial_index(native_connection.target_spatial_index, I, J);
			connection["weight"] = native_connection.weight;
			connection["eligibility_trace"] = native_connection.eligibility_trace;
			connections[connection_index] = connection;
		}

		neuron["connections"] = connections;
		neurons[neuron_index] = neuron;
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
