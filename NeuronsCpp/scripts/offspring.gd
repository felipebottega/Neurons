class_name Offspring
extends RefCounted


const BLOCK_SIZE: int = 4
const DEFAULT_MUTATION_PROBABILITY: float = 0.001

const GENE_NAMES = [
	&"max_connections",
	&"beta",
	&"connections",
	&"activation_threshold",
	&"refractory_strength",
	&"exponential_factor",
	&"decay_factor",
	&"retention_factor",
	&"polarity_factor",
	&"hebbian_plasticity_rate",
	&"modulatory_release_factor",
	&"modulatory_sensitivity",
	&"modulatory_dynamics",
	&"structural_plasticity",
	&"input_influence"
]


## Creates a child genome from two parent genomes.
## [param parent_a] is the first parent genome.
## [param parent_b] is the second parent genome.
## [param mutation_probability] defines the mutation probability of each digit.
## Returns the generated child genome.
func generate(parent_a: Genome, parent_b: Genome, mutation_probability: float = DEFAULT_MUTATION_PROBABILITY) -> Genome:
	assert(parent_a != null and parent_b != null, "Both parent genomes are required.")
	assert(mutation_probability >= 0.0 and mutation_probability <= 1.0, "Mutation probability must be between 0 and 1.")

	var child = Genome.new()

	for gene_name in GENE_NAMES:
		var gene_a: Array[int] = parent_a.get(gene_name)
		var gene_b: Array[int] = parent_b.get(gene_name)
		child.set(gene_name, _crossover_gene(gene_a, gene_b))

	_mutate_genome(child, mutation_probability)

	return child


## Crosses two homologous genes in blocks of four digits.
## [param gene_a] is the gene inherited from the first parent.
## [param gene_b] is the homologous gene inherited from the second parent.
## Returns the crossed gene.
func _crossover_gene(gene_a: Array[int], gene_b: Array[int]) -> Array[int]:
	assert(gene_a.size() == gene_b.size(), "Homologous genes must have the same size.")
	assert(gene_a.size() % BLOCK_SIZE == 0, "Gene size must be a multiple of four.")

	var child_gene: Array[int] = []
	child_gene.resize(gene_a.size())

	for block_start in range(0, gene_a.size(), BLOCK_SIZE):
		var source_gene: Array[int] = gene_a if randi_range(0, 1) == 0 else gene_b

		for index in range(block_start, block_start + BLOCK_SIZE):
			child_gene[index] = source_gene[index]

	return child_gene


## Mutates every digit of the genome independently.
## [param genome] is the genome to mutate.
## [param mutation_probability] defines the mutation probability of each digit.
## Returns nothing.
func _mutate_genome(genome: Genome, mutation_probability: float) -> void:
	for gene_name in GENE_NAMES:
		var gene: Array[int] = genome.get(gene_name)

		for index in gene.size():
			if randf() < mutation_probability:
				gene[index] = randi_range(0, 3)
