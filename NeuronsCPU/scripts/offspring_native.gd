class_name OffspringNative
extends RefCounted

const BLOCK_SIZE := 4
const DEFAULT_MUTATION_PROBABILITY := 0.001

func genome_from_dictionary(values: Dictionary) -> GenomeNative:
    var result := GenomeNative.new()
    for key in GenomeNative.GENES:
        if not values.has(key):
            return null
        var value: Array = values[key]
        if value.size() != int(GenomeNative.GENES[key]):
            return null
        var digits: Array[int] = []
        for raw in value:
            if int(raw) < 0 or int(raw) > 3:
                return null
            digits.append(int(raw))
        result.set(key, digits)
    return result

func generate(first: GenomeNative, second: GenomeNative, mutation_probability: float = DEFAULT_MUTATION_PROBABILITY) -> GenomeNative:
    if first == null or second == null or mutation_probability < 0.0 or mutation_probability > 1.0:
        return null
    var child := GenomeNative.new()
    for name in GenomeNative.GENES:
        var one: Array = first.get(name)
        var two: Array = second.get(name)
        var gene: Array[int] = []
        for index in one.size():
            gene.append(0)
        for block in range(0, gene.size(), BLOCK_SIZE):
            var source: Array = one if randi_range(0, 1) == 0 else two
            for index in range(block, mini(block + BLOCK_SIZE, gene.size())):
                gene[index] = int(source[index])
        for index in gene.size():
            if randf() < mutation_probability:
                gene[index] = randi_range(0, 3)
        child.set(name, gene)
    return child
