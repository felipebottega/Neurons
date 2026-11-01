# NeuronsCPU — implementação GDScript

Esta é a versão CPU em GDScript, com layouts importados do NeuronsGPU
(arquivo enviado como referência em 30/10/2026).

## Cenas

- `scenes/brain_explorer.tscn`
- `scenes/brain_offspring.tscn`
- `scenes/brain_diff.tscn`
- `scenes/brain_viewer.tscn`
- `scenes/brain_failures.tscn`
- `scenes/simulations.tscn`
- `scenes/agent_brain_preview.tscn`
- `scenes/validation.tscn`

## Backend

Os scripts `brain_builder_native.gd`, `brain_dynamics_native.gd`,
`genome_native.gd` e `offspring_native.gd` têm nomes de interfaces para
compatibilidade com as cenas modernas, mas são **100% GDScript**.
Não exigem GDExtension nem DLL.

O construtor usa densidade `eta`, escala fixa do campo espacial,
conectividade inicial em duas passagens, seleção de candidatos com
H e resgate axial com H². A plasticidade estrutural busca o cérebro inteiro.
Os resultados numéricos não são bit a bit equivalentes à GPU (precisão
numérica/ordem de operações e geradores aleatórios são diferentes).

A cena `validation` usa `validation_genome.json` e produz
`validation_cpu.jsonl` em res:// ou user://.

**Validação realizada:** auditoria estática de cenas, scripts, UIDs,
referências e estrutura de arquivos. Esta versão **não foi executada
no Godot 4.7.2 neste ambiente**, de modo que a validação de execução,
confronto numérico e testes interativos ainda são necessários.

Para abrir: extraia o ZIP e abra `project.godot` no Godot 4.7.2.

## Correção de compatibilidade GDScript

- Indentação de `scripts/brain_dynamics.gd` padronizada com tabulações;
- Tipos explícitos em `scripts/brain_dynamics_native.gd` para evitar inferência de `Variant` quando os avisos são tratados como erros.

