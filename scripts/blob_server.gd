# blob_server.gd
# Versão corrigida e completa (Godot 4.x)
# - Coloque como Autoload singleton com o nome "BlobServer".
# - Use BlobServer.process_batch(batch) para processar pedidos no mesmo formato do servidor Python.
# - Retorna um Array de resultados; para gen_brain/offspring inclui "gens" e "gens_num" para compatibilidade.

extends Node
class_name BlobServer

# ===== CONSTANTES E VARIÁVEIS GLOBAIS =====
const NUM_SYMBOLS := 4
const NUM_DIGITS := 5
const NUM_DIGITS_2 := 2
const NUM_DIGITS_3 := 6
const NUM_DIGITS_4 := 4
const A_MIN := 4
const A_MAX := 7
const B_MIN := 4
const B_MAX := 7
const C_MIN := 4
const C_MAX := 7
const NUM_INPUTS := 16
const NUM_OUTPUTS := 4
const MAPPING := {0: 0.2, 1: 0.4, 2: 0.6, 3: 0.8}
const ATCG_MAPPING := {0: "A", 1: "T", 2: "C", 3: "G"}

var SHOW := false
var SHOW_ATCG := false
var NPC : Dictionary = {}  # Dicionário global de NPCs

# Variáveis para salvar atividades neuronais
var it := -1
var it_real := -1
var blob_history := []
var num_connections_history := []
var N := 100

# Random generator
var rng := RandomNumberGenerator.new()

func _ready() -> void:
	rng.randomize()
	# limpa arquivos históricos (comportamento semelhante ao Python original)
	var f := File.new()
	if f.open("user://blob_history.txt", File.WRITE) == OK:
		f.store_string("")
		f.close()
	if f.open("user://num_connections_history.txt", File.WRITE) == OK:
		f.store_string("")
		f.close()
	print("BlobServer pronto")

# ===== API pública: processa um lote (substitui o handler WebSocket) =====
# batch: Array de Dictionaries no formato do Python
func process_batch(batch: Array) -> Array:
	var results : Array = []
	for npc in batch:
		var npc_action : String = npc.get("npc_action", "")
		var npc_id : String = npc.get("id", "")
		var inputs : Array = npc.get("inputs", [])

		if npc_action == "gen_brain":
			var tuple := gen_brain()
			var gens := tuple[0]
			var gens_num := tuple[1]
			# garante id
			gens_num["id"] = npc_id
			# salva internamente
			NPC[npc_id] = gens_num
			NPC[npc_id]["gens"] = gens
			extras(npc_id)
			# Retorna também os genes (compatibilidade com cliente)
			results.append({
				"id": gens_num["id"],
				"npc_action": "gen_brain",
				"gens": gens,
				"gens_num": gens_num
			})
		elif npc_action == "process":
			# se NPC não existir, retorna outputs zeros (evita erros)
			if not NPC.has(npc_id):
				var zeros := []
				for i in range(NUM_OUTPUTS):
					zeros.append(0)
				results.append({
					"id": npc_id,
					"outputs": zeros,
					"size": A_MAX * B_MAX * C_MAX,
					"num_connections": 0,
					"npc_action": "process"
				})
			else:
				var outputs := process_npc(npc_id, inputs)
				results.append({
					"id": npc_id,
					"outputs": outputs,
					"size": A_MAX * B_MAX * C_MAX,
					"num_connections": NPC[npc_id]["num_connections"],
					"npc_action": "process"
				})
		elif npc_action == "offspring":
			var npc_id1 : String = npc.get("id1", "")
			var npc_id2 : String = npc.get("id2", "")
			var child := offspring(npc_id1, npc_id2)
			var gens = child[0]
			var gens_num = child[1]
			if gens == "":
				results.append({"id": "", "npc_action": "offspring"})
			else:
				var newid := "blob_" + str(int(NPC.size() + rng.randi_range(0, 999)))
				gens_num["id"] = newid
				NPC[newid] = gens_num
				NPC[newid]["gens"] = gens
				extras(newid)
				# inclui genes no resultado
				results.append({
					"id": newid,
					"npc_action": "offspring",
					"gens": gens,
					"gens_num": gens_num
				})
	# fim for
	return results

# ===== Funções auxiliares convertidas =====

func gen_brain() -> Array:
	var success := false
	var gens := {}
	var gens_num := {}

	while not success:
		gens.clear()
		gens_num.clear()

		# Dimensions: gens["dims"] é 3 x NUM_DIGITS
		gens["dims"] = []
		for i in range(3):
			var row := []
			for j in range(NUM_DIGITS):
				row.append(rng.randi_range(0, NUM_SYMBOLS - 1))
			gens["dims"].append(row)

		var digits = gens["dims"][0]
		var a = max(A_MIN, digits2int(digits) % (A_MAX + 1))
		digits = gens["dims"][1]
		var b = max(B_MIN, digits2int(digits) % (B_MAX + 1))
		digits = gens["dims"][2]
		var c = max(C_MIN, digits2int(digits) % (C_MAX + 1))

		gens_num["dims"] = [a, b, c]
		var num_neurons = a * b * c

		if SHOW:
			print("Dimensions = %d x %d x %d    Number of neurons = %d" % [a, b, c, num_neurons])

		# Connections with inputs
		gens["inputs"] = []
		for i in range(NUM_INPUTS):
			var row := []
			for j in range(3 * NUM_DIGITS):
				row.append(rng.randi_range(0, NUM_SYMBOLS - 1))
			gens["inputs"].append(row)
		gens_num["inputs"] = []

		for i in range(NUM_INPUTS):
			var neuron_x = digits2int(gens["inputs"][i].slice(0, NUM_DIGITS)) % a
			var neuron_y = digits2int(gens["inputs"][i].slice(NUM_DIGITS, 2 * NUM_DIGITS)) % b
			var neuron_z = digits2int(gens["inputs"][i].slice(2 * NUM_DIGITS, 3 * NUM_DIGITS)) % c
			gens_num["inputs"].append(str(neuron_x) + "-" + str(neuron_y) + "-" + str(neuron_z))

		# remove duplicatas mantendo compatibilidade
		gens_num["inputs"] = Array(gens_num["inputs"]).duplicate()
		# unique modifies in place on Godot arrays; ensure unique
		gens_num["inputs"].unique()
		var num_connections_input_neuron = gens_num["inputs"].size()
		if num_connections_input_neuron != NUM_INPUTS:
			if SHOW:
				print("É necessário repetir a operação pois existem inputs não conectados a neurônios.")
			continue

		if SHOW:
			print("Neurons connected to inputs: %s    Number of connections: %d" % [str(gens_num["inputs"]), num_connections_input_neuron])

		# Connections with outputs
		gens["outputs"] = []
		for i in range(NUM_OUTPUTS):
			var row_o := []
			for j in range(3 * NUM_DIGITS):
				row_o.append(rng.randi_range(0, NUM_SYMBOLS - 1))
			gens["outputs"].append(row_o)
		gens_num["outputs"] = []

		for i in range(NUM_OUTPUTS):
			var neuron_x = digits2int(gens["outputs"][i].slice(0, NUM_DIGITS)) % a
			var neuron_y = digits2int(gens["outputs"][i].slice(NUM_DIGITS, 2 * NUM_DIGITS)) % b
			var neuron_z = digits2int(gens["outputs"][i].slice(2 * NUM_DIGITS, 3 * NUM_DIGITS)) % c
			gens_num["outputs"].append(str(neuron_x) + "-" + str(neuron_y) + "-" + str(neuron_z))

		gens_num["outputs"] = Array(gens_num["outputs"]).duplicate()
		gens_num["outputs"].unique()
		var num_connections_output_neuron = gens_num["outputs"].size()
		if num_connections_output_neuron != NUM_OUTPUTS:
			if SHOW:
				print("É necessário repetir a operação pois existem outputs não conectados a neurônios.")
			continue

		success = true
		for x in gens_num["outputs"]:
			if x in gens_num["inputs"]:
				if SHOW:
					print("É necessário repetir a operação pois um neurônio conectado ao output já está sendo usado por um de input: %s" % x)
				success = false
				break
		if not success:
			continue

		if SHOW:
			print("Neurons connected to outputs: %s    Number of connections: %d" % [str(gens_num["outputs"]), num_connections_output_neuron])

		# Output thresholds
		gens["outputs_thr"] = []
		for i in range(len(gens["outputs"])):
			gens["outputs_thr"].append(rng.randi_range(0, NUM_SYMBOLS - 1))
		gens_num["outputs_thr"] = {}
		var w := 0
		for x in gens_num["outputs"]:
			gens_num["outputs_thr"][x] = MAPPING.get(gens["outputs_thr"][w], 0.2)
			w += 1

		# Region sizes
		gens["regions"] = []
		for i in range(num_neurons):
			var rowr := []
			for j in range(3 * NUM_DIGITS_2):
				rowr.append(rng.randi_range(0, NUM_SYMBOLS - 1))
			gens["regions"].append(rowr)
		gens_num["regions"] = {}
		w = 0
		for i in range(a):
			for j in range(b):
				for k in range(c):
					var region_size_x = digits2int(gens["regions"][w].slice(0, NUM_DIGITS_2)) % a
					var region_size_y = digits2int(gens["regions"][w].slice(NUM_DIGITS_2, 2*NUM_DIGITS_2)) % b
					var region_size_z = digits2int(gens["regions"][w].slice(2*NUM_DIGITS_2, 3*NUM_DIGITS_2)) % c
					gens_num["regions"][str(i)+"-"+str(j)+"-"+str(k)] = [region_size_x, region_size_y, region_size_z]
					w += 1

		# Initial connections
		gens["num_initial_connections"] = []
		for i in range(NUM_DIGITS_4):
			gens["num_initial_connections"].append(rng.randi_range(0, NUM_SYMBOLS - 1))
		var num_initial_connections = digits2int(gens["num_initial_connections"]) % max(1, (num_neurons - 1))
		gens_num["num_initial_connections"] = num_initial_connections
		gens["connections"] = {}
		gens_num["connections"] = {}

		for i in range(a):
			for j in range(b):
				for k in range(c):
					var key := str(i)+"-"+str(j)+"-"+str(k)
					gens["connections"][key] = []
					gens_num["connections"][key] = []
					# fill random candidate encoded coordinates
					for w_i in range(num_initial_connections):
						var cand := []
						for z in range(3 * NUM_DIGITS_2):
							cand.append(rng.randi_range(0, NUM_SYMBOLS - 1))
						gens["connections"][key].append(cand)
					# convert candidate -> actual connections constrained by region
					for w_i in range(num_initial_connections):
						var neuron_x = digits2int(gens["connections"][key][w_i].slice(0, NUM_DIGITS_2)) % a
						var neuron_y = digits2int(gens["connections"][key][w_i].slice(NUM_DIGITS_2, 2*NUM_DIGITS_2)) % b
						var neuron_z = digits2int(gens["connections"][key][w_i].slice(2*NUM_DIGITS_2, 3*NUM_DIGITS_2)) % c
						var cand_key = str(neuron_x)+"-"+str(neuron_y)+"-"+str(neuron_z)
						if cand_key in gens_num["connections"][key]:
							continue
						# Check reverse connection absence (approximation)
						if (cand_key in gens_num["connections"]) and (key in gens_num["connections"][cand_key]):
							continue
						var radius = gens_num["regions"][key]
						if i - radius[0] <= neuron_x <= i + radius[0] and j - radius[1] <= neuron_y <= j + radius[1] and k - radius[2] <= neuron_z <= k + radius[2]:
							gens_num["connections"][key].append(cand_key)

		# Probabilities of making connections
		gens["probs"] = []
		for i in range(num_neurons):
			gens["probs"].append(rng.randi_range(0, NUM_SYMBOLS - 1))
		gens_num["probs"] = {}
		w = 0
		for i in range(a):
			for j in range(b):
				for k in range(c):
					gens_num["probs"][str(i)+"-"+str(j)+"-"+str(k)] = MAPPING.get(gens["probs"][w], 0.2)
					w += 1

		# Threshold per neuron
		gens["thr"] = []
		for i in range(num_neurons):
			gens["thr"].append(rng.randi_range(0, NUM_SYMBOLS - 1))
		gens_num["thr"] = {}
		w = 0
		for i in range(a):
			for j in range(b):
				for k in range(c):
					gens_num["thr"][str(i)+"-"+str(j)+"-"+str(k)] = MAPPING.get(gens["thr"][w], 0.2)
					w += 1

		# Decay (concatena os dígitos como string e converte para float 0.xxx)
		var decay_str := StringArray(gens["decay"])  # StringArray devolve a concatenação
		gens["decay"] = []
		for i in range(NUM_DIGITS_3):
			gens["decay"].append(rng.randi_range(0, NUM_SYMBOLS - 1))
		decay_str = StringArray(gens["decay"])
		gens_num["decay"] = String("0." + decay_str).to_float()

		# inactive iterations
		gens["inactive_iter"] = []
		for i in range(NUM_DIGITS_2):
			gens["inactive_iter"].append(rng.randi_range(0, NUM_SYMBOLS - 1))
		gens_num["inactive_iter"] = digits2int(gens["inactive_iter"]) + 1

		# exp response (seguir comportamento do Python: //10 + 1)
		gens["exp_response"] = []
		for i in range(NUM_DIGITS_2):
			gens["exp_response"].append(rng.randi_range(0, NUM_SYMBOLS - 1))
		gens_num["exp_response"] = int(digits2int(gens["exp_response"]) / 10) + 1

		# weakened_iter
		gens["weakened_iter"] = []
		for i in range(NUM_DIGITS_2):
			gens["weakened_iter"].append(rng.randi_range(0, NUM_SYMBOLS - 1))
		gens_num["weakened_iter"] = gens_num["inactive_iter"] + digits2int(gens["weakened_iter"]) + 1

		if SHOW or SHOW_ATCG:
			print("Genes gerados")

		# marca sucesso para sair do loop
		# (já foi definido success = true mais acima)
		# garante que saímos do while
		# (o loop já foi preparado para validar condições acima)
	# fim while
	return [gens, gens_num]

func offspring(npc_id1: String, npc_id2: String) -> Array:
	var success := false
	var trial := 0
	if not NPC.has(npc_id1) or not NPC.has(npc_id2):
		return ["", ""]

	while not success:
		trial += 1
		if trial >= 3:
			return ["", ""]

		var gens := {}
		var gens_num := {}
		var a := rng.randi_range(min(NPC[npc_id1]["dims"][0], NPC[npc_id2]["dims"][0]), max(NPC[npc_id1]["dims"][0], NPC[npc_id2]["dims"][0]))
		var b := rng.randi_range(min(NPC[npc_id1]["dims"][1], NPC[npc_id2]["dims"][1]), max(NPC[npc_id1]["dims"][1], NPC[npc_id2]["dims"][1]))
		var c := rng.randi_range(min(NPC[npc_id1]["dims"][2], NPC[npc_id2]["dims"][2]), max(NPC[npc_id1]["dims"][2], NPC[npc_id2]["dims"][2]))
		gens["dims"] = [a, b, c]
		gens_num["dims"] = [a, b, c]
		var num_neurons := a * b * c

		# Inputs crossover
		var cut := rng.randi_range(int(0.4 * NUM_INPUTS), int(0.6 * NUM_INPUTS))
		gens["inputs"] = []
		for i in range(NUM_INPUTS):
			if i < cut:
				gens["inputs"].append(NPC[npc_id1]["gens"]["inputs"][i])
			else:
				gens["inputs"].append(NPC[npc_id2]["gens"]["inputs"][i])
		gens_num["inputs"] = []
		for i in range(NUM_INPUTS):
			var neuron_x = digits2int(gens["inputs"][i].slice(0, NUM_DIGITS)) % a
			var neuron_y = digits2int(gens["inputs"][i].slice(NUM_DIGITS, 2*NUM_DIGITS)) % b
			var neuron_z = digits2int(gens["inputs"][i].slice(2*NUM_DIGITS, 3*NUM_DIGITS)) % c
			gens_num["inputs"].append(str(neuron_x)+"-"+str(neuron_y)+"-"+str(neuron_z))
		gens_num["inputs"] = Array(gens_num["inputs"]).duplicate()
		gens_num["inputs"].unique()
		if gens_num["inputs"].size() != NUM_INPUTS:
			continue

		# Outputs crossover
		cut = rng.randi_range(int(0.4 * NUM_OUTPUTS), int(0.6 * NUM_OUTPUTS))
		gens["outputs"] = []
		for i in range(NUM_OUTPUTS):
			if i < cut:
				gens["outputs"].append(NPC[npc_id1]["gens"]["outputs"][i])
			else:
				gens["outputs"].append(NPC[npc_id2]["gens"]["outputs"][i])
		gens_num["outputs"] = []
		for i in range(NUM_OUTPUTS):
			var neuron_x = digits2int(gens["outputs"][i].slice(0, NUM_DIGITS)) % a
			var neuron_y = digits2int(gens["outputs"][i].slice(NUM_DIGITS, 2*NUM_DIGITS)) % b
			var neuron_z = digits2int(gens["outputs"][i].slice(2*NUM_DIGITS, 3*NUM_DIGITS)) % c
			gens_num["outputs"].append(str(neuron_x)+"-"+str(neuron_y)+"-"+str(neuron_z))
		gens_num["outputs"] = Array(gens_num["outputs"]).duplicate()
		gens_num["outputs"].unique()
		if gens_num["outputs"].size() != NUM_OUTPUTS:
			continue

		success = true
		for x in gens_num["outputs"]:
			if x in gens_num["inputs"]:
				success = false
				break
		if not success:
			continue

		# outputs_thr crossover
		cut = rng.randi_range(int(0.4 * len(gens["outputs"])), int(0.6 * len(gens["outputs"])))
		gens["outputs_thr"] = []
		for i in range(len(gens["outputs"])):
			if i < cut:
				gens["outputs_thr"].append(NPC[npc_id1]["gens"]["outputs_thr"][i])
			else:
				gens["outputs_thr"].append(NPC[npc_id2]["gens"]["outputs_thr"][i])
		gens_num["outputs_thr"] = {}
		var w := 0
		for x in gens_num["outputs"]:
			gens_num["outputs_thr"][x] = MAPPING.get(gens["outputs_thr"][w], 0.2)
			w += 1

		# regions crossover (simplified)
		gens["regions"] = []
		for i in range(num_neurons):
			if i < NPC[npc_id1]["gens"]["regions"].size():
				gens["regions"].append(NPC[npc_id1]["gens"]["regions"][i])
			elif i < NPC[npc_id2]["gens"]["regions"].size():
				gens["regions"].append(NPC[npc_id2]["gens"]["regions"][i])
			else:
				var row := []
				for j in range(3 * NUM_DIGITS_2):
					row.append(rng.randi_range(0, NUM_SYMBOLS - 1))
				gens["regions"].append(row)
		gens_num["regions"] = {}
		w = 0
		for i in range(a):
			for j in range(b):
				for k in range(c):
					var region_size_x = digits2int(gens["regions"][w].slice(0, NUM_DIGITS_2)) % a
					var region_size_y = digits2int(gens["regions"][w].slice(NUM_DIGITS_2, 2*NUM_DIGITS_2)) % b
					var region_size_z = digits2int(gens["regions"][w].slice(2*NUM_DIGITS_2, 3*NUM_DIGITS_2)) % c
					gens_num["regions"][str(i)+"-"+str(j)+"-"+str(k)] = [region_size_x, region_size_y, region_size_z]
					w += 1

		# num_initial_connections crossover
		if rng.randf() > 0.5:
			gens["num_initial_connections"] = NPC[npc_id1]["gens"]["num_initial_connections"]
		else:
			gens["num_initial_connections"] = NPC[npc_id2]["gens"]["num_initial_connections"]
		var num_initial_connections = digits2int(gens["num_initial_connections"]) % max(1, (num_neurons - 1))
		gens_num["num_initial_connections"] = num_initial_connections

		# connections build
		gens["connections"] = {}
		gens_num["connections"] = {}
		for i in range(a):
			for j in range(b):
				for k in range(c):
					var key = str(i)+"-"+str(j)+"-"+str(k)
					gens["connections"][key] = []
					gens_num["connections"][key] = []
					var cut_conn = rng.randi_range(int(0.4 * num_initial_connections), int(0.6 * num_initial_connections))
					for u in range(num_initial_connections):
						var candidate := []
						if u < cut_conn:
							if NPC[npc_id1]["gens"]["connections"].has(key) and u < NPC[npc_id1]["gens"]["connections"][key].size():
								candidate = NPC[npc_id1]["gens"]["connections"][key][u]
							elif NPC[npc_id2]["gens"]["connections"].has(key) and u < NPC[npc_id2]["gens"]["connections"][key].size():
								candidate = NPC[npc_id2]["gens"]["connections"][key][u]
							else:
								for z in range(3 * NUM_DIGITS_2):
									candidate.append(rng.randi_range(0, NUM_SYMBOLS - 1))
						else:
							if NPC[npc_id2]["gens"]["connections"].has(key) and u < NPC[npc_id2]["gens"]["connections"][key].size():
								candidate = NPC[npc_id2]["gens"]["connections"][key][u]
							elif NPC[npc_id1]["gens"]["connections"].has(key) and u < NPC[npc_id1]["gens"]["connections"][key].size():
								candidate = NPC[npc_id1]["gens"]["connections"][key][u]
							else:
								for z in range(3 * NUM_DIGITS_2):
									candidate.append(rng.randi_range(0, NUM_SYMBOLS - 1))
						gens["connections"][key].append(candidate)
					# convert to actual connections
					for w_i in range(num_initial_connections):
						var neuron_x = digits2int(gens["connections"][key][w_i].slice(0, NUM_DIGITS_2)) % a
						var neuron_y = digits2int(gens["connections"][key][w_i].slice(NUM_DIGITS_2, 2*NUM_DIGITS_2)) % b
						var neuron_z = digits2int(gens["connections"][key][w_i].slice(2*NUM_DIGITS_2, 3*NUM_DIGITS_2)) % c
						var candk = str(neuron_x)+"-"+str(neuron_y)+"-"+str(neuron_z)
						if candk in gens_num["connections"][key]:
							continue
						var radius = gens_num["regions"][key]
						if i - radius[0] <= neuron_x <= i + radius[0] and j - radius[1] <= neuron_y <= j + radius[1] and k - radius[2] <= neuron_z <= k + radius[2]:
							gens_num["connections"][key].append(candk)

		# probs crossover
		var cutp = rng.randi_range(int(0.4 * num_neurons), int(0.6 * num_neurons))
		gens["probs"] = []
		for i in range(num_neurons):
			if i < cutp and i < NPC[npc_id1]["gens"]["probs"].size():
				gens["probs"].append(NPC[npc_id1]["gens"]["probs"][i])
			elif i < NPC[npc_id2]["gens"]["probs"].size():
				gens["probs"].append(NPC[npc_id2]["gens"]["probs"][i])
			else:
				gens["probs"].append(rng.randi_range(0, NUM_SYMBOLS - 1))
		gens_num["probs"] = {}
		w = 0
		for i in range(a):
			for j in range(b):
				for k in range(c):
					gens_num["probs"][str(i)+"-"+str(j)+"-"+str(k)] = MAPPING.get(gens["probs"][w], 0.2)
					w += 1

		# thr crossover
		cutp = rng.randi_range(int(0.4 * num_neurons), int(0.6 * num_neurons))
		gens["thr"] = []
		for i in range(num_neurons):
			if i < cutp and i < NPC[npc_id1]["gens"]["thr"].size():
				gens["thr"].append(NPC[npc_id1]["gens"]["thr"][i])
			elif i < NPC[npc_id2]["gens"]["thr"].size():
				gens["thr"].append(NPC[npc_id2]["gens"]["thr"][i])
			else:
				gens["thr"].append(rng.randi_range(0, NUM_SYMBOLS - 1))
		gens_num["thr"] = {}
		w = 0
		for i in range(a):
			for j in range(b):
				for k in range(c):
					gens_num["thr"][str(i)+"-"+str(j)+"-"+str(k)] = MAPPING.get(gens["thr"][w], 0.2)
					w += 1

		# decay crossover (usa string concatenada)
		if rng.randf() > 0.5:
			gens["decay"] = NPC[npc_id1]["gens"]["decay"]
		else:
			gens["decay"] = NPC[npc_id2]["gens"]["decay"]
		var decay_s = StringArray(gens["decay"])
		gens_num["decay"] = String("0." + decay_s).to_float()

		# inactive_iter, exp_response, weakened_iter
		gens["inactive_iter"] = []
		for i in range(NUM_DIGITS_2):
			gens["inactive_iter"].append(rng.randi_range(0, NUM_SYMBOLS - 1))
		gens_num["inactive_iter"] = digits2int(gens["inactive_iter"]) + 1

		if rng.randf() > 0.5:
			gens["exp_response"] = NPC[npc_id1]["gens"]["exp_response"]
		else:
			gens["exp_response"] = NPC[npc_id2]["gens"]["exp_response"]
		# garantir inteiro >=1 similar ao Python original
		gens_num["exp_response"] = int(digits2int(gens["exp_response"]) / 10) + 1
		if gens_num["exp_response"] < 1:
			gens_num["exp_response"] = 1

		if rng.randf() > 0.5:
			gens["weakened_iter"] = NPC[npc_id1]["gens"]["weakened_iter"]
		else:
			gens["weakened_iter"] = NPC[npc_id2]["gens"]["weakened_iter"]
		gens_num["weakened_iter"] = gens_num["inactive_iter"] + digits2int(gens["weakened_iter"]) + 1

		success = true

	return [gens, gens_num]

func extras(npc_id: String) -> void:
	NPC[npc_id]["neuron_signals"] = {}
	NPC[npc_id]["connection_strengths"] = {}
	NPC[npc_id]["inactive_counter"] = {}
	for neuron in NPC[npc_id]["connections"]:
		NPC[npc_id]["inactive_counter"][neuron] = NPC[npc_id]["inactive_iter"]
	var total := 0
	for neuron in NPC[npc_id]["connections"]:
		NPC[npc_id]["connection_strengths"][neuron] = {}
		for neuron2 in NPC[npc_id]["connections"][neuron]:
			NPC[npc_id]["connection_strengths"][neuron][neuron2] = 1
			total += 1
	NPC[npc_id]["num_connections"] = total
	NPC[npc_id]["num_childs"] = 0

func digits2int(digits: Array) -> int:
	var s := ""
	for x in digits:
		s += str(x)
	# se string vazia, retorna 0
	if s == "":
		return 0
	return int(s)

func exp_response(x: float, k: float) -> float:
	# mesma fórmula do Python
	return 1.0 - exp(-x / float(k))

# ===== Processamento =====
func process_npc(npc_id: String, inputs: Array) -> Array:
	if not NPC.has(npc_id):
		var zeros := []
		for i in range(NUM_OUTPUTS):
			zeros.append(0)
		return zeros

	var neuron_signals = NPC[npc_id]["neuron_signals"]
	var neuron_thrs = NPC[npc_id]["thr"]
	var neuron_connections = NPC[npc_id]["connections"]
	var e_orig = NPC[npc_id]["decay"]
	var e_now = NPC[npc_id]["connection_strengths"]
	var exp_factor = NPC[npc_id]["exp_response"]

	var neuron_signals_tmp := {}
	var to_zero := []

	if npc_id == "blob_0":
		var pair = process_npc_neurons_blob_0(npc_id, neuron_signals, neuron_thrs, neuron_connections, e_orig, e_now, exp_factor)
		neuron_signals_tmp = pair[0]
		to_zero = pair[1]
	else:
		var pair2 = process_npc_neurons(npc_id, neuron_signals, neuron_thrs, neuron_connections, e_orig, e_now, exp_factor)
		neuron_signals_tmp = pair2[0]
		to_zero = pair2[1]

	process_npc_neurons_updates(npc_id, neuron_signals, neuron_signals_tmp, to_zero)
	process_npc_inputs(npc_id, inputs)
	var outputs = process_npc_outputs(npc_id)
	return outputs

func process_npc_neurons_blob_0(npc_id, neuron_signals, neuron_thrs, neuron_connections, e_orig, e_now, exp_factor):
	it_real += 1
	if SHOW:
		print("Processando sinais - iteração %d" % it_real)
	if it_real % N == 0 or it_real < N:
		it += 1

	var neuron_signals_tmp := {}
	var to_zero := []

	for neuron in neuron_signals.keys():
		var signal = neuron_signals[neuron]
		var thr = neuron_thrs[neuron]
		blob_history.append(str(it) + " " + neuron + " " + str(signal) + "\n")

		if signal > thr:
			if NPC[npc_id]["inactive_counter"][neuron] == NPC[npc_id]["inactive_iter"]:
				NPC[npc_id]["inactive_counter"][neuron] -= 1
				signal = exp_response(signal, exp_factor)
				to_zero.append(neuron)
				for neuron2 in neuron_connections[neuron]:
					var e = (1 + e_now[neuron][neuron2]) / 2.0
					update_connections_strengthening(npc_id, e, neuron, neuron2)
					if neuron_signals_tmp.has(neuron2):
						neuron_signals_tmp[neuron2] += e * signal
					else:
						neuron_signals_tmp[neuron2] = e * signal
				create_new_connections(npc_id, neuron)

		if NPC[npc_id]["inactive_counter"][neuron] != NPC[npc_id]["inactive_iter"]:
			NPC[npc_id]["inactive_counter"][neuron] -= 1
			if NPC[npc_id]["inactive_counter"][neuron] < 0:
				NPC[npc_id]["inactive_counter"][neuron] = NPC[npc_id]["inactive_iter"]

		update_connections_weakening(npc_id, e_orig, e_now, neuron)

	return [neuron_signals_tmp, to_zero]

func process_npc_neurons(npc_id, neuron_signals, neuron_thrs, neuron_connections, e_orig, e_now, exp_factor):
	var neuron_signals_tmp := {}
	var to_zero := []

	for neuron in neuron_signals.keys():
		var signal = neuron_signals[neuron]
		var thr = neuron_thrs[neuron]
		if signal > thr:
			if NPC[npc_id]["inactive_counter"][neuron] == NPC[npc_id]["inactive_iter"]:
				NPC[npc_id]["inactive_counter"][neuron] -= 1
				signal = exp_response(signal, exp_factor)
				to_zero.append(neuron)
				for neuron2 in neuron_connections[neuron]:
					var e = (1 + e_now[neuron][neuron2]) / 2.0
					update_connections_strengthening(npc_id, e, neuron, neuron2)
					if neuron_signals_tmp.has(neuron2):
						neuron_signals_tmp[neuron2] += e * signal
					else:
						neuron_signals_tmp[neuron2] = e * signal
				create_new_connections(npc_id, neuron)

		if NPC[npc_id]["inactive_counter"][neuron] != NPC[npc_id]["inactive_iter"]:
			NPC[npc_id]["inactive_counter"][neuron] -= 1
			if NPC[npc_id]["inactive_counter"][neuron] < 0:
				NPC[npc_id]["inactive_counter"][neuron] = NPC[npc_id]["inactive_iter"]

		update_connections_weakening(npc_id, e_orig, e_now, neuron)

	return [neuron_signals_tmp, to_zero]

func create_new_connections(npc_id: String, neuron: String) -> void:
	var dims = NPC[npc_id]["dims"]
	var a := int(dims[0])
	var b := int(dims[1])
	var c := int(dims[2])
	var radius := NPC[npc_id]["regions"][neuron]
	var parts := neuron.split("-")
	var neuron_x := int(parts[0])
	var neuron_y := int(parts[1])
	var neuron_z := int(parts[2])

	for i in range(max(0, neuron_x - radius[0]), min(a, neuron_x + radius[0] + 1)):
		for j in range(max(0, neuron_y - radius[1]), min(b, neuron_y + radius[1] + 1)):
			for k in range(max(0, neuron_z - radius[2]), min(c, neuron_z + radius[2] + 1)):
				if NPC[npc_id]["probs"].get(neuron, 0.0) < rng.randf():
					continue
				var neuron2 := str(i)+"-"+str(j)+"-"+str(k)
				if not NPC[npc_id]["neuron_signals"].has(neuron2):
					continue
				if NPC[npc_id]["neuron_signals"][neuron2] <= 0:
					continue
				if not (neuron2 in NPC[npc_id]["connections"][neuron]) and not (neuron in NPC[npc_id]["connections"][neuron2]):
					NPC[npc_id]["connections"][neuron].append(neuron2)
					NPC[npc_id]["connection_strengths"][neuron][neuron2] = 1
					if npc_id == "blob_0":
						print("%s: Criou nova conexão!!! %s ===> %s" % [npc_id, neuron, neuron2])
					return
	return

func update_connections_weakening(npc_id, e_orig, e_now, neuron) -> void:
	var copy_keys = []
	for k in e_now[neuron].keys():
		copy_keys.append(k)
	for neuron2 in copy_keys:
		# mantém condições idênticas ao Python original (checa se outputs/inputs)
		if not (neuron2 in NPC[npc_id]["outputs"]) or not (neuron2 in NPC[npc_id]["inputs"]):
			continue
		var e = e_now[neuron][neuron2] - pow(e_orig, 2)
		if e < 0:
			NPC[npc_id]["connections"][neuron].erase(neuron2)
			e_now[neuron].erase(neuron2)
		else:
			e_now[neuron][neuron2] = e
	return

func update_connections_strengthening(npc_id, e, neuron, neuron2) -> void:
	NPC[npc_id]["connection_strengths"][neuron][neuron2] = e
	return

func process_npc_neurons_updates(npc_id, neuron_signals, neuron_signals_tmp, to_zero) -> void:
	for neuron in to_zero:
		NPC[npc_id]["neuron_signals"][neuron] = 0
	for neuron in neuron_signals_tmp.keys():
		NPC[npc_id]["neuron_signals"][neuron] = neuron_signals_tmp[neuron]
	var total := 0
	for neuron in NPC[npc_id]["connections"]:
		total += NPC[npc_id]["connections"][neuron].size()
	NPC[npc_id]["num_connections"] = total
	if it_real % N == 0 or it_real < N:
		num_connections_history.append(str(npc_id) + " " + str(it) + " " + str(NPC[npc_id]["num_connections"]) + "\n")
	return

func process_npc_inputs(npc_id: String, inputs: Array) -> void:
	var exp_factor = NPC[npc_id]["exp_response"]
	var idx := 0
	for neuron in NPC[npc_id]["inputs"]:
		var signal = inputs[idx] if idx < inputs.size() else 0
		NPC[npc_id]["neuron_signals"][neuron] = exp_response(signal, exp_factor)
		idx += 1
	return

func process_npc_outputs(npc_id: String) -> Array:
	var outputs := []
	var exp_factor = NPC[npc_id]["exp_response"]
	for i in range(NPC[npc_id]["outputs"].size()):
		var neuron = NPC[npc_id]["outputs"][i]
		if NPC[npc_id]["neuron_signals"].has(neuron):
			var signal = NPC[npc_id]["neuron_signals"][neuron]
			var thr = NPC[npc_id]["outputs_thr"][neuron]
			outputs.append(1 if exp_response(signal, exp_factor) > thr else 0)
		else:
			outputs.append(0)
	return outputs

# salva históricos em disco (chame no _exit_tree do seu nó principal)
func save_histories() -> void:
	var f := File.new()
	if f.open("user://blob_history.txt", File.WRITE) == OK:
		for line in blob_history:
			f.store_string(line)
		f.close()
	if f.open("user://num_connections_history.txt", File.WRITE) == OK:
		for line in num_connections_history:
			f.store_string(line)
		f.close()
	print("BlobServer: históricos salvos")

# Helper: concatena array de inteiros em string (ex: [1,2,3] -> "123")
func StringArray(arr: Array) -> String:
	var s := ""
	for v in arr:
		s += str(v)
	return s
