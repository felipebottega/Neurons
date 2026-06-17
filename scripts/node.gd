extends Node

var blob_scene: PackedScene = preload("res://scenes/blob.tscn")
@export var websocket_url : String = "ws://127.0.0.1:8000/ws"  # URL do servidor WebSocket
var socket : WebSocketPeer  # Instância do cliente WebSocket
var connected : bool = false  # Flag de conexão
var waiting_server = true
var num_blobs = 0

func _ready():
	socket = WebSocketPeer.new()  # cria a instância
	connect_to_server()  # conecta ao servidor
	
	# Gera todos os blobs uma vez só. Espera um pequeno tempo para garantir conexão com o servidor.
	await get_tree().create_timer(2).timeout   
	new_blobs(0)  

func connect_to_server():
	"""
	Conexão WebSocket.
	"""
	
	var err = socket.connect_to_url(websocket_url)
	if err != OK:
		push_error("❌ Falha ao conectar ao servidor WebSocket!")
		return
	print("Tentando conectar ao servidor WebSocket...")

func _process(_delta):
	socket.poll()  # Processa eventos pendentes

	match socket.get_ready_state():
		# Ações enquanto a conexão está aberta.
		WebSocketPeer.STATE_OPEN:
			if not connected:
				connected = true
				print("✅ Conectado ao servidor WebSocket!")

			# Recebe mensagens do servidor e converte em dicionário.
			while socket.get_available_packet_count() > 0:
				var msg = socket.get_packet().get_string_from_utf8()
				handle_received_data(msg)
				
			# Gera mais blobs quando a população estiver abaixo da metade inicial.
			if not waiting_server:
				if len(Global.blobs_alive) < Global.max_blobs/2 and Global.frame_count > 1000:
					new_blobs(Global.frame_count + 20000)
				
			# Processa as computações neuronais frame a frame para todos os blobs.
			if not waiting_server:
				process_frame()

		# Desconexão.
		WebSocketPeer.STATE_CLOSING, WebSocketPeer.STATE_CLOSED:
			if connected:
				handle_disconnection()
				
	Global.frame_count += 1
	
	# O blob_0 é imortal para garantir que os arquivos de histórico serão salvos corretamente (gambiarra).
	var blob = get_node_or_null("../blob_0")
	if blob:
		blob.ENERGY = 1.0

func handle_received_data(msg: String):
	"""
	Processa dados recebidos do servidor.
	"""
	
	var data = JSON.parse_string(msg)
	
	if typeof(data) == TYPE_DICTIONARY and data.has("results"):
		for result in data["results"]:
			if result["npc_action"] == "gen_brain":
				spawn_blob(result, false)
			elif result["npc_action"] == "offspring":
				if result["id"] != "":
					spawn_blob(result, true)
			else:
				actions(result)

	waiting_server = false

func new_blobs(id_increment: int):
	"""
	Envia requisição para criar os blobs.
	"""
	
	if not connected:
		return

	# O batch contém os pedidos que serão enviados ao servidor.
	var batch = []
	
	# Envia um pedido para gerar vários blobs.
	for i in range(Global.max_blobs):
		var blob_id = str("blob_%s" % [i + id_increment])
		batch.append({"id": blob_id, "npc_action": "gen_brain"})
		
	send_to_server(batch)
	
func spawn_blob(result: Dictionary, offspring: bool):
	"""
	Cria um blob.
	"""
	
	var blob = blob_scene.instantiate()
	blob.GENS = result    # atribui o JSON do servidor ao atributo 'GENS' do blob
	var blob_id = blob.GENS["id"]
	blob.name = blob_id
	Global.blobs_alive.append(blob_id)
	blob.connect("offspring", Callable(self, "_on_body_entered"))
	
	if offspring:    # energia do blob que acabou de nascer
		blob.ENERGY = 0.5
	
	num_blobs += 1
	get_tree().get_current_scene().add_child(blob)
	print("🟢 Blob criado com sucesso: ", blob.GENS["id"])
	
func process_frame():
	"""
	Envia os inputs dos blobs para processar os sinais deste frame.
	"""
	
	if not connected:
		return

	# A variável batch contém os pedidos que serão enviados ao servidor.
	var batch = []
	
	for blob_id in Global.blobs_alive:
		var blob = get_node_or_null("../" + blob_id)
	
		if blob:
			batch.append({"id": blob_id, "inputs": blob.get_inputs(), "npc_action": "process"})
			blob.LIFE_STEPS += 1
	
	# Atualiza informações no heads-up display.
	update_hud()
		
	# Envia pedido para o servidor.
	send_to_server(batch)
	
	# Atualiza frame.
	Global.process_frame_iterator += 1
	
func update_hud():
	var blob_id = Global.blobs_alive.max()
	if blob_id:
		var blob = get_node_or_null("../" + blob_id)
		if blob:
			var marker = get_node_or_null("../" + blob_id + "/Sprite2D")
			marker.visible = true
			
			for blob_id2 in Global.blobs_alive:
				if blob_id2 != blob_id:
					var blob2 = get_node_or_null("../" + blob_id2)
					if blob2:
						var marker2 = get_node_or_null("../" + blob_id2 + "/Sprite2D")
						marker2.visible = false
			
			$"../HUD/Iteration".text = "Iteration = " + str(Global.process_frame_iterator)
			$"../HUD/Population".text = "Population = " + str(len(Global.blobs_alive))
			$"../HUD/NumBlobs".text = "Num Blobs history = " + str(num_blobs)
			var formatted_inputs = blob.get_inputs().map(func(x): return "%0.3f" % x)
			var text1 = """BlobID {0}
			
							 INPUTS
							 Vel x+      {1}
							 Vel x-      {2}
							 Vel y+      {3}
							 Vel y-      {4}
							 AngVel+     {5}
							 AngVel-     {6}
							 Rot         {7}
							 Energy      {8}
							 InvEnergy   {9}
							 FoodClose   {10}
							 Collision   {11}
							 InvFoodDist {12}
						
							 OUTPUTS
							 Right       {13}
							 Left        {14}
							 Rot+        {15}
							 Rot-        {16}""".format([blob_id] + formatted_inputs)
			$"../HUD/BlobInfo".text = text1
		
func actions(result: Dictionary):
	"""
	Aplica as ações dos blobs a partir do resultado emitido pelo servidor.
	"""
	
	if result:
		var blob = get_node_or_null("../%s" % [result["id"]])
		
		if blob:
			if result["id"] == "blob_0":
				blob.modulate = Color(0.1, 0.2, 0.2, 1)
			else:
				var new_color = min(1, result["num_connections"]/(10 * result["size"]))
				blob.modulate = Color(1, new_color, new_color, 1)
				
			blob.get_outputs(result["outputs"])
			
			if len(result["outputs"]) != len(blob.OUTPUTS):
				print('Número de outputs do server e engine estão diferentes.')
				get_tree().quit()
			
func _on_body_entered(name1, name2):
	"""
	A colisão de 2 blobs faz com um novo blob seja criado, o filho deles.
	"""
	
	var batch = [{"id1": name1, "id2": name2, "npc_action": "offspring"}]
	send_to_server(batch)
	
func send_to_server(batch):
	waiting_server = true
	var payload = {"batch": batch}
	socket.send_text(JSON.stringify(payload))
	
func handle_disconnection():
	"""
	Trata desconexão.
	"""
	
	print("⚠️ Desconectado do servidor WebSocket")
	connected = false
