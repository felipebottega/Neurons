extends Node2D

var food_scene: PackedScene = preload("res://scenes/food.tscn")
var num_food = 100

# Called when the node enters the scene tree for the first time.
func _ready() -> void:
	# Aloca comidas pelo ambiente.
	for i in range(num_food):
		new_food()
		
func _process(_delta: float) -> void:
	if len(Global.blobs_alive) == 0 and Global.process_frame_iterator > 1:
		print('Contagem de blobs está zerada.')
		get_tree().quit()
				
	# Atualiza a listagem de blobs vivos.
	Global.blobs_alive = []
	
	for child in get_children():
		if "blob_" in child.name:
			Global.blobs_alive.append(str(child.name))
		# Bug que às vezes acontece de gerar mais blobs e não nomear.
		if "Rigid" in child.name:
			child.queue_free()
	
	# Verifica se algum blob deve ser removido da cena.
	for blob_id in Global.blobs_alive:
		var blob = get_node_or_null(blob_id)
		
		if blob:
			if blob.ENERGY < 0:
				Global.blobs_alive.erase(blob_id)
				blob.queue_free()
				
	# Comida nova aparece no ambiente.
	if Global.process_frame_iterator % 100 == 0:
		new_food()
		num_food  += 1
				
func new_food():
	var screen_size = get_viewport_rect().size    # pega tamanho da viewport
	var margin = 0.13    # margem da tela para colocar comida
	var food = food_scene.instantiate()
	food.position = Vector2(randf_range(screen_size.x * margin, screen_size.x * (1 - margin)),
							randf_range(screen_size.y * margin, screen_size.y * (1 - margin)))
	add_child(food)
	food.connect("entered_food_area", Callable(self, "_entered_food_area"))
	food.connect("exited_food_area", Callable(self, "_exited_food_area"))
	food.connect("eat", Callable(self, "_on_food_eat"))

func _entered_food_area(blob, distance) -> void:
	blob.CLOSE_TO_FOOD += 1
	blob.INVERSE_DISTANCE_TO_FOOD = min(10, 100/distance)

func _exited_food_area(blob) -> void:
	blob.CLOSE_TO_FOOD -= 1 
	blob.INVERSE_DISTANCE_TO_FOOD = 0

func _on_food_eat(blob) -> void:
	blob.ENERGY = min(2.0, blob.ENERGY + 0.5)
