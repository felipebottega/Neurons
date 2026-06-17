extends RigidBody2D


var GENS : Dictionary
var ENERGY = 1.0
var SPEED = 100
var ROTATION = PI/10
var CLOSE_TO_FOOD = 0
var INVERSE_DISTANCE_TO_FOOD = 0
var NUM_CHILD = 0
var MAX_NUM_CHILD = 4
var COOL_DOWN_CHILD = 50
var LAST_CHILD_STEP = 0
var LIFE_STEPS = 0
var COLLISION = 0
var OUTPUTS = [0, 0, 0, 0]

signal offspring


func _ready() -> void:
	randomize()    # garante que a posição aleatória seja diferente a cada execução
	var screen_size = get_viewport_rect().size    # pega tamanho da viewport
	global_position = Vector2(randf() * screen_size.x, randf() * screen_size.y)    # define posição aleatória
	$AnimatedSprite2D.play("born")
	await $AnimatedSprite2D.animation_finished
	$AnimatedSprite2D.play("idle")

func get_inputs():
	var inputs = [max(0, linear_velocity.x),    # velocidade x positiva
				  max(0, -linear_velocity.x),    # velocidade x negative
				  max(0, linear_velocity.y),    # velocidade y positiva
				  max(0, -linear_velocity.y),    # velocidade y negativa
				  max(0, angular_velocity),    # velociadde angular positiva
				  max(0, -angular_velocity),    # velociadde angular negativa 
				  abs(rotation/(2*PI)), 
				  ENERGY,
				  1/(1+ENERGY), 
				  CLOSE_TO_FOOD,
				  COLLISION,
				  INVERSE_DISTANCE_TO_FOOD] + OUTPUTS
	return inputs
	
func get_outputs(outputs):
	OUTPUTS = outputs
	
	if outputs[0] == 1:
		linear_velocity = Vector2.RIGHT.rotated(rotation) * SPEED
		ENERGY -= 0.002
	if outputs[1] == 1:
		linear_velocity = Vector2.LEFT.rotated(rotation) * SPEED
		if outputs[0] != 1:
			ENERGY -= 0.002
	if outputs[2] == 1:
		angular_velocity = ROTATION   # gira continuamente
		ENERGY -= 0.0015
	if outputs[3] == 1:
		angular_velocity = -ROTATION   # gira continuamente
		if outputs[2] != 1:
			ENERGY -= 0.0015
		
	# Decaimento default de energia a cada iteração do blob.
	ENERGY -= 0.001
		
func _on_body_entered(body):
	# Máximo de MAX_NUM_CHILD filhos por blob.
	if NUM_CHILD <= MAX_NUM_CHILD and "blob_" in body.name and LIFE_STEPS > 100 and len(Global.blobs_alive) < Global.max_population:
		COLLISION = 1 
		
		if LAST_CHILD_STEP + COOL_DOWN_CHILD < LIFE_STEPS:
			LAST_CHILD_STEP = LIFE_STEPS
		else:
			return
		
		# Condição para evitar detecções duplicadas de colisões.
		if int(name.split("_")[1]) < int(body.name.split("_")[1]):
			NUM_CHILD += 1
			ENERGY -= 0.01    # gasta bastante energia para procriar
			offspring.emit(name, body.name)
	else:
		COLLISION = 0
