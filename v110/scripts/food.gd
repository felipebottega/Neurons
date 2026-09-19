extends Node2D


signal entered_food_area
signal exited_food_area
signal eat


func _on_detection_area_body_entered(blob: Node2D) -> void:
	if blob.name.begins_with("blob_"):  # garante que é um blob
		var distance = global_position.distance_to(blob.global_position)
		entered_food_area.emit(blob, distance)
		
func _on_detection_area_body_exited(blob: Node2D) -> void:
	if blob.name.begins_with("blob_"):  
		exited_food_area.emit(blob)

func _on_eat_area_body_entered(blob: Node2D) -> void:
	if blob.name.begins_with("blob_"):  
		eat.emit(blob)
		queue_free()  # remove esta comida da cena
