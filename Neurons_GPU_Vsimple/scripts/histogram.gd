class_name Histogram
extends Control


var bins: PackedInt32Array = PackedInt32Array()
var x_min: float = 0.0
var x_max: float = 1.0
var integer_x_labels: bool = false


func set_bins(new_bins: PackedInt32Array) -> void:
	bins = new_bins
	queue_redraw()


## Sets the values represented by the horizontal histogram range.
func set_x_range(min_value: float, max_value: float, use_integer_labels: bool = false) -> void:
	x_min = min_value
	x_max = max_value
	integer_x_labels = use_integer_labels
	queue_redraw()


func _draw() -> void:
	if bins.is_empty():
		return

	var bottom_margin := 25.0
	var plot_height := size.y - bottom_margin

	var max_count := 1
	for count in bins:
		max_count = maxi(max_count, count)

	var bar_width := size.x / bins.size()

	for i in bins.size():
		var height := plot_height * float(bins[i]) / max_count

		var rect := Rect2(i * bar_width, plot_height - height, bar_width - 1.0, height)

		draw_rect(rect, Color.BLUE)

	var font := ThemeDB.fallback_font
	var font_size := ThemeDB.fallback_font_size

	var label_count := 5

	for i in label_count:
		var fraction := float(i) / (label_count - 1)
		var value := lerpf(x_min, x_max, fraction)
		var label := str(roundi(value)) if integer_x_labels else str(snappedf(value, 0.01))
		var x := fraction * size.x

		draw_string(font, Vector2(x - 10.0, size.y - 5.0), label, HORIZONTAL_ALIGNMENT_LEFT, -1, font_size)
