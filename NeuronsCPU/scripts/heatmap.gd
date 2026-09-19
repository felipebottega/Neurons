class_name Heatmap
extends Control


const PALETTE: Array[Color] = [
	Color("#0000BB"),
	Color("#0000F6"),
	Color("#0020FF"),
	Color("#0054FF"),
	Color("#008CFF"),
	Color("#00C0FF"),
	Color("#0FF8E7"),
	Color("#39FFBE"),
	Color("#66FF90"),
	Color("#90FF66"),
	Color("#BEFF39"),
	Color("#E7FF0F"),
	Color("#FFD300"),
	Color("#FFA300"),
	Color("#FF6F00"),
	Color("#FF3F00"),
	Color("#F60B00"),
	Color("#BB0000"),
]

const COLORBAR_GAP := 12.0
const COLORBAR_HEIGHT := 12.0
const COLORBAR_LABEL_GAP := 4.0
const COLORBAR_FONT_SIZE := 12


var values: PackedFloat64Array = PackedFloat64Array()
var grid_size: Vector2i = Vector2i.ZERO
var value_min: float = 0.0
var value_max: float = 1.0


## Sets the scalar field displayed by the heatmap.
## [param new_values] contains one scalar value for each grid cell.
## [param new_grid_size] defines the horizontal and vertical neuron counts.
## [param min_value] defines the palette value represented by its first color.
## [param max_value] defines the palette value represented by its last color.
## Returns nothing.
func set_data(new_values: PackedFloat64Array, new_grid_size: Vector2i, min_value: float, max_value: float) -> void:
	values = new_values
	grid_size = new_grid_size
	value_min = min_value
	value_max = max_value
	queue_redraw()


func _draw() -> void:
	if values.is_empty() or grid_size.x <= 0 or grid_size.y <= 0:
		return

	var cell_size := minf(size.x / grid_size.x, size.y / grid_size.y)
	var map_size := Vector2(cell_size * grid_size.x, cell_size * grid_size.y)
	var offset := (size - map_size) / 2.0
	var map_rect := Rect2(offset, map_size)
	var value_span := value_max - value_min

	for j in grid_size.y:
		for i in grid_size.x:
			var index := i + grid_size.x * j
			var normalized := 0.5 if is_zero_approx(value_span) else (values[index] - value_min) / value_span
			var color := _palette_color(clampf(normalized, 0.0, 1.0))
			var rect := Rect2(
				offset + Vector2(i * cell_size, j * cell_size),
				Vector2(cell_size, cell_size)
			)
			draw_rect(rect, color)

	_draw_colorbar(map_rect)


## Draws the horizontal palette scale and its numerical values below the heatmap.
func _draw_colorbar(map_rect: Rect2) -> void:
	var bar_rect := Rect2(
		Vector2(map_rect.position.x, map_rect.end.y + COLORBAR_GAP),
		Vector2(map_rect.size.x, COLORBAR_HEIGHT)
	)
	var value_span := value_max - value_min

	if is_zero_approx(value_span):
		draw_rect(bar_rect, _palette_color(0.5))
		_draw_colorbar_label(bar_rect, 0.5, value_min)
		return

	var segment_count := maxi(1, ceili(bar_rect.size.x))
	var segment_width := bar_rect.size.x / segment_count

	for x in segment_count:
		var normalized := float(x) / maxf(1.0, segment_count - 1.0)
		var segment_rect := Rect2(
			bar_rect.position + Vector2(x * segment_width, 0.0),
			Vector2(segment_width + 0.5, bar_rect.size.y)
		)
		draw_rect(segment_rect, _palette_color(normalized))

	_draw_colorbar_label(bar_rect, 0.0, value_min)
	_draw_colorbar_label(bar_rect, 0.5, (value_min + value_max) / 2.0)
	_draw_colorbar_label(bar_rect, 1.0, value_max)


## Draws one centered colorbar value at the requested normalized position.
func _draw_colorbar_label(bar_rect: Rect2, normalized: float, value: float) -> void:
	var font := get_theme_default_font()
	var text := _format_colorbar_value(value)
	var text_size := font.get_string_size(text, HORIZONTAL_ALIGNMENT_LEFT, -1.0, COLORBAR_FONT_SIZE)
	var center_x := lerpf(bar_rect.position.x, bar_rect.end.x, normalized)
	var text_x := clampf(center_x - text_size.x / 2.0, bar_rect.position.x, bar_rect.end.x - text_size.x)
	var text_y := bar_rect.end.y + COLORBAR_LABEL_GAP + text_size.y
	var font_color := get_theme_color("font_color", "Label")

	draw_string(
		font,
		Vector2(text_x, text_y),
		text,
		HORIZONTAL_ALIGNMENT_LEFT,
		-1.0,
		COLORBAR_FONT_SIZE,
		font_color
	)


## Formats colorbar values compactly while preserving useful precision.
func _format_colorbar_value(value: float) -> String:
	if is_equal_approx(value, round(value)):
		return str(int(round(value)))

	var _scale := maxf(absf(value_min), absf(value_max))
	var span := absf(value_max - value_min)

	if (_scale > 0.0 and _scale < 0.001) or _scale >= 10000.0:
		return "%.2e" % value
	if span >= 10.0:
		return "%.1f" % value
	if span >= 1.0:
		return "%.2f" % value
	if span >= 0.1:
		return "%.3f" % value

	return "%.4f" % value


## Returns the interpolated palette color for a normalized value in [0, 1].
func _palette_color(value: float) -> Color:
	if PALETTE.size() == 1:
		return PALETTE[0]

	var scaled := value * (PALETTE.size() - 1)
	var left := floori(scaled)
	var right := mini(left + 1, PALETTE.size() - 1)
	var fraction := scaled - left

	return PALETTE[left].lerp(PALETTE[right], fraction)
