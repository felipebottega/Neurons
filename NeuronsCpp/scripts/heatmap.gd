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

const NAN_COLOR: Color = Color("333333ff")
const COLORBAR_GAP := 26.0
const COLORBAR_HEIGHT := 8.0
const COLORBAR_LABEL_GAP := 1.0
const COLORBAR_FONT_SIZE := 12
const AXIS_TICK_LENGTH := 5.0
const AXIS_LABEL_GAP := 1.0
const AXIS_FONT_SIZE := 12
const AXIS_LEFT_MARGIN := 48.0
const AXIS_BOTTOM_MARGIN := 22.0

var values: PackedFloat64Array = PackedFloat64Array()
var grid_size: Vector2i = Vector2i.ZERO
var value_min: float = 0.0
var value_max: float = 1.0
var x_axis_min: float = NAN
var x_axis_max: float = NAN
var y_axis_min: float = NAN
var y_axis_max: float = NAN

## Sets the numerical ranges displayed on the heatmap axes.
## Leaving any range as NaN hides that axis values.
func set_axis_ranges(x_min: float = NAN, x_max: float = NAN, y_min: float = NAN, y_max: float = NAN) -> void:
	x_axis_min = x_min
	x_axis_max = x_max
	y_axis_min = y_min
	y_axis_max = y_max
	queue_redraw()


## Sets the scalar field displayed by the heatmap.
## [param new_values] contains one scalar value for each grid cell.
## [param new_grid_size] defines the horizontal and vertical neuron counts.
## [param min_value] defines the palette value represented by its first color.
## [param max_value] defines the palette value represented by its last color.
## Returns nothing.
func set_data(
		new_values: PackedFloat64Array,
		new_grid_size: Vector2i,
		min_value: float,
		max_value: float
) -> void:
	values = new_values
	grid_size = new_grid_size
	value_min = min_value
	value_max = max_value
	queue_redraw()


func _draw() -> void:
	if values.is_empty() or grid_size.x <= 0 or grid_size.y <= 0:
		return

	var show_x_axis = not is_nan(x_axis_min) and not is_nan(x_axis_max)
	var show_y_axis = not is_nan(y_axis_min) and not is_nan(y_axis_max)
	var left_margin = AXIS_LEFT_MARGIN if show_y_axis else 0.0
	var bottom_margin = AXIS_BOTTOM_MARGIN if show_x_axis else 0.0
	var available_size = Vector2(maxf(1.0, size.x - left_margin), maxf(1.0, size.y - bottom_margin))
	var cell_size = minf(available_size.x / grid_size.x, available_size.y / grid_size.y)
	var map_size = Vector2(cell_size * grid_size.x, cell_size * grid_size.y)
	var offset = Vector2(left_margin, 0.0) + (available_size - map_size) / 2.0
	var map_rect = Rect2(offset, map_size)
	var value_span = value_max - value_min

	for j in grid_size.y:
		for i in grid_size.x:
			var index = i + grid_size.x * j
			var value = values[index]
			var color = NAN_COLOR

			if not is_nan(value):
				var normalized = 0.5 if is_zero_approx(value_span) else (value - value_min) / value_span
				color = _palette_color(clampf(normalized, 0.0, 1.0))

			var rect = Rect2(
				offset + Vector2(i * cell_size, j * cell_size),
				Vector2(cell_size, cell_size)
			)
			draw_rect(rect, color)

	_draw_axes(map_rect, show_x_axis, show_y_axis)
	_draw_colorbar(map_rect)


## Draws minimum and maximum numerical ticks for the optional X and Y axes.
func _draw_axes(map_rect: Rect2, show_x_axis: bool, show_y_axis: bool) -> void:
	var font = get_theme_default_font()
	var font_color = get_theme_color("font_color", "Label")

	if show_x_axis:
		var axis_y = map_rect.end.y
		draw_line(Vector2(map_rect.position.x, axis_y), Vector2(map_rect.position.x, axis_y + AXIS_TICK_LENGTH), font_color)
		draw_line(Vector2(map_rect.end.x, axis_y), Vector2(map_rect.end.x, axis_y + AXIS_TICK_LENGTH), font_color)
		_draw_axis_label(font, font_color, _format_axis_value(x_axis_min), Vector2(map_rect.position.x, axis_y + AXIS_TICK_LENGTH + AXIS_LABEL_GAP), false)
		_draw_axis_label(font, font_color, _format_axis_value(x_axis_max), Vector2(map_rect.end.x, axis_y + AXIS_TICK_LENGTH + AXIS_LABEL_GAP), true)

	if show_y_axis:
		var axis_x = map_rect.position.x
		draw_line(Vector2(axis_x - AXIS_TICK_LENGTH, map_rect.position.y), Vector2(axis_x, map_rect.position.y), font_color)
		draw_line(Vector2(axis_x - AXIS_TICK_LENGTH, map_rect.end.y), Vector2(axis_x, map_rect.end.y), font_color)
		_draw_axis_label(font, font_color, _format_axis_value(y_axis_max), Vector2(axis_x - AXIS_TICK_LENGTH - AXIS_LABEL_GAP, map_rect.position.y), true)
		var y_min_text = _format_axis_value(y_axis_min)
		var y_min_size = font.get_string_size(y_min_text, HORIZONTAL_ALIGNMENT_LEFT, -1.0, AXIS_FONT_SIZE)
		_draw_axis_label(font, font_color, y_min_text, Vector2(axis_x - AXIS_TICK_LENGTH - AXIS_LABEL_GAP, map_rect.end.y - y_min_size.y), true)


func _draw_axis_label(font: Font, font_color: Color, text: String, anchor: Vector2, align_right: bool) -> void:
	var text_size = font.get_string_size(text, HORIZONTAL_ALIGNMENT_LEFT, -1.0, AXIS_FONT_SIZE)
	var x = anchor.x - text_size.x if align_right else anchor.x
	var y = anchor.y + text_size.y
	draw_string(font, Vector2(x, y), text, HORIZONTAL_ALIGNMENT_LEFT, -1.0, AXIS_FONT_SIZE, font_color)


func _format_axis_value(value: float) -> String:
	if is_equal_approx(value, round(value)):
		return str(int(round(value)))
	if absf(value) >= 10000.0 or (absf(value) > 0.0 and absf(value) < 0.001):
		return "%.2f" % value
	if absf(value) >= 10.0:
		return "%.1f" % value
	if absf(value) >= 1.0:
		return "%.2f" % value
	return "%.3f" % value


## Draws the horizontal palette scale and its numerical values below the heatmap.
func _draw_colorbar(map_rect: Rect2) -> void:
	var x_axis_extra = AXIS_BOTTOM_MARGIN if not is_nan(x_axis_min) and not is_nan(x_axis_max) else 0.0
	var bar_rect = Rect2(
		Vector2(map_rect.position.x, map_rect.end.y + COLORBAR_GAP + x_axis_extra),
		Vector2(map_rect.size.x, COLORBAR_HEIGHT)
	)
	var value_span = value_max - value_min

	if is_zero_approx(value_span):
		draw_rect(bar_rect, _palette_color(0.5))
		_draw_colorbar_label(bar_rect, 0.5, value_min)
		return

	var segment_count = maxi(1, ceili(bar_rect.size.x))
	var segment_width = bar_rect.size.x / segment_count

	for x in segment_count:
		var normalized = float(x) / maxf(1.0, segment_count - 1.0)
		var segment_rect = Rect2(
			bar_rect.position + Vector2(x * segment_width, 0.0),
			Vector2(segment_width + 0.5, bar_rect.size.y)
		)
		draw_rect(segment_rect, _palette_color(normalized))

	_draw_colorbar_label(bar_rect, 0.0, value_min)
	_draw_colorbar_label(bar_rect, 0.5, (value_min + value_max) / 2.0)
	_draw_colorbar_label(bar_rect, 1.0, value_max)


## Draws one centered colorbar value at the requested normalized position.
func _draw_colorbar_label(bar_rect: Rect2, normalized: float, value: float) -> void:
	var font = get_theme_default_font()
	var text = _format_colorbar_value(value)
	var text_size = font.get_string_size(text, HORIZONTAL_ALIGNMENT_LEFT, -1.0, COLORBAR_FONT_SIZE)
	var center_x = lerpf(bar_rect.position.x, bar_rect.end.x, normalized)
	var text_x = clampf(center_x - text_size.x / 2.0, bar_rect.position.x, bar_rect.end.x - text_size.x)
	var text_y = bar_rect.end.y + COLORBAR_LABEL_GAP + text_size.y
	var font_color = get_theme_color("font_color", "Label")

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

	var _scale = maxf(absf(value_min), absf(value_max))
	var span = absf(value_max - value_min)

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

	var scaled = value * (PALETTE.size() - 1)
	var left = floori(scaled)
	var right = mini(left + 1, PALETTE.size() - 1)
	var fraction = scaled - left

	return PALETTE[left].lerp(PALETTE[right], fraction)
