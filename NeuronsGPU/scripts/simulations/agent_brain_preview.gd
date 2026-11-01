extends SubViewportContainer

# Pure visualization: never creates or advances a neural dynamics backend.
# Signal values arrive in spatial order from BrainPopulationGPU on demand.
const SIGNAL_PALETTE: Array[Color] = [
	Color("#0000BB"), Color("#0000F6"), Color("#0020FF"),
	Color("#0054FF"), Color("#008CFF"), Color("#00C0FF"),
	Color("#0FF8E7"), Color("#39FFBE"), Color("#66FF90"),
	Color("#90FF66"), Color("#BEFF39"), Color("#E7FF0F"),
	Color("#FFD300"), Color("#FFA300"), Color("#FF6F00"),
	Color("#FF3F00"), Color("#F60B00"), Color("#BB0000")
]
const NEURON_SPACING: float = 1.0
const BASE_NEURON_RADIUS: float = 0.23
const ORBIT_SENSITIVITY: float = 0.008
const ZOOM_IN_FACTOR: float = 0.88
const ZOOM_OUT_FACTOR: float = 1.14
const PAN_SENSITIVITY: float = 0.0015
const MAX_PITCH: float = PI * 89.0 / 180.0

@onready var subviewport: SubViewport = $SubViewport
@onready var neurons_instance: MultiMeshInstance3D = $SubViewport/BrainRoot/Neurons
@onready var camera: Camera3D = $SubViewport/BrainRoot/Camera3D
@onready var world_environment: WorldEnvironment = $SubViewport/BrainRoot/WorldEnvironment

var _size: Vector3i = Vector3i.ZERO
var _spatial_indices: PackedInt32Array = PackedInt32Array()
var _multimesh: MultiMesh
var _orbit_yaw: float = 0.0
var _orbit_pitch: float = 0.0
var _orbit_distance: float = 3.0
var _min_orbit_distance: float = 1.0
var _max_orbit_distance: float = 100.0
var _orbit_dragging: bool = false
var _pan_dragging: bool = false
var _camera_target: Vector3 = Vector3.ZERO


func _ready() -> void:
	# Receive pointer input only inside the preview, without forwarding wheel
	# events to the surrounding UI. OFF creates no preview node at all.
	mouse_filter = Control.MOUSE_FILTER_STOP
	mouse_force_pass_scroll_events = false
	tooltip_text = "Right drag: rotate | Middle drag: pan | Wheel: zoom"
	var environment := Environment.new()
	environment.background_mode = Environment.BG_COLOR
	environment.background_color = Color.BLACK
	world_environment.environment = environment
	camera.current = true
	camera.fov = 60.0
	camera.far = 100000.0
	subviewport.render_target_update_mode = SubViewport.UPDATE_DISABLED


func clear_brain() -> void:
	_orbit_dragging = false
	_pan_dragging = false
	_size = Vector3i.ZERO
	_spatial_indices = PackedInt32Array()
	_multimesh = null
	neurons_instance.multimesh = null
	_request_render()


func set_brain(brain: Dictionary) -> void:
	if brain.is_empty() or not brain.has("size") or not brain.has("neurons"):
		clear_brain()
		return
	_size = brain["size"]
	var neurons: Array = brain["neurons"]
	var sphere := SphereMesh.new()
	sphere.radial_segments = 8
	sphere.rings = 4
	# Increased radius relative to the full-size viewer to stay legible
	# in the much smaller simulation preview.
	sphere.radius = BASE_NEURON_RADIUS
	sphere.height = BASE_NEURON_RADIUS * 2.0
	var material := StandardMaterial3D.new()
	material.shading_mode = BaseMaterial3D.SHADING_MODE_UNSHADED
	material.vertex_color_use_as_albedo = true
	material.albedo_color = Color.WHITE
	sphere.material = material

	var multimesh := MultiMesh.new()
	multimesh.transform_format = MultiMesh.TRANSFORM_3D
	multimesh.use_colors = true
	multimesh.mesh = sphere
	multimesh.instance_count = neurons.size()
	_spatial_indices.resize(neurons.size())
	var plane: int = _size.x * _size.y
	for index in neurons.size():
		var neuron: Dictionary = neurons[index]
		var p: Vector3i = neuron["position"]
		_spatial_indices[index] = p.x + _size.x * p.y + plane * p.z
		var point := Vector3(
			(float(p.x) - 0.5 * float(_size.x - 1)) * NEURON_SPACING,
			(float(p.y) - 0.5 * float(_size.y - 1)) * NEURON_SPACING,
			(float(p.z) - 0.5 * float(_size.z - 1)) * NEURON_SPACING
		)
		multimesh.set_instance_transform(index, Transform3D(Basis.IDENTITY, point))
		multimesh.set_instance_color(index, SIGNAL_PALETTE[0])
	_multimesh = multimesh
	neurons_instance.multimesh = multimesh
	_frame_brain()
	_request_render()


func update_signals(signals: PackedFloat32Array) -> void:
	if _multimesh == null or signals.size() != _size.x * _size.y * _size.z:
		return
	var last_color: Color = Color.BLACK
	for index in _spatial_indices.size():
		var value: float = clampf(signals[_spatial_indices[index]], 0.0, 1.0)
		var scaled: float = value * float(SIGNAL_PALETTE.size() - 1)
		var left: int = mini(int(floor(scaled)), SIGNAL_PALETTE.size() - 1)
		var right: int = mini(left + 1, SIGNAL_PALETTE.size() - 1)
		last_color = SIGNAL_PALETTE[left].lerp(SIGNAL_PALETTE[right], scaled - float(left))
		_multimesh.set_instance_color(index, last_color)
	_request_render()


func _frame_brain() -> void:
	var extent := Vector3(
		float(maxi(_size.x - 1, 0)) * NEURON_SPACING,
		float(maxi(_size.y - 1, 0)) * NEURON_SPACING,
		float(maxi(_size.z - 1, 0)) * NEURON_SPACING
	)
	var radius: float = maxf(1.0, 0.5 * extent.length() + BASE_NEURON_RADIUS)
	var aspect: float = maxf(0.1, size.x / maxf(1.0, size.y))
	var distance: float = 1.35 * radius / (tan(deg_to_rad(camera.fov * 0.5)) * minf(aspect, 1.0))
	_orbit_dragging = false
	_pan_dragging = false
	_camera_target = Vector3.ZERO
	_orbit_yaw = deg_to_rad(45.0)
	_orbit_pitch = deg_to_rad(25.0)
	_orbit_distance = maxf(3.0, distance)
	# Allow close inspection while keeping the camera away from its pivot.
	_min_orbit_distance = maxf(0.5, radius * 0.35)
	_max_orbit_distance = maxf(_orbit_distance * 20.0, _min_orbit_distance * 2.0)
	_update_orbit_camera()


func _gui_input(event: InputEvent) -> void:
	# Match brain_viewer.gd camera controls exactly. This only changes the
	# preview camera; no neural state is read or recomputed.
	if _multimesh == null:
		return
	if event is InputEventMouseButton:
		var button := event as InputEventMouseButton
		if button.button_index == MOUSE_BUTTON_RIGHT:
			_orbit_dragging = button.pressed
			accept_event()
		elif button.button_index == MOUSE_BUTTON_MIDDLE:
			_pan_dragging = button.pressed
			accept_event()
		elif button.pressed and button.button_index == MOUSE_BUTTON_WHEEL_UP:
			_zoom_orbit(pow(ZOOM_IN_FACTOR, button.factor))
			accept_event()
		elif button.pressed and button.button_index == MOUSE_BUTTON_WHEEL_DOWN:
			_zoom_orbit(pow(ZOOM_OUT_FACTOR, button.factor))
			accept_event()
	elif event is InputEventMouseMotion:
		var motion := event as InputEventMouseMotion
		if _orbit_dragging:
			if not Input.is_mouse_button_pressed(MOUSE_BUTTON_RIGHT):
				_orbit_dragging = false
				return
			_orbit_yaw = wrapf(_orbit_yaw - motion.relative.x * ORBIT_SENSITIVITY, -PI, PI)
			_orbit_pitch = clampf(_orbit_pitch + motion.relative.y * ORBIT_SENSITIVITY, -MAX_PITCH, MAX_PITCH)
			_update_orbit_camera()
			accept_event()
		elif _pan_dragging:
			if not Input.is_mouse_button_pressed(MOUSE_BUTTON_MIDDLE):
				_pan_dragging = false
				return
			var right := camera.global_transform.basis.x.normalized()
			var up := camera.global_transform.basis.y.normalized()
			var pan_scale := _orbit_distance * PAN_SENSITIVITY
			_camera_target += (-right * motion.relative.x + up * motion.relative.y) * pan_scale
			_update_orbit_camera()
			accept_event()


func _zoom_orbit(factor: float) -> void:
	_orbit_distance = clampf(_orbit_distance * factor, _min_orbit_distance, _max_orbit_distance)
	_update_orbit_camera()


func _update_orbit_camera() -> void:
	var horizontal := cos(_orbit_pitch)
	var offset := Vector3(
		horizontal * sin(_orbit_yaw),
		sin(_orbit_pitch),
		horizontal * cos(_orbit_yaw)
	) * _orbit_distance
	camera.global_position = _camera_target + offset
	camera.look_at(_camera_target, Vector3.UP)
	_request_render()

func _request_render() -> void:
	if is_node_ready() and visible:
		subviewport.render_target_update_mode = SubViewport.UPDATE_ONCE


func request_render() -> void:
	_request_render()
