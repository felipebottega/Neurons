class_name NeuronsModelSettingsDialog
extends AcceptDialog

# The only commit is AcceptDialog.confirmed; closing the dialog does not
# modify the simulation and leaves Start disabled until a later confirmation.
signal settings_confirmed(settings: Dictionary)

var _draft: Dictionary = NeuronsModelSettings.defaults()
var _controls: Dictionary = {}


func _ready() -> void:
    title = "Neural Model Settings — GPU V11"
    # Avoid AcceptDialog.dialog_text: its built-in label overlaps custom content.
    exclusive = true
    unresizable = true
    min_size = Vector2i(720, 640)
    get_ok_button().text = "OK — Apply"
    confirmed.connect(_on_confirmed)

    var content := VBoxContainer.new()
    content.add_theme_constant_override("separation", 8)
    add_child(content)

    var introduction := Label.new()
    introduction.text = "Configure the model once, then click OK. Restart the scene to change it."
    introduction.add_theme_font_size_override("font_size", 14)
    content.add_child(introduction)

    _add_section(content, "Constants", "These constants affect the initial graph or the neural dynamics.")
    _add_slider(content, "search_divisor", "Initial search S(y) divisor", 2.0, 30.0, 1.0,
        "Changes the initial connectivity search radius. Original value: 10. Also affects the later reconnection search radius.", "%.0f")
    _add_slider(content, "fatigue_divisor", "Fatigue recovery divisor", 1.0, 8.0, 0.25,
        "For a neuron that does not fire: rho <- rho / divisor. Original value: 2. Fatigue OFF overrides this control.", "%.2f")
    _add_slider(content, "decay_exponent", "Synaptic decay exponent", 6.0, 64.0, 2.0,
        "Passive loss per iteration: exp(-exponent / D). Original value: 32. Synaptic decay OFF overrides this control.", "%.0f")

    content.add_child(HSeparator.new())
    _add_section(content, "Optional equations", "ON reproduces V11. OFF isolates that mechanism without changing the genetic equations.")
    _add_toggle(content, "fatigue_enabled", "Refractory fatigue",
        "OFF: ignores the refractory contribution to the firing threshold and keeps fatigue at zero.")
    _add_toggle(content, "state_homeostasis_enabled", "Neuronal and synaptic homeostasis",
        "OFF: skips low/high activity corrections of neuronal state and the associated downscaling of incoming non-input synapses. ON: synaptic downscaling occurs only with the high-activity state correction. Isolated-neuron reconnection is unaffected.")
    _add_toggle(content, "synaptic_decay_enabled", "Passive synaptic decay",
        "OFF: disables passive weight loss, but leaves Hebbian and modulatory weight updates enabled.")
    _add_toggle(content, "hebbian_plasticity_enabled", "Hebbian plasticity",
        "OFF: disables Hebbian weight reinforcement without disabling eligibility traces, endogenous modulation or decay.")
    _add_toggle(content, "structural_plasticity_enabled", "Structural plasticity",
        "OFF: prevents activity-dependent formation of new internal connections. Homeostatic reconnection remains enabled.")

    var reset_button := Button.new()
    reset_button.text = "Restore V11 defaults"
    reset_button.tooltip_text = "Restore the V11 numerical constants and enable all five optional mechanisms."
    reset_button.pressed.connect(_reset_defaults)
    content.add_child(reset_button)

    var note := Label.new()
    note.text = "All settings are locked after OK. The save-file format remains unchanged."
    note.add_theme_color_override("font_color", Color(0.68, 0.77, 0.75))
    note.add_theme_font_size_override("font_size", 13)
    content.add_child(note)


func _add_section(parent: VBoxContainer, heading: String, description: String) -> void:
    var label := Label.new()
    label.text = heading
    label.add_theme_font_size_override("font_size", 20)
    parent.add_child(label)
    var info := Label.new()
    info.text = description
    info.add_theme_font_size_override("font_size", 13)
    info.add_theme_color_override("font_color", Color(0.69, 0.77, 0.76))
    parent.add_child(info)


func _add_slider(parent: VBoxContainer, key: String, label_text: String,
        minimum: float, maximum: float, increment: float, explanation: String,
        value_format: String) -> void:
    var row := HBoxContainer.new()
    row.add_theme_constant_override("separation", 12)
    row.tooltip_text = explanation
    parent.add_child(row)
    var name_label := Label.new()
    name_label.custom_minimum_size = Vector2(235, 0)
    name_label.text = label_text
    name_label.tooltip_text = explanation
    row.add_child(name_label)
    var slider := HSlider.new()
    slider.custom_minimum_size = Vector2(310, 26)
    slider.size_flags_horizontal = Control.SIZE_EXPAND_FILL
    slider.min_value = minimum
    slider.max_value = maximum
    slider.step = increment
    slider.value = float(_draft[key])
    slider.tooltip_text = explanation
    row.add_child(slider)
    _controls[key] = slider
    var current_value := Label.new()
    current_value.custom_minimum_size = Vector2(56, 0)
    current_value.horizontal_alignment = HORIZONTAL_ALIGNMENT_RIGHT
    current_value.text = value_format % slider.value
    row.add_child(current_value)
    slider.value_changed.connect(func(value: float) -> void:
        _draft[key] = value
        current_value.text = value_format % value)


func _add_toggle(parent: VBoxContainer, key: String, label_text: String, explanation: String) -> void:
    var row := HBoxContainer.new()
    row.tooltip_text = explanation
    parent.add_child(row)
    var name_label := Label.new()
    name_label.text = label_text
    name_label.size_flags_horizontal = Control.SIZE_EXPAND_FILL
    name_label.tooltip_text = explanation
    row.add_child(name_label)
    var toggle := Button.new()
    toggle.toggle_mode = true
    toggle.custom_minimum_size = Vector2(86, 28)
    toggle.button_pressed = bool(_draft[key])
    toggle.text = "ON" if toggle.button_pressed else "OFF"
    toggle.tooltip_text = explanation
    row.add_child(toggle)
    _controls[key] = toggle
    toggle.toggled.connect(func(enabled: bool) -> void:
        _draft[key] = enabled
        toggle.text = "ON" if enabled else "OFF")


func _reset_defaults() -> void:
    _draft = NeuronsModelSettings.defaults()
    for key in _controls:
        var control: Control = _controls[key]
        if control is HSlider:
            control.value = float(_draft[key])
        elif control is Button:
            control.button_pressed = bool(_draft[key])
            control.text = "ON" if control.button_pressed else "OFF"


func _on_confirmed() -> void:
    settings_confirmed.emit(_draft.duplicate(true))
