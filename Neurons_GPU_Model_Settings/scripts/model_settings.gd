class_name NeuronsModelSettings
extends RefCounted

# Immutable per-scene experimental settings. Defaults reproduce GPU V11.
# Keep UI limits and GPU packing in sync when introducing new controls.
const DEFAULTS := {
    "search_divisor": 10.0,
    "fatigue_divisor": 2.0,
    "decay_exponent": 32.0,
    "fatigue_enabled": true,
    "state_homeostasis_enabled": true,
    "synaptic_decay_enabled": true,
    "hebbian_plasticity_enabled": true,
    "structural_plasticity_enabled": true,
}

const FLAG_FATIGUE: int = 1
const FLAG_STATE_HOMEOSTASIS: int = 2
const FLAG_SYNAPTIC_DECAY: int = 4
const FLAG_HEBBIAN_PLASTICITY: int = 8
# Structural plasticity runs as a separate GPU stage; the CPU plan omits
# that stage when disabled rather than dispatching a no-op shader.

static func defaults() -> Dictionary:
    return DEFAULTS.duplicate(true)


static func flags(settings: Dictionary) -> int:
    var bits: int = 0
    if bool(settings.get("fatigue_enabled", true)):
        bits |= FLAG_FATIGUE
    if bool(settings.get("state_homeostasis_enabled", true)):
        bits |= FLAG_STATE_HOMEOSTASIS
    if bool(settings.get("synaptic_decay_enabled", true)):
        bits |= FLAG_SYNAPTIC_DECAY
    if bool(settings.get("hebbian_plasticity_enabled", true)):
        bits |= FLAG_HEBBIAN_PLASTICITY
    return bits


static func summary(settings: Dictionary) -> String:
    return (
        "MODEL SETTINGS (fixed at startup)\n"
        + "S divisor: %.0f  |  Fatigue divisor: %.2f  |  Decay exponent: %.0f\n" % [
            float(settings.get("search_divisor", 10.0)),
            float(settings.get("fatigue_divisor", 2.0)),
            float(settings.get("decay_exponent", 32.0))]
        + "Fatigue: %s  |  State + synaptic homeostasis: %s  |  Synaptic decay: %s" % [
            "ON" if bool(settings.get("fatigue_enabled", true)) else "OFF",
            "ON" if bool(settings.get("state_homeostasis_enabled", true)) else "OFF",
            "ON" if bool(settings.get("synaptic_decay_enabled", true)) else "OFF"]
        + "\nHebbian plasticity: %s  |  Structural plasticity: %s" % [
            "ON" if bool(settings.get("hebbian_plasticity_enabled", true)) else "OFF",
            "ON" if bool(settings.get("structural_plasticity_enabled", true)) else "OFF"]
    )
