#include "test_class.h"
#include "brain_dynamics_native.h"
#include "functions_native.h"
#include "genome_native.h"
#include "brain_builder_native.h"
#include "offspring_native.h"

#include <gdextension_interface.h>
#include <godot_cpp/core/class_db.hpp>
#include <godot_cpp/core/defs.hpp>
#include <godot_cpp/godot.hpp>

using namespace godot;

void initialize_neurons_module(ModuleInitializationLevel p_level) {
	if (p_level != MODULE_INITIALIZATION_LEVEL_SCENE) {
		return;
	}

	ClassDB::register_class<TestClass>();
	ClassDB::register_class<BrainDynamicsNative>();
	ClassDB::register_class<FunctionsNative>();
	ClassDB::register_class<GenomeNative>();
	ClassDB::register_class<BrainBuilderNative>();
	ClassDB::register_class<OffspringNative>();
}

void uninitialize_neurons_module(ModuleInitializationLevel p_level) {
	if (p_level != MODULE_INITIALIZATION_LEVEL_SCENE) {
		return;
	}
}

extern "C" {

GDExtensionBool GDE_EXPORT neurons_library_init(
	GDExtensionInterfaceGetProcAddress p_get_proc_address,
	const GDExtensionClassLibraryPtr p_library,
	GDExtensionInitialization *r_initialization
) {
	GDExtensionBinding::InitObject init_obj(
		p_get_proc_address,
		p_library,
		r_initialization
	);

	init_obj.register_initializer(initialize_neurons_module);
	init_obj.register_terminator(uninitialize_neurons_module);
	init_obj.set_minimum_library_initialization_level(MODULE_INITIALIZATION_LEVEL_SCENE);

	return init_obj.init();
}

}
