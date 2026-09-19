#include "test_class.h"

#include <godot_cpp/core/class_db.hpp>

using namespace godot;

void TestClass::_bind_methods() {
	ClassDB::bind_method(D_METHOD("hello"), &TestClass::hello);
}

TestClass::TestClass() {
}

TestClass::~TestClass() {
}

String TestClass::hello() const {
	return "Hello from C++";
}