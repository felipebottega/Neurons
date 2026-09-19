#ifndef TEST_CLASS_H
#define TEST_CLASS_H

#include <godot_cpp/classes/ref_counted.hpp>

namespace godot {

class TestClass : public RefCounted {
	GDCLASS(TestClass, RefCounted);

protected:
	static void _bind_methods();

public:
	TestClass();
	~TestClass();

	String hello() const;
};

}

#endif