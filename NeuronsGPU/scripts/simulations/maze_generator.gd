extends RefCounted
## One connected, acyclic cell graph per maze: one topological route
## from start to target. The continuous arena may have several geometric
## trajectories through the same sequence of corridors.
##
## This helper owns layout, walls and an approximate walkable-space shortest
## distance field. Agents never receive the field or the route as input.

const GRID_SIZES = [
	Vector2i(2, 2), Vector2i(2, 2),
	Vector2i(3, 2), Vector2i(3, 2),
	Vector2i(3, 3), Vector2i(3, 3),
	Vector2i(4, 3), Vector2i(4, 3),
	Vector2i(4, 4), Vector2i(4, 4)
]
const DESIRED_HOPS = [2, 3, 3, 4, 4, 5, 5, 6, 7, 8]
const NEIGHBOR_OFFSETS = [
	Vector2i(1, 0), Vector2i(-1, 0),
	Vector2i(0, 1), Vector2i(0, -1)
]
const FIELD_DIRECTIONS = [
	Vector2i(1, 0), Vector2i(-1, 0),
	Vector2i(0, 1), Vector2i(0, -1),
	Vector2i(1, 1), Vector2i(1, -1),
	Vector2i(-1, 1), Vector2i(-1, -1)
]
const DISTANCE_FIELD_STEP: float = 7.0

var grid_size: Vector2i = Vector2i(2, 2)
var arena_size: Vector2 = Vector2.ZERO
var cell_size: Vector2 = Vector2.ZERO
var start_position: Vector2 = Vector2.ZERO
var target_position: Vector2 = Vector2.ZERO
var initial_distance: float = 1.0
var walls: Array[Rect2] = []
# Debug visualization only; it is never provided to the neural inputs.
var solution_path: PackedVector2Array = PackedVector2Array()

var _rng := RandomNumberGenerator.new()
var _links: Array = []
var _start_cell: int = 0
var _target_cell: int = 0
var _field_width: int = 0
var _field_height: int = 0
var _field_dx: float = 1.0
var _field_dy: float = 1.0
var _blocked := PackedByteArray()
var _distance_field := PackedFloat64Array()
var _agent_half_size: float = 0.0


func _init() -> void:
	_rng.randomize()


func generate(difficulty: int, size: Vector2, thickness: float, agent_size: float) -> void:
	var level: int = clampi(difficulty, 1, 10) - 1
	arena_size = size
	grid_size = GRID_SIZES[level]
	cell_size = Vector2(size.x / float(grid_size.x), size.y / float(grid_size.y))
	var cell_count: int = grid_size.x * grid_size.y
	_links.clear()
	for _index in cell_count:
		_links.append([])

	if level == 0:
		_generate_first_level()
	else:
		_start_cell = _rng.randi_range(0, cell_count - 1)
		_generate_spanning_tree(cell_count)
		_choose_target(DESIRED_HOPS[level])

	start_position = _center(_start_cell)
	target_position = _center(_target_cell)
	_build_solution_path()
	_build_walls(thickness)
	_agent_half_size = agent_size * 0.5
	_build_distance_field(_agent_half_size)
	initial_distance = maxf(distance_to_target(start_position), 0.00001)


func _center(index: int) -> Vector2:
	return Vector2(
		(float(index % grid_size.x) + 0.5) * cell_size.x,
		(float(int(index / grid_size.x)) + 0.5) * cell_size.y
	)


func _neighbor_indices(cell: int) -> Array[int]:
	var neighbors: Array[int] = []
	var x: int = cell % grid_size.x
	var y: int = int(cell / grid_size.x)
	for offset in NEIGHBOR_OFFSETS:
		var nx: int = x + offset.x
		var ny: int = y + offset.y
		if nx >= 0 and nx < grid_size.x and ny >= 0 and ny < grid_size.y:
			neighbors.append(ny * grid_size.x + nx)
	return neighbors


func _link(a: int, b: int) -> void:
	_links[a].append(b)
	_links[b].append(a)


func _generate_first_level() -> void:
	# Two open edges form exactly one turn, followed by a straight section.
	# The fourth cell is attached beyond the goal, not left disconnected.
	_start_cell = _rng.randi_range(0, 3)
	var x: int = _start_cell % 2
	var y: int = int(_start_cell / 2)
	var diagonal: int = (1 - y) * 2 + (1 - x)
	var middle: int
	if _rng.randi_range(0, 1) == 0:
		middle = y * 2 + (1 - x)
	else:
		middle = (1 - y) * 2 + x
	var last_cell: int = 6 - _start_cell - middle - diagonal
	_link(_start_cell, middle)
	_link(middle, diagonal)
	_link(diagonal, last_cell)
	_target_cell = diagonal


func _generate_spanning_tree(cell_count: int) -> void:
	# Randomized depth-first backtracker. Every visited cell gets exactly one
	# parent edge; an acyclic connected graph of cell_count - 1 open edges.
	var visited := PackedByteArray()
	visited.resize(cell_count)
	visited[_start_cell] = 1
	var stack: Array[int] = [_start_cell]
	while not stack.is_empty():
		var current: int = stack.back()
		var options: Array[int] = []
		for neighbor in _neighbor_indices(current):
			if visited[neighbor] == 0:
				options.append(neighbor)
		if options.is_empty():
			stack.pop_back()
			continue
		var chosen: int = options[_rng.randi_range(0, options.size() - 1)]
		_link(current, chosen)
		visited[chosen] = 1
		stack.append(chosen)


func _choose_target(desired_hops: int) -> void:
	var hops := PackedInt32Array()
	hops.resize(_links.size())
	hops.fill(-1)
	hops[_start_cell] = 0
	var queue: Array[int] = [_start_cell]
	var head: int = 0
	while head < queue.size():
		var current: int = queue[head]
		head += 1
		for next_cell in _links[current]:
			if hops[next_cell] == -1:
				hops[next_cell] = hops[current] + 1
				queue.append(next_cell)
	var best_error: int = 100000
	var candidates: Array[int] = []
	for cell in _links.size():
		if cell == _start_cell:
			continue
		var error: int = absi(hops[cell] - desired_hops)
		if error < best_error:
			best_error = error
			candidates.clear()
		if error == best_error:
			candidates.append(cell)
	_target_cell = candidates[_rng.randi_range(0, candidates.size() - 1)]


func _build_solution_path() -> void:
	# Reconstruct the one route in the maze's spanning tree, from the exact
	# starting position to the target. The polyline traverses open cell centers,
	# so it cannot cross a closed wall. It exists for rendering only.
	solution_path.clear()
	var predecessors := PackedInt32Array()
	predecessors.resize(_links.size())
	predecessors.fill(-1)
	var queue: Array[int] = [_start_cell]
	predecessors[_start_cell] = _start_cell
	var head: int = 0
	while head < queue.size():
		var current: int = queue[head]
		head += 1
		if current == _target_cell:
			break
		for neighbor in _links[current]:
			if predecessors[neighbor] == -1:
				predecessors[neighbor] = current
				queue.append(neighbor)

	if predecessors[_target_cell] == -1:
		push_error("Maze generator: no path from start to target.")
		return
	var reversed_cells: Array[int] = []
	var cell: int = _target_cell
	while cell != _start_cell:
		reversed_cells.append(cell)
		cell = predecessors[cell]
	reversed_cells.append(_start_cell)
	for i in range(reversed_cells.size() - 1, -1, -1):
		solution_path.append(_center(reversed_cells[i]))


func _build_walls(thickness: float) -> void:
	walls.clear()
	var half: float = thickness * 0.5
	for y in grid_size.y:
		for x in grid_size.x:
			var cell: int = y * grid_size.x + x
			if x + 1 < grid_size.x and not (cell + 1 in _links[cell]):
				walls.append(Rect2(
					Vector2(float(x + 1) * cell_size.x - half, float(y) * cell_size.y - half),
					Vector2(thickness, cell_size.y + thickness)
				))
			if y + 1 < grid_size.y and not (cell + grid_size.x in _links[cell]):
				walls.append(Rect2(
					Vector2(float(x) * cell_size.x - half, float(y + 1) * cell_size.y - half),
					Vector2(cell_size.x + thickness, thickness)
				))


func _is_walkable(position: Vector2, margin: float) -> bool:
	if position.x < margin or position.y < margin:
		return false
	if position.x > arena_size.x - margin or position.y > arena_size.y - margin:
		return false
	for wall in walls:
		if wall.grow(margin).has_point(position):
			return false
	return true


func _field_index(x: int, y: int) -> int:
	return y * _field_width + x


func _heap_push(heap: Array[Vector2], entry: Vector2) -> void:
	heap.append(entry)
	var index: int = heap.size() - 1
	while index > 0:
		var parent: int = int((index - 1) / 2)
		if heap[parent].x <= entry.x:
			break
		heap[index] = heap[parent]
		index = parent
	heap[index] = entry


func _heap_pop(heap: Array[Vector2]) -> Vector2:
	var result: Vector2 = heap[0]
	var last: Vector2 = heap.pop_back()
	if heap.is_empty():
		return result
	var index: int = 0
	while index * 2 + 1 < heap.size():
		var child: int = index * 2 + 1
		if child + 1 < heap.size() and heap[child + 1].x < heap[child].x:
			child += 1
		if last.x <= heap[child].x:
			break
		heap[index] = heap[child]
		index = child
	heap[index] = last
	return result


func _build_distance_field(agent_margin: float) -> void:
	# Dijkstra over free 2D space. The fitness is based on walking around
	# walls, not Euclidean distance through walls. Grid resolution controls
	# the geometric approximation error, not what the agent can sense.
	_field_width = maxi(3, int(ceil(arena_size.x / DISTANCE_FIELD_STEP)) + 1)
	_field_height = maxi(3, int(ceil(arena_size.y / DISTANCE_FIELD_STEP)) + 1)
	_field_dx = arena_size.x / float(_field_width - 1)
	_field_dy = arena_size.y / float(_field_height - 1)
	var count: int = _field_width * _field_height
	_blocked.resize(count)
	_distance_field.resize(count)
	_distance_field.fill(INF)
	for y in _field_height:
		for x in _field_width:
			var index: int = _field_index(x, y)
			var position: Vector2 = Vector2(x * _field_dx, y * _field_dy)
			_blocked[index] = 0 if _is_walkable(position, agent_margin + 0.01) else 1

	var gx: int = clampi(int(round(target_position.x / _field_dx)), 0, _field_width - 1)
	var gy: int = clampi(int(round(target_position.y / _field_dy)), 0, _field_height - 1)
	var goal_index: int = _field_index(gx, gy)
	if _blocked[goal_index] != 0:
		push_error("Maze generator: target cell is not traversable.")
		return
	_distance_field[goal_index] = 0.0
	var heap: Array[Vector2] = []
	_heap_push(heap, Vector2(0.0, float(goal_index)))
	while not heap.is_empty():
		var entry: Vector2 = _heap_pop(heap)
		var current: int = int(entry.y)
		if entry.x > _distance_field[current] + 0.00001:
			continue
		var cx: int = current % _field_width
		var cy: int = int(current / _field_width)
		for offset in FIELD_DIRECTIONS:
			var nx: int = cx + offset.x
			var ny: int = cy + offset.y
			if nx < 0 or nx >= _field_width or ny < 0 or ny >= _field_height:
				continue
			var neighbor: int = _field_index(nx, ny)
			if _blocked[neighbor] != 0:
				continue
			if offset.x != 0 and offset.y != 0:
				if _blocked[_field_index(nx, cy)] != 0 or _blocked[_field_index(cx, ny)] != 0:
					continue
			var step_length: float = Vector2(offset.x * _field_dx, offset.y * _field_dy).length()
			var candidate: float = entry.x + step_length
			if candidate < _distance_field[neighbor]:
				_distance_field[neighbor] = candidate
				_heap_push(heap, Vector2(candidate, float(neighbor)))


func distance_to_target(position: Vector2) -> float:
	if position.distance_squared_to(target_position) <= 0.000001:
		return 0.0
	if _field_width == 0:
		return INF
	var fx: float = clampf(position.x / _field_dx, 0.0, float(_field_width - 1))
	var fy: float = clampf(position.y / _field_dy, 0.0, float(_field_height - 1))
	var x0: int = clampi(int(floor(fx)), 0, _field_width - 2)
	var y0: int = clampi(int(floor(fy)), 0, _field_height - 2)
	var x1: int = x0 + 1
	var y1: int = y0 + 1
	var a: int = _field_index(x0, y0)
	var b: int = _field_index(x1, y0)
	var c: int = _field_index(x0, y1)
	var d: int = _field_index(x1, y1)
	if (
		_blocked[a] == 0 and _blocked[b] == 0
		and _blocked[c] == 0 and _blocked[d] == 0
		and not is_inf(_distance_field[a]) and not is_inf(_distance_field[b])
		and not is_inf(_distance_field[c]) and not is_inf(_distance_field[d])
	):
		# Bilinear interpolation gives a continuous potential over free quads.
		var tx: float = fx - float(x0)
		var ty: float = fy - float(y0)
		var top: float = lerpf(_distance_field[a], _distance_field[b], tx)
		var bottom: float = lerpf(_distance_field[c], _distance_field[d], tx)
		return lerpf(top, bottom, ty)
	# Close to a wall, use only field vertices that are actually visible
	# from the agent's center. Without this test a candidate across a wall
	# could artificially shorten the fitness distance through an obstacle.
	var best: float = INF
	for iy in range(maxi(y0 - 1, 0), mini(y1 + 2, _field_height)):
		for ix in range(maxi(x0 - 1, 0), mini(x1 + 2, _field_width)):
			var index: int = _field_index(ix, iy)
			if _blocked[index] == 0 and not is_inf(_distance_field[index]):
				var sample_position := Vector2(ix * _field_dx, iy * _field_dy)
				if _segment_is_clear(position, sample_position):
					best = minf(best, _distance_field[index] + position.distance_to(sample_position))
	return best


func _segment_is_clear(start: Vector2, end: Vector2) -> bool:
	# Uses the same inflated wall rectangles as physical collision. Only
	# called for near-wall field samples (not all interior interpolations).
	var clearance: float = _agent_half_size + 0.01
	for wall in walls:
		var inflated: Rect2 = wall.grow(clearance)
		if inflated.has_point(start) or inflated.has_point(end):
			return false
		if _segment_intersects_rect(start, end, inflated):
			return false
	return true


func _segment_intersects_rect(start: Vector2, end: Vector2, rect: Rect2) -> bool:
	# Segment/AABB clipping (slab method). Godot's Rect2 has no
	# intersects_segment() method. Treat edge contact as collision too.
	var direction: Vector2 = end - start
	var minimum: Vector2 = rect.position
	var maximum: Vector2 = rect.end
	var t_enter: float = 0.0
	var t_exit: float = 1.0
	for axis in 2:
		var origin: float = start.x if axis == 0 else start.y
		var delta: float = direction.x if axis == 0 else direction.y
		var lower: float = minimum.x if axis == 0 else minimum.y
		var upper: float = maximum.x if axis == 0 else maximum.y
		if absf(delta) < 0.000000001:
			if origin < lower or origin > upper:
				return false
			continue
		var t1: float = (lower - origin) / delta
		var t2: float = (upper - origin) / delta
		if t1 > t2:
			var swap: float = t1
			t1 = t2
			t2 = swap
		t_enter = maxf(t_enter, t1)
		t_exit = minf(t_exit, t2)
		if t_enter > t_exit:
			return false
	return true


func progress(position: Vector2) -> float:
	var distance: float = distance_to_target(position)
	if is_inf(distance):
		return 0.0
	return clampf(1.0 - distance / initial_distance, 0.0, 1.0)
