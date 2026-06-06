extends PathFollow3D

## NPC convoy follower: mirrors leader /convoy/status over HTTP.
@export var convoy_host: String = "127.0.0.1"
@export var convoy_port: int = 5000
@export var cruise_speed: float = 0.18
@export var slow_factor: float = 0.55
@export var start_progress: float = 0.0
@export var poll_interval_s: float = 0.12

var _http: HTTPRequest
var _poll_timer: float = 0.0
var _target_speed: float = 0.0
var _leader_state: String = "CRUISING"
var _request_in_flight: bool = false


func _ready() -> void:
	rotation_mode = PathFollow3D.ROTATION_Y
	loop = true
	progress = start_progress
	_http = HTTPRequest.new()
	add_child(_http)
	_http.request_completed.connect(_on_status_response)


func _process(delta: float) -> void:
	if _target_speed > 0.0:
		progress += _target_speed * delta

	_poll_timer += delta
	if _poll_timer >= poll_interval_s and not _request_in_flight:
		_poll_timer = 0.0
		_request_in_flight = true
		var url := "http://%s:%d/convoy/status" % [convoy_host, convoy_port]
		_http.request(url)


func _on_status_response(result: int, response_code: int, _headers: PackedStringArray, body: PackedByteArray) -> void:
	_request_in_flight = false
	if result != HTTPRequest.RESULT_SUCCESS or response_code != 200:
		_target_speed = 0.0
		return

	var text := body.get_string_from_utf8()
	var data = JSON.parse_string(text)
	if typeof(data) != TYPE_DICTIONARY:
		_target_speed = 0.0
		return

	_leader_state = str(data.get("state", "STOPPED")).to_upper()
	var leader_speed := float(data.get("speed", 0.0))

	if _leader_state == "STOPPED" or leader_speed <= 0.01:
		_target_speed = 0.0
	elif _leader_state == "SLOW":
		_target_speed = cruise_speed * slow_factor
	else:
		_target_speed = cruise_speed
