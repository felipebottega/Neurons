extends Node

var frame_count = 0    # contagem real de frames  
var process_frame_iterator = 0    # contagem de frames do ponto de vista dos blobs
var blob_count = 0
var max_blobs = 20
var blobs_alive = []
var max_population = 2 * max_blobs
