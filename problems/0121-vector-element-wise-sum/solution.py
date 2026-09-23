def vector_sum(a: list[int|float], b: list[int|float]) -> list[int|float]:
	
	if len(a)==len(b):
		return [x + y for x, y in zip(a, b)]
	else:
		return -1
	# Return the element-wise sum of vectors 'a' and 'b'.
	# If vectors have different lengths, return -1.
	pass