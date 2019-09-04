import numpy as np
import struct
import datetime


COMMON_CODES = ("INT_8U", "CHAR_U", "CHAR_U", "INT_4U")
COMMON_LEN = 14
FRSH_CODES = ("STRING", "INT_2U", "STRING", "INT_4U")
FRSE_CODES = ("STRING",)*3 + ("INT_4U",)
FRSH_CLASS = 1
FRSE_CLASS = 2


def get_type_code(ligo_name, str_len=None):
	types = {
		# "LIGO name": (num bytes, struct code)
		"CHAR":   (1, "c"),
		"CHAR_U": (1, "B"),
		"INT_2S": (2, "h"),
		"INT_2U": (2, "H"),
		"INT_4S": (4, "i"),
		"INT_4U": (4, "I"),
		"INT_8S": (8, "q"),
		"INT_8U": (8, "Q"),
		"REAL_4": (4, "f"),
		"REAL_8": (8, "d"),
	}
	types.update({
		"PTR_STRUCT": (types["INT_2U"][0] + types["INT_4U"][0], types["INT_2U"][1] + types["INT_4U"][1]),
		"COMPLEX_8":  (types["REAL_4"][0] + types["REAL_4"][0], types["REAL_4"][1] + types["REAL_4"][1]),
		"COMPLEX_16": (types["REAL_8"][0] + types["REAL_8"][0], types["REAL_8"][1] + types["REAL_8"][1]),
	})

	if ligo_name == "STRING":
		if str_len is None or str_len < 1:
			raise ValueError
		return types["INT_2U"][0] + str_len*types["CHAR"][0], types["INT_2U"][1] + str(str_len) + "s"
	elif ligo_name[:len("PTR_STRUCT")] == "PTR_STRUCT":
		return types["PTR_STRUCT"]
	else:
		return types[ligo_name]


def get_str_len(data):
	length, format_str = get_type_code("INT_2U")
	return struct.unpack("<" + format_str, data[:length].tobytes())[0]


def build_format_str(data, size_codes, var_names=None):
	if var_names is not None and len(var_names) != len(size_codes):
		raise ValueError

	idx = 0
	var_start_locs, var_format_codes = [], []
	# We make a list of format string elements rather than just building the string here because we need to index the elements (the format string's characters don't directly correspond to the size_codes, e.g. if there were PTR_STRUCTs)
	# TODO? refactor some to funcs

	for size_code in size_codes:
		if size_code[-1] == "]":
			# Array; we have to get the dimensions from previous variables
			if var_names is None:
				raise ValueError

			n_dims = size_code.count("[")
			if n_dims != size_code.count("]"):
				raise ValueError

			size_code_base = size_code.split("[")[0]
			arr_len_names = [name[:-1] for name in size_code.split("[")[1:]]  # Remove "]"s

			dim_idxs = [var_names.index(name) for name in arr_len_names]
			dimensions = []
			for dim_idx in dim_idxs:
				# If 2+ dimensional, arrays follow row-major (C-style) order (here we treat them as 1D)
				dim_bytes_start = var_start_locs[dim_idx]
				dim_size_code = var_format_codes[dim_idx]

				if len(dim_size_code) != 1:
					# Can't be a type we need more than one format character for (i.e. STRING, PTR_STRUCT, COMPLEX*)
					raise ValueError

				dim_bytes_len = struct.calcsize(dim_size_code)
				dimension = struct.unpack(dim_size_code, data[dim_bytes_start:dim_bytes_start + dim_bytes_len].tobytes())[0]

				special_no_data_names = "nStat", "nADC", "nProc", "nSim", "nSer", "nSummary", "nEvent", "nSimEvent"
				# "nEvent" and "nSimEvent" aren't used as array dimensions for other variables
				if dimension == 2**32 - 1 and any(name in arr_len_names for name in special_no_data_names):
					dimension = 0

				dimensions.append(dimension)
			dims_prod = np.prod(dimensions)

			if size_code_base == "STRING":
				idx_add = 0
				fs_add = ""
				for _ in range(dims_prod):
					str_len = get_str_len(data[idx + idx_add:])
					idx_add_add, fs_add_add = get_type_code(size_code_base, str_len)
					fs_add += fs_add_add
					idx_add += idx_add_add

			else:
				idx_add, fs_add = get_type_code(size_code_base)
				fs_add *= dims_prod  # TODO? go back to integer counts if we change parse_struct to not rely on string length anywhere
				idx_add *= dims_prod

		elif size_code == "STRING":
			str_len = get_str_len(data[idx:])
			idx_add, fs_add = get_type_code(size_code, str_len)

		else:
			idx_add, fs_add = get_type_code(size_code)

		var_start_locs.append(idx)
		var_format_codes.append(fs_add)
		idx += idx_add

	assert(idx == struct.calcsize("<" + "".join(var_format_codes)))
	assert(len(size_codes) == len(var_start_locs))
	assert(len(size_codes) == len(var_format_codes))

	return idx, var_format_codes


def read_struct(data, size_codes, expected_size=None, var_names=None):
	n_bytes, format_str = build_format_str(data, size_codes, var_names)
	if expected_size is not None and n_bytes != expected_size:
		raise ValueError
	return n_bytes, struct.unpack("<" + "".join(format_str), data[:n_bytes].tobytes()), format_str


def read_common(data):
	return read_struct(data, COMMON_CODES, COMMON_LEN)


def read_struct_with_common(data, size_codes, var_names=None):
	# Reads the common elements of all frame structures, the first of which is the byte length of the entire structure; then reads the rest of the structure and checks that that its total length matches that byte length
	common_bytes, common, common_format_str = read_common(data)  # Already checks that common_bytes == COMMON_LEN
	tot_len = common[0]
	remaining_bytes, remaining, format_str = read_struct(data[common_bytes:], size_codes, expected_size=tot_len - common_bytes, var_names=var_names)
	return common_bytes + remaining_bytes, common + remaining, common_format_str + format_str


def read_frsh(data):
	return read_struct_with_common(data, FRSH_CODES)


def read_frse(data):
	return read_struct_with_common(data, FRSE_CODES)


def read_frheader(data):
	# TODO? any checks in here?
	size_codes = ("CHAR",)*5 + ("CHAR_U",)*21 + ("REAL_4", "REAL_8") + ("CHAR_U",)*2
	return read_struct(data, size_codes, 40)


def parse_dict(data):
	idx = 0

	n_bytes, frsh_out, _ = read_frsh(data)
	idx += n_bytes
	# read_frsh -> read_struct_with_common -> read_struct already checks that frsh_length == n_bytes
	assert(len(frsh_out) == 10)
	frsh_length        = frsh_out[0]
	frsh_chktype       = frsh_out[1]
	frsh_class         = frsh_out[2]
	frsh_instance      = frsh_out[3]
	struct_name_len    = frsh_out[4]
	struct_name        = frsh_out[5].decode()[:-1]
	struct_class       = frsh_out[6]
	struct_comment_len = frsh_out[7]
	struct_comment     = frsh_out[8].decode()[:-1]
	frsh_chksum        = frsh_out[9]

	if frsh_class != FRSH_CLASS:
		raise ValueError

	var_names, size_codes, var_comments = [], [], []
	while True:
		cur_class = read_common(data[idx:])[1][2]  # We do this again in read_frse; how slow?
		if cur_class != FRSE_CLASS:
			break

		n_bytes, frse_out, _ = read_frse(data[idx:])
		idx += n_bytes
		# read_frse -> read_struct_with_common -> read_struct already checks that frse_length == n_bytes
		assert(len(frse_out) == 11)
		frse_length      = frse_out[0]
		frse_chktype     = frse_out[1]
		frse_class       = frse_out[2]
		frse_instance    = frse_out[3]
		elem_name_len    = frse_out[4]
		elem_name        = frse_out[5].decode()[:-1]
		elem_class_len   = frse_out[6]
		elem_class       = frse_out[7].decode()[:-1]
		elem_comment_len = frse_out[8]
		elem_comment     = frse_out[9].decode()[:-1]
		frse_chksum      = frse_out[10]

		var_names.append(elem_name)
		size_codes.append(elem_class)
		var_comments.append(elem_comment)

	return idx, struct_name, struct_class, struct_comment, var_names, size_codes, var_comments


def parse_struct(data, size_codes, var_names):
	n_bytes, struct_data, format_str = read_struct_with_common(data, size_codes, var_names)

	instance = {}
	out_idx = len(COMMON_CODES)
	for var_name, format_el in zip(var_names, format_str[out_idx:]):
		# STRINGs come back as two items (the string length, then the string itself -- for example the format "H19s" will be two items: the unsigned int 19, followed by a 19-character string). struct.unpack (called in read_struct and returned here as struct_data) interprets a pattern consisting of an integer followed by "s" as a single string, so the entire string will be one element of struct_data and n_items should be 2.
		# Arrays, however, aren't automatically grouped this way -- each byte is one element of struct_data. So we have to set n_items appropriately based on the format.
		# The length of the format_el string (excluding digits for length of STRINGs) should directly correspond to the number of elements read into struct_data for the current variable/size_code
		# TODO? Find a different way (and use integer counts for repeats) because the below line slows us down significantly
		n_items = sum(not ch.isdigit() for ch in format_el)
		var_items = list(struct_data[out_idx:out_idx + n_items])  # TODO? More efficient if this isn't a list?
		out_idx += n_items

		# Discard the STRING length element; also decode the string and discard the NULL terminator
		str_len_idxs = []
		var_items_idx = 0
		for format_el_idx in range(len(format_el) - 1):
			if format_el[format_el_idx + 1].isdigit() and not format_el[format_el_idx].isdigit():
				assert(format_el[format_el_idx] == "H")
				str_len_idxs.append(var_items_idx)
				var_items[var_items_idx + 1] = var_items[var_items_idx + 1].decode()[:-1]
			if not format_el[format_el_idx].isdigit():
				var_items_idx += 1
		var_items = [item for i, item in enumerate(var_items) if i not in str_len_idxs]

		instance[var_name] = var_items

	return n_bytes, instance


def read_frame(frame_data, print_progress=False):
	idx, header_out, _ = read_frheader(frame_data)

	pct_done = 0
	classes = {}
	instances = {}
	while idx < len(frame_data):
		next_class, next_instance = read_common(frame_data[idx:])[1][2:4]
		if next_class == FRSH_CLASS:
			n_bytes, new_struct_name, new_struct_class, new_struct_comment, new_struct_var_names, new_struct_size_codes, new_struct_var_comments = parse_dict(frame_data[idx:])
			classes[new_struct_class] = (new_struct_name, new_struct_comment, new_struct_var_names, new_struct_size_codes, new_struct_var_comments)

		else:
			if next_class not in instances:
				instances[next_class] = []

#             if next_instance != len(instances[next_class]):
#                 raise ValueError
			# Instance counts are reset at the end of each (sub)frame

			var_names, size_codes = classes[next_class][2:4]
			n_bytes, instance = parse_struct(frame_data[idx:], size_codes, var_names)
			instances[next_class].append(instance)

		idx += n_bytes

		if print_progress:
			if idx*100 // len(frame_data) - pct_done >= 1:
				pct_done = idx*100 // len(frame_data)
				print("{}: {}%".format(datetime.datetime.now(), pct_done))

	assert(idx == len(frame_data))

	return classes, instances
