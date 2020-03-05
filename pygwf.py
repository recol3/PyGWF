import numpy as np
import struct
import re
import datetime
import zlib


COMMON_CODES = ("INT_8U", "CHAR_U", "CHAR_U", "INT_4U")
COMMON_LEN = 14
FRSH_CODES = ("STRING", "INT_2U", "STRING", "INT_4U")
FRSE_CODES = ("STRING",)*3 + ("INT_4U",)
FRSH_CLASS = 1
FRSE_CLASS = 2

ligo_dtypes = {
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
ligo_dtypes.update({
	"PTR_STRUCT": (ligo_dtypes["INT_2U"][0] + ligo_dtypes["INT_4U"][0], ligo_dtypes["INT_2U"][1] + ligo_dtypes["INT_4U"][1]),
	"COMPLEX_8":  (ligo_dtypes["REAL_4"][0] + ligo_dtypes["REAL_4"][0], ligo_dtypes["REAL_4"][1] + ligo_dtypes["REAL_4"][1]),
	"COMPLEX_16": (ligo_dtypes["REAL_8"][0] + ligo_dtypes["REAL_8"][0], ligo_dtypes["REAL_8"][1] + ligo_dtypes["REAL_8"][1]),
})


def get_type_code(ligo_name, str_len=None):
	if ligo_name == "STRING":
		if str_len is None or str_len < 1:
			raise ValueError
		return ligo_dtypes["INT_2U"][0] + str_len * ligo_dtypes["CHAR"][0], ligo_dtypes["INT_2U"][1] + str(str_len) + "s"
	elif ligo_name[:len("PTR_STRUCT")] == "PTR_STRUCT":
		return ligo_dtypes["PTR_STRUCT"]
	else:
		return ligo_dtypes[ligo_name]


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

			# If 2+ dimensional, arrays follow row-major (C-style) order (here we treat them as 1D)
			dimensions = []
			for name in arr_len_names:
				if name.isdigit():
					dimensions.append(int(name))
				else:
					dim_idx = var_names.index(name)
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
				idx_add_base, fs_add_base = get_type_code(size_code_base)
				fs_add = str(dims_prod) + fs_add_base if len(fs_add_base) == 1 else fs_add_base * dims_prod
				idx_add = idx_add_base * dims_prod

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


repeat_count_regex = re.compile(r"^\d+")
string_regex = re.compile(r"H\d+s")
def parse_struct(data, size_codes, var_names):
	n_bytes, struct_data, format_str = read_struct_with_common(data, size_codes, var_names)

	instance = {}
	out_idx = len(COMMON_CODES)
	for var_name, format_el in zip(var_names, format_str[out_idx:]):
		# STRINGs come back as two items (the string length, then the string itself -- for example the format "H19s" will be two items: the unsigned int 19, followed by a 19-character string). struct.unpack (called in read_struct and returned here as struct_data) interprets a pattern consisting of an integer followed by "s" as a single string, so the entire string will be one element of struct_data and n_items should be 2.
		# Arrays, however, aren't automatically grouped this way -- each byte is one element of struct_data. So we have to set n_items appropriately based on the format.
		if len(format_el) > 0 and format_el[0].isdigit():
			n_items = int(repeat_count_regex.search(format_el).group())
			var_items = struct_data[out_idx:out_idx + n_items]
		elif len(format_el) > 0 and format_el[0] == "H" and format_el[-1] == "s":
			n_strings = len(string_regex.findall(format_el))
			var_items = []
			for i in range(0, n_strings*2, 2):
				# Skip the STRING length element(s) and discard NULL terminator(s)
				var_items.append(struct_data[out_idx + i + 1].decode()[:-1])
				# TODO multiple null characters allowed at end of string -- check for
			n_items = 2*n_strings
		else:
			n_items = len(format_el)
			var_items = struct_data[out_idx:out_idx + n_items]

		out_idx += n_items
		instance[var_name] = var_items

	return n_bytes, instance


def get_dtype_nbits(type_id):
	# Only data types present in aux frame are 2 (REAL_8), 3 (REAL_4), and 10 (INT_4U), so these methods haven't been tested with other types, but it shouldn't be a problem (except perhaps for complex data; we raise a ValueError below until we test that)
	if type_id == 0:
		# CHAR
		block_nbits_len = 3
		dtype = np.dtype(np.int8)
	elif type_id == 1:
		# INT_2S
		block_nbits_len = 4
		dtype = np.dtype(np.int16)
	elif type_id == 2:
		# REAL_8
		block_nbits_len = 6
		dtype = np.dtype(np.float64)
	elif type_id == 3:
		# REAL_4
		block_nbits_len = 5
		dtype = np.dtype(np.float32)
	elif type_id == 4:
		# INT_4S
		block_nbits_len = 5
		dtype = np.dtype(np.int32)
	elif type_id == 5:
		# INT_8S
		block_nbits_len = 6
		dtype = np.dtype(np.int64)
	elif type_id == 6:
		# COMPLEX_8 (pair of REAL_4)
		raise NotImplementedError  # TODO untested (no complex data in raw aux frames; hoft?)
		block_nbits_len = 5
		dtype = np.dtype(np.complex64)
	elif type_id == 7:
		# COMPLEX_16 (pair of REAL_8)
		raise NotImplementedError  # TODO untested (no complex data in raw aux frames; hoft?)
		block_nbits_len = 6
		dtype = np.dtype(np.complex128)
	# elif type == 8:
		# STRING, but STRINGs don't have a fixed byte length and the specification doesn't define one, so fall through to ValueError
	elif type_id == 9:
		# INT_2U
		block_nbits_len = 4
		dtype = np.dtype(np.uint16)
	elif type_id == 10:
		# INT_4U
		block_nbits_len = 5
		dtype = np.dtype(np.uint32)
	elif type_id == 11:
		# INT_8U
		block_nbits_len = 6
		dtype = np.dtype(np.uint64)
	elif type_id == 12:
		# CHAR_U
		block_nbits_len = 3
		dtype = np.dtype(np.uint8)
	else:
		raise ValueError

	return dtype, block_nbits_len


def decompress_gzip(data, data_type, n_data):
	# TODO test with compression ID 259 (none in raw aux frames; hoft?)
	dtype, _ = get_dtype_nbits(data_type)
	data = zlib.decompress(b"".join(data))  # Bytes
	assert(len(data) == n_data*dtype.itemsize)
	return np.frombuffer(data, dtype=np.uint8).view(dtype)


def decompress_zsup(data, data_type, n_data, n_bytes):
	dtype, block_nbits_len = get_dtype_nbits(data_type)

	# Initialize decompressed as an array of unsigned ints with the same number of bits as the data dtype. After all the bit-level processing, we will view decompressed as the actual dtype.
	decompressed = np.zeros(n_data, dtype=np.dtype("uint{}".format(dtype.itemsize * 8)))

	block_size = struct.unpack("<H", b"".join(data[:2]))[0]  # Always the first 2 bytes (unsigned short)

	# Get bits, reshape them to a n_bytes x 8 matrix and flip rows to reverse bit order, then flatten back to 1D. This makes it so each data element's bits are contiguous (and little-endian). (Later, for each block, we have to flip the block element's bits back to big-endian, accounting for the element's number of bits.)
	# In NumPy 1.17+, unpackbits and packbits accept bitorder="little" as a parameter; this is a workaround for older versions. Probably unnecessary since LIGO software seems to generally be compatible with NumPy 1.18+, but also doesn't seem to hurt performance.
	# TODO? Remove bit order workaround?
	data_bits = np.unpackbits(np.frombuffer(b"".join(data[2:]), dtype=np.uint8)).astype(np.bool)
	data_bits_swapped = np.fliplr(data_bits.reshape(-1, 8)).flatten()
	if len(data_bits_swapped) != 8*(n_bytes - 2):
		raise ValueError

	n_data_filled, bit_idx = 0, 0
	while n_data_filled < n_data:  # Loop over blocks
		# Find the number of bits per data element for this block
		block_nbits_bits = data_bits_swapped[bit_idx:bit_idx + block_nbits_len][::-1]
		block_nbits = np.packbits(np.hstack((np.zeros(8 - block_nbits_len, dtype=np.bool), block_nbits_bits)))[0] + 1
		bit_idx += block_nbits_len

		if n_data_filled + block_size > n_data:
			block_size = n_data - n_data_filled

		if block_nbits < 1:
			raise ValueError
		elif block_nbits == 1:
			# This block is all 0s and there is no more data for this block; next bits are block_nbits_bits for the next block
			decompressed[n_data_filled:n_data_filled + block_size] = 0
		else:
			# For np.packbits to properly read the bits, we need to flip block elements' bits back to big-endian and pad with zeros on the left. We pad with enough zeros to match the number of bits of the final dtype (i.e. the dtype of decompressed) so we can copy directly into decompressed.
			block_bits = np.zeros((block_size, dtype.itemsize * 8), dtype=np.bool)
			block_bits[:, -block_nbits:] = np.fliplr(data_bits_swapped[bit_idx:bit_idx + block_nbits*block_size].reshape(-1, block_nbits))
			bit_idx += block_nbits*block_size

			# Read the bits and view them as unsigned ints with the correct number of bits and big-endian byte order
			uint_dtype = np.dtype("uint{}".format(dtype.itemsize * 8)).newbyteorder(">")
			block_decompressed = np.packbits(block_bits).view(uint_dtype)

			# Convert to signed int and subtract the value that was added during compression to make them unsigned
			block_decompressed = block_decompressed.astype(np.int) - 2**(block_nbits - 1) + 1

			decompressed[n_data_filled:n_data_filled + block_size] = block_decompressed

		n_data_filled += block_size

	# assert(bit_idx == len(data_bits_swapped))  # There are sometimes extra bits at the end...
	assert(not np.any(data_bits_swapped[bit_idx:]))  # ...which should be all 0

	# Undo differentiation
	for i in range(len(decompressed) - 1):
		decompressed[i + 1] = decompressed[i] + decompressed[i + 1]

	# View as actual dtype
	decompressed = decompressed.view(dtype)

	return decompressed


def decompress_frvect(frvect_instance):
	compress_id = frvect_instance["compress"][0]
	data = frvect_instance["data"]
	data_type = frvect_instance["type"][0]
	n_bytes = frvect_instance["nBytes"][0]
	n_data = frvect_instance["nData"][0]

	if compress_id == 257:
		return decompress_gzip(data=data, data_type=data_type, n_data=n_data)
	elif compress_id == 261 or compress_id == 264 or compress_id == 266:
		return decompress_zsup(data=data, data_type=data_type, n_data=n_data, n_bytes=n_bytes)
	else:
		# Not implemented/tested (none in raw auxiliary frames): 256 (uncompressed raw values); 259 (gzip, differential values)
		raise NotImplementedError


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
