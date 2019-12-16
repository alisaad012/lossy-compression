import gpt_parsing
import arithmetic_compress
import arithmetic_decompress

def main(input_file):
	filename = '.'.join(input_file.split('.')[:-1])
	encoded_file = f"{filename}_encoded.txt"
	encoded_file_compressed = f"{filename}_encoded_compressed.txt"
	encoded_file_decompressed = f"{filename}_encoded_decompressed.txt"
	decoded_file = f"{filename}_decoded.txt"

	# Encode file with GPT-2
	gpt_parsing.parsing(input_file, encoded_file, encode=True, overwrite=True)

	# Compress with Arithmetic Coding
	arithmetic_compress.main(encoded_file, encoded_file_compressed)

	# Uncompress with Arithmetic Coding
	arithmetic_decompress.main(encoded_file_compressed, encoded_file_decompressed)

	# Decode with GPT-2
	gpt_parsing.parsing(encoded_file_decompressed, decoded_file, encode=False, overwrite=True)

	print("Complete")


if __name__ == '__main__':
	if len(sys.argv) == 2:
		main(sys.argv[1])
	else:
		print("Usage: python main.py input_file")
