import os
import sys

from collections import OrderedDict

import gpt2_parsing
import arithmetic_compress
import arithmetic_decompress

def main(input_file):
    filename = '.'.join(input_file.split('.')[:-1])
    compressed_file = f"{filename}_compressed.txt"
    encoded_file = f"{filename}_encoded.txt"
    encoded_file_compressed = f"{filename}_encoded_compressed.txt"
    encoded_file_decompressed = f"{filename}_encoded_decompressed.txt"
    decoded_file = f"{filename}_decoded.txt"

    # Compress without GPT-2
    arithmetic_compress.main((input_file, compressed_file))

    # Encode file with GPT-2
    gpt2_parsing.parsing(input_file, encoded_file, encode=True, overwrite=True)

    # Compress with Arithmetic Coding
    arithmetic_compress.main((encoded_file, encoded_file_compressed))

    # Uncompress with Arithmetic Coding
    arithmetic_decompress.main((encoded_file_compressed, encoded_file_decompressed))

    # Decode with GPT-2
    gpt2_parsing.parsing(encoded_file_decompressed, decoded_file, encode=False, overwrite=True)

    files = OrderedDict()
    files[input_file] = "Original"
    files[encoded_file] = "Encoded with GPT-2"
    files[compressed_file] = "Compressed without GPT-2"
    files[encoded_file_compressed] = "Compressed with GPT-2"
    files[decoded_file] = "Decompressed and decoded with GPT-2"

    max_len_filename = max([len(filename) for filename in files.keys()])
    max_len_comment = max([len(comment) for comment in files.values()])

    original_filesize = os.stat(input_file).st_size

    print("========== Summary ==========")
    for file_, comment in files.items():
        filesize = os.stat(file_).st_size
        print(f"{file_:{max_len_filename}}   {comment:{max_len_comment}} {filesize:>10} bytes   {filesize/original_filesize*100:>5.2f}%")

if __name__ == '__main__':
    if len(sys.argv) == 2:
        main(sys.argv[1])
    else:
        print("Usage: python main.py input_file")
