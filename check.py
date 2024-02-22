import os
import numpy as np

def get_directory_size(directory):
    total_size = 0
    for root, _, filenames in os.walk(directory):
        for filename in filenames:
            file_path = os.path.join(root, filename)
            total_size += os.path.getsize(file_path)
    return total_size

def get_npy_file_sizes(directory):
    npy_file_sizes = {}
    for root, _, filenames in os.walk(directory):
        for filename in filenames:
            if filename.endswith(".npy"):
                file_path = os.path.join(root, filename)
                size_bytes = os.path.getsize(file_path)
                npy_file_sizes[filename] = size_bytes
    return npy_file_sizes

def get_npy_file_shapes(directory):
    npy_file_shapes = {}
    for root, _, filenames in os.walk(directory):
        for filename in filenames:
            if filename.endswith(".npy"):
                file_path = os.path.join(root, filename)
                array = np.load(file_path)
                npy_file_shapes[filename] = array.shape
    return npy_file_shapes


if __name__ == "__main__":
    dir = "imagenet-mini"
    total_size_bytes = get_directory_size(dir)
    total_size_mb = total_size_bytes / (1024 * 1024)
    print(f"Total size of original data: {total_size_mb:.2f} MB")

    output_directory = "greyScaleMini"

    total_size_bytes = get_directory_size(output_directory)
    total_size_mb = total_size_bytes / (1024 * 1024)
    print(f"Total size of preprocessed data: {total_size_mb:.2f} MB")

    npy_file_sizes = get_npy_file_sizes(output_directory)
    for filename, size_bytes in npy_file_sizes.items():
        print(f"Size of {filename}: {size_bytes} bytes")
        break 

    npy_file_shapes = get_npy_file_shapes(output_directory)
    for filename, shape in npy_file_shapes.items():
        print(f"Shape of {filename}: {shape}")
        break
