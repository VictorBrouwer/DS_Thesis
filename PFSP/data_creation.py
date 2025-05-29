import os

def create_files(data, num_files, output_dir, file_prefix="output"):
    """
    Creates a specified number of files from the given data in the specified directory.

    Args:
        data: A string containing the data to be written to files.
        num_files: The number of files to be created.
        output_dir: The directory where the files will be created.
        file_prefix: The prefix for the output files.

    Returns:
        None
    """
    if not data:
        raise ValueError("Data is empty. Cannot create files.")
    if num_files <= 0:
        raise ValueError("Number of files must be greater than zero.")

    os.makedirs(output_dir, exist_ok=True)

    # Split the data into chunks
    chunk_size = (len(data) + num_files - 1) // num_files  # Ceiling division
    data_chunks = [data[i:i + chunk_size] for i in range(0, len(data), chunk_size)]

    for i, chunk in enumerate(data_chunks):
        file_name = f"{file_prefix}_{i+1:02d}.txt"
        file_path = os.path.join(output_dir, file_name)
        try:
            with open(file_path, "w") as f:
                f.write(chunk)
        except IOError as e:
            print(f"Failed to write to {file_path}: {e}")

# Read data from the input file
input_filename = "data.txt"
try:
    with open(input_filename, "r") as f:
        data = f.read()
except FileNotFoundError:
    print(f"Input file '{input_filename}' not found.")
    exit(1)

# Specify the output directory and number of files
output_dir = "data"
num_files = 10  # Adjust as needed

# Create files
create_files(data, num_files, output_dir, file_prefix="tai50_20")
