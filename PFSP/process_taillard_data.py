import os
import re

def parse_taillard_instances(data_content):
    """
    Parses Taillard benchmark instances from a string, preserving the original format.
    In Taillard format, each instance has machine_count lines of data (not job_count).

    Args:
        data_content: A string containing the Taillard benchmark data.

    Returns:
        A list of tuples, where each tuple contains (num_jobs, num_machines, complete_instance_text).
    """
    instances = []
    lines = data_content.strip().split('\n')
    line_idx = 0
    num_lines = len(lines)
    
    while line_idx < num_lines:
        # Look for the descriptor line
        if line_idx < num_lines and "number of jobs" in lines[line_idx].lower():
            descriptor_line = lines[line_idx]
            line_idx += 1
            
            # Next line should be the header with job/machine counts
            if line_idx < num_lines:
                header_line = lines[line_idx].strip()
                header_match = re.match(r'^\s*(\d+)\s+(\d+)', header_line)
                
                if header_match:
                    num_jobs = int(header_match.group(1))
                    num_machines = int(header_match.group(2))
                    line_idx += 1
                    
                    # Next should be "processing times :"
                    processing_times_line = ""
                    if line_idx < num_lines and "processing times" in lines[line_idx].lower():
                        processing_times_line = lines[line_idx]
                        line_idx += 1
                    
                    # Now read the job data lines - IMPORTANT: We expect num_machines lines, not num_jobs
                    # Each line represents one machine's processing times for all jobs
                    job_lines = []
                    lines_read = 0
                    
                    while line_idx < num_lines and lines_read < num_machines:
                        job_line = lines[line_idx]
                        # Check if this looks like a data line (numbers)
                        if re.search(r'\d+\s+\d+', job_line):
                            job_lines.append(job_line)
                            lines_read += 1
                            line_idx += 1
                        else:
                            # Not a data line, might be the start of next instance
                            break
                    
                    # If we got all the expected lines, create the instance
                    if lines_read == num_machines:
                        # Construct the complete instance text
                        instance_text = descriptor_line + '\n' + header_line + '\n' + processing_times_line + '\n'
                        instance_text += '\n'.join(job_lines)
                        
                        instances.append((num_jobs, num_machines, instance_text))
                    else:
                        print(f"Warning: Incomplete instance data. Expected {num_machines} machine lines, found {lines_read}.")
                        # Skip to next potential instance
                else:
                    # Not a valid header line
                    line_idx += 1
            else:
                # Reached end of file
                break
        else:
            # Not a descriptor line, move to next line
            line_idx += 1
    
    return instances

def save_instances(instances, base_output_dir):
    """
    Saves parsed instances into size-specific folders.

    Args:
        instances: A list of tuples, where each tuple contains (num_jobs, num_machines, instance_text).
        base_output_dir: The base directory to save the instance folders.
    """
    instance_counters = {}
    for num_jobs, num_machines, instance_text in instances:
        folder_name = f"j{num_jobs}_m{num_machines}"
        output_subdir = os.path.join(base_output_dir, folder_name)
        os.makedirs(output_subdir, exist_ok=True)

        # Increment counter for this instance size
        instance_key = (num_jobs, num_machines)
        instance_counters[instance_key] = instance_counters.get(instance_key, 0) + 1
        instance_num = instance_counters[instance_key]

        file_name = f"j{num_jobs}_m{num_machines}_{instance_num:02d}.txt"
        file_path = os.path.join(output_subdir, file_name)

        try:
            with open(file_path, "w") as f:
                f.write(instance_text)
            # print(f"Saved instance to {file_path}")
        except IOError as e:
            print(f"Failed to write to {file_path}: {e}")

# --- Main Execution ---
input_filename = os.path.join("data", "data.txt")
base_output_dir = "data"

try:
    with open(input_filename, "r") as f:
        data_content = f.read()
except FileNotFoundError:
    print(f"Error: Input file '{input_filename}' not found.")
    exit(1)
except IOError as e:
    print(f"Error reading input file '{input_filename}': {e}")
    exit(1)

if not data_content.strip():
    print("Error: Input file is empty.")
    exit(1)

parsed_instances = parse_taillard_instances(data_content)

if not parsed_instances:
    print("No valid Taillard instances found or parsed correctly in the input file.")
    print("Please ensure the input file contains instances in the Taillard benchmark format.")
    exit(1)

save_instances(parsed_instances, base_output_dir)

print(f"Successfully processed {len(parsed_instances)} instances.")
print(f"Organized instances into subdirectories within '{base_output_dir}'.")

# Output a summary of instance counts per folder
instance_count_summary = {}
for num_jobs, num_machines, _ in parsed_instances:
    folder_name = f"j{num_jobs}_m{num_machines}"
    instance_count_summary[folder_name] = instance_count_summary.get(folder_name, 0) + 1

print("\nInstances per folder:")
for folder, count in sorted(instance_count_summary.items()):
    print(f"- {folder}: {count} instances") 