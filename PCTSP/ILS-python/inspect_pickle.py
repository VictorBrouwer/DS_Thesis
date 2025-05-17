#!/usr/bin/env python3
import pickle
import sys
import os

def inspect_pickle(pickle_file):
    """Inspect the structure of a pickle file"""
    with open(pickle_file, 'rb') as f:
        data = pickle.load(f)
    
    print(f"Type of data: {type(data)}")
    
    if isinstance(data, list):
        print(f"Number of instances: {len(data)}")
        
        if len(data) > 0:
            print("\nFirst instance structure:")
            first_instance = data[0]
            print(f"Type of instance: {type(first_instance)}")
            
            if isinstance(first_instance, dict):
                print("Keys in the instance:")
                for key in first_instance.keys():
                    value = first_instance[key]
                    print(f"  - {key}: {type(value)}")
                    if hasattr(value, 'shape'):
                        print(f"    Shape: {value.shape}")
                    elif isinstance(value, (list, tuple)):
                        print(f"    Length: {len(value)}")
            else:
                print(f"First instance content: {first_instance}")
    else:
        print("Data structure:")
        print(data)

def main():
    if len(sys.argv) < 2:
        print("Usage: python inspect_pickle.py <pickle_file>")
        print("Example: python inspect_pickle.py ../pctsp_data/pctsp_20_20_instances.pkl")
        sys.exit(1)
    
    pickle_file = sys.argv[1]
    
    # Adjust path if needed for ILS-python folder
    if not os.path.exists(pickle_file) and pickle_file.startswith("pctsp_data/"):
        parent_dir_pickle = os.path.join("..", pickle_file)
        if os.path.exists(parent_dir_pickle):
            pickle_file = parent_dir_pickle
    
    if not os.path.exists(pickle_file):
        print(f"Error: File {pickle_file} not found")
        print("Available pickle files:")
        parent_data_dir = os.path.join("..", "pctsp_data")
        if os.path.exists(parent_data_dir):
            for f in os.listdir(parent_data_dir):
                if f.endswith(".pkl"):
                    print(f"  - ../pctsp_data/{f}")
        sys.exit(1)
    
    inspect_pickle(pickle_file)

if __name__ == "__main__":
    main() 