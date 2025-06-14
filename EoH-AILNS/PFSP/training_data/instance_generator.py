import random

# Define instance configurations
configurations = [
    (20, 10, "j20_m10"),  # (jobs, machines, prefix)
    (50, 20, "j50_m20"),
    (100, 10, "j100_m10")
]

# Generate 2 instances for each configuration
for n, m, prefix in configurations:
    for instance in range(2):
        # Generate random seed
        seed = random.randint(1, 2147483647)
        
        # Generate processing times matrix (m x n)
        processing_times = [[random.randint(1, 99) for _ in range(n)] for _ in range(m)]
        
        # Calculate bounds (simplified version - in real Taillard instances these are optimal bounds)
        lower_bound = sum(min(row) for row in processing_times)
        upper_bound = int(lower_bound * 1.1)  # 10% above lower bound as a simple approximation
        
        file_name = f"{prefix}_{instance + 1}.txt"
        with open(file_name, 'w') as file:
            # Write header
            file.write("number of jobs, number of machines, initial seed, upper bound and lower bound :\n")
            file.write(f"{n:12d} {m:12d} {seed:12d} {upper_bound:12d} {lower_bound:12d}\n")
            
            # Write processing times
            file.write("processing times :\n")
            for i in range(m):
                file.write(" ".join(f"{processing_times[i][j]:3d}" for j in range(n)) + "\n")