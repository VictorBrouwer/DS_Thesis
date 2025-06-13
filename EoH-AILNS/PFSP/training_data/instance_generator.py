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
        processing_times = [[random.randint(1, 100) for _ in range(m)] for _ in range(n)]
        file_name = f"{prefix}_{instance + 1}.txt"
        with open(file_name, 'w') as file:
            file.write(f"{n} {m}\n")
            for i in range(n):
                for j in range(m):
                    file.write(f"{j} {processing_times[i][j]} ")
                file.write("\n")