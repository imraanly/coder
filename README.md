# coder
import random
from deap import base, creator, tools, algorithms

# Step 1: Define the Problem (TSP)
# Create fitness class: we want to minimize the route distance
creator.create("FitnessMin", base.Fitness, weights=(-1.0,))  # Minimize distance
creator.create("Individual", list, fitness=creator.FitnessMin)

# Distance matrix for 6 cities
distance_matrix = [
    [0, 10, 15, 20, 25, 30],
    [10, 0, 35, 25, 17, 28],
    [15, 35, 0, 30, 20, 22],
    [20, 25, 30, 0, 18, 26],
    [25, 17, 20, 18, 0, 16],
    [30, 28, 22, 26, 16, 0]
]

# Step 2: Define the Evaluation Function
def evaluate(individual):
    total_distance = 0
    for i in range(len(individual) - 1):
        total_distance += distance_matrix[individual[i]][individual[i + 1]]
    total_distance += distance_matrix[individual[-1]][individual[0]]  # Return to start
    return total_distance,

# Step 3: Define the Toolbox and Genetic Algorithm Components
toolbox = base.Toolbox()

# Generate individual (permutation of cities)
toolbox.register("indices", random.sample, range(len(distance_matrix)), len(distance_matrix))
toolbox.register("individual", tools.initIterate, creator.Individual, toolbox.indices)

# Create population
toolbox.register("population", tools.initRepeat, list, toolbox.individual)

# Define crossover (Order Crossover)
toolbox.register("mate", tools.cxOrdered)

# Define mutation (Swap mutation)
toolbox.register("mutate", tools.mutShuffleIndexes, indpb=0.2)

# Define selection (Tournament Selection)
toolbox.register("select", tools.selTournament, tournsize=3)

# Step 4: Set up the Genetic Algorithm
def main():
    # Initialize population
    population = toolbox.population(n=100)
    
    # Evaluate the population
    for ind in population:
        ind.fitness.values = evaluate(ind)

    # Set parameters for the algorithm
    generations = 500
    cxpb = 0.7  # Crossover probability
    mutpb = 0.2  # Mutation probability

    # Run the Genetic Algorithm
    for gen in range(generations):
        print(f"Generation {gen}")
        
        # Select the next generation
        offspring = toolbox.select(population, len(population))
        offspring = list(map(toolbox.clone, offspring))

        # Apply crossover and mutation
        for child1, child2 in zip(offspring[::2], offspring[1::2]):
            if random.random() < cxpb:
                toolbox.mate(child1, child2)
                del child1.fitness.values
                del child2.fitness.values

        for mutant in offspring:
            if random.random() < mutpb:
                toolbox.mutate(mutant)
                del mutant.fitness.values

        # Evaluate the offspring
        for ind in offspring:
            if not ind.fitness.valid:
                ind.fitness.values = evaluate(ind)

        # Replace the old population with the new one
        population[:] = offspring

    # Return the best solution
    best_individual = tools.selBest(population, 1)[0]
    print(f"Best route: {best_individual}")
    print(f"Total distance: {best_individual.fitness.values[0]}")

if __name__ == "__main__":
    main()
