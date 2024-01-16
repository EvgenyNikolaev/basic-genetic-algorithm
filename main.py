import random
import matplotlib.pyplot as plt
import numpy as np


def genome_to_real(x, genome_size, precision):
    return (x - ((2 ** genome_size) / 2)) / (1 / precision)


def fitness_func(x, solution_search_boundaries_predicate):
    if not solution_search_boundaries_predicate(x):
        return -np.inf
    return -x * np.sin(1 / x)


def create_individual(genome_size):
    return random.getrandbits(genome_size)


def initialize_population(population_size, genome_size):
    return [create_individual(genome_size) for _ in range(population_size)]


def evaluate_fitness(individual, genome_size, precision, solution_search_boundaries_predicate):
    return fitness_func(genome_to_real(individual, genome_size, precision), solution_search_boundaries_predicate)


def select_individuals(population, tournament_size, genome_size, precision, solution_search_boundaries_predicate):
    tournament = random.sample(population, tournament_size)
    return max(tournament,
               key=lambda x: evaluate_fitness(x, genome_size, precision, solution_search_boundaries_predicate))


def crossover(parent1, parent2, genome_size):
    crossover_point = random.randint(1, genome_size - 1)
    mask1 = 2 ** genome_size - 1 << crossover_point
    mask2 = 2 ** genome_size - 1 >> genome_size - crossover_point
    offspring1 = (parent1 & mask1) | (parent2 & mask2)
    offspring2 = (parent2 & mask1) | (parent1 & mask2)
    return offspring1, offspring2


def mutate(individual, genome_size, mutation_probability):
    for bit in range(genome_size):
        if random.random() <= mutation_probability:
            individual ^= 1 << bit
    return individual


def get_selection(population, tournament_size, genome_size, precision, solution_search_boundaries_predicate):
    selected_population = []
    for _ in range(len(population)):
        selected_individual = select_individuals(population, tournament_size, genome_size, precision,
                                                 solution_search_boundaries_predicate)
        selected_population.append(selected_individual)
    return selected_population


def get_parents(population, parenthood_probability):
    return list(filter(lambda genome: random.random() < parenthood_probability, population))


def get_offsprings(population, genome_size):
    offsprings = []
    for i in range(len(population)):
        if i + 1 == len(population):
            break
        offspring1, offspring2 = crossover(population[i], population[i + 1], genome_size)
        offsprings.extend([offspring1, offspring2])
    return offsprings


def get_mutated(population, genome_size, mutation_probability):
    for i in range(len(population)):
        population[i] = mutate(population[i], genome_size, mutation_probability)
    return population


def get_strongest(population, population_size, genome_size, precision, solution_search_boundaries_predicate):
    return sorted(population, key=lambda x: evaluate_fitness(
        x, genome_size, precision, solution_search_boundaries_predicate), reverse=True)[:population_size]


def get_best_individual(population, genome_size, precision, solution_search_boundaries_predicate):
    return max(population,
               key=lambda x: evaluate_fitness(x, genome_size, precision, solution_search_boundaries_predicate))


def get_average_fitness(population, genome_size, precision, solution_search_boundaries_predicate):
    return sum(map(lambda x: evaluate_fitness(x, genome_size, precision, solution_search_boundaries_predicate),
                   population)) / len(population)


def plot_mutation_probability_and_generations_count_correlation_graph(generations_count_history,
                                                                      mutation_probabilities):
    fig, ax = plt.subplots()
    ax.plot(mutation_probabilities, generations_count_history)

    ax.set(xlabel='Mutation probability', ylabel='Number of generations until convergence')
    ax.grid()

    plt.show()


def plot_mutation_probability_and_best_fitness_correlation_graph(best_individual_fitness,
                                                                 mutation_probabilities):
    fig, ax = plt.subplots()
    ax.plot(mutation_probabilities, best_individual_fitness)

    ax.set(xlabel='Mutation probability', ylabel='Best individual fitness')
    ax.grid()

    plt.show()


def print_generation_stats(generation,
                           best_individual,
                           best_individual_fitness,
                           average_fitness,
                           genome_size,
                           precision,
                           mutation_probability,
                           std_deviation):
    print(
        f"Generation {generation}: "
        f"Best Individual = {genome_to_real(best_individual, genome_size, precision)}, "
        f"Fitness = {best_individual_fitness}, "
        f"Mutation probability = {mutation_probability}, "
        f"Standard deviation = {std_deviation}, "
        f"Average = {average_fitness}")


def check_termination_criteria(average_history, std_deviation_history, convergence_threshold):
    if len(average_history) >= 10:
        std_deviation = np.std(average_history[-10:])
        std_deviation_history.append(std_deviation)
        if std_deviation < convergence_threshold:
            print(f"Converged: Standard Deviation = {std_deviation}")
            return True, std_deviation
        else:
            return False, std_deviation

    return False, 0


def run_genetic_algorithm(population_size,
                          tournament_size,
                          mutation_probability,
                          parenthood_probability,
                          precision,
                          convergence_threshold,
                          genome_size,
                          solution_search_boundaries_predicate):
    population = initialize_population(population_size, genome_size)

    best_individuals_history = []
    average_history = []
    std_deviation_history = []

    generation = 0

    while True:
        population = get_selection(population, tournament_size, genome_size, precision,
                                   solution_search_boundaries_predicate)
        population = get_parents(population, parenthood_probability)
        population.extend(get_offsprings(population, genome_size))
        population = get_mutated(population, genome_size, mutation_probability)
        population = get_strongest(population, population_size, genome_size, precision,
                                   solution_search_boundaries_predicate)

        best_individual = get_best_individual(population, genome_size, precision, solution_search_boundaries_predicate)
        best_individual_fitness = evaluate_fitness(best_individual, genome_size, precision,
                                                   solution_search_boundaries_predicate)
        average_fitness = get_average_fitness(population, genome_size, precision, solution_search_boundaries_predicate)
        average_history.append(average_fitness)
        best_individuals_history.append(genome_to_real(best_individual, genome_size, precision))

        generation += 1
        terminate, std_deviation = check_termination_criteria(average_history, std_deviation_history,
                                                              convergence_threshold)
        print_generation_stats(generation,
                               best_individual,
                               best_individual_fitness,
                               average_fitness,
                               genome_size,
                               precision,
                               mutation_probability,
                               std_deviation)

        if terminate:
            break

    return generation, genome_to_real(best_individual, genome_size, precision), best_individual_fitness


if __name__ == "__main__":

    mutation_probabilities = np.arange(0.01, 0.31, 0.01)
    results = []
    for mp in mutation_probabilities:
        results.append(run_genetic_algorithm(
            population_size=1000,
            tournament_size=3,
            mutation_probability=mp,
            parenthood_probability=0.7,
            precision=0.001,
            convergence_threshold=0.03,
            genome_size=14,
            solution_search_boundaries_predicate=lambda x: (-5 <= x < 0) or (0 < x <= 5)
        ))

    plot_mutation_probability_and_generations_count_correlation_graph(list(map(lambda r: r[0], results)),
                                                                      mutation_probabilities)

    plot_mutation_probability_and_best_fitness_correlation_graph(list(map(lambda r: r[2], results)),
                                                                 mutation_probabilities)
    for index, result in enumerate(results):
        print(f"Generation: {index}, generations: {result[0]}, best individual: {result[1]}, fitness: {result[2]}")
