import random
import math
import matplotlib.pyplot as plt


def read_tsp_file(file_path):
    cities = []
    with open(file_path, 'r') as f:
        lines = f.readlines()

        reading_coordinates = False
        for line in lines:
            line = line.strip()
            if line == "NODE_COORD_SECTION":
                reading_coordinates = True
                continue
            if line == "EOF":
                break
            if reading_coordinates and line:
                parts = line.split()
                if len(parts) == 3:
                    city_id = int(parts[0])
                    x = float(parts[1])
                    y = float(parts[2])
                    cities.append((city_id, x, y))

    return cities


def euclidean_distance(city1, city2):
    return math.sqrt((city2[1] - city1[1]) ** 2 + (city2[2] - city1[2]) ** 2)


def greedy_algorithm(cities, start):
    unvisited = cities[:]
    current_city = unvisited.pop(start)
    route = [current_city]

    while unvisited:
        nearest_city = None
        min_distance = float('inf')
        for city in unvisited:
            distance = euclidean_distance(current_city, city)
            if distance < min_distance:
                min_distance = distance
                nearest_city = city
        unvisited.remove(nearest_city)
        route.append(nearest_city)
        current_city = nearest_city

    return route


def random_algorithm(cities):
    unvisited = cities[:]
    random.shuffle(unvisited)
    return unvisited


def calculate_route_length(route):
    length = 0
    for i in range(len(route) - 1):
        length += euclidean_distance(route[i], route[i + 1])
    length += euclidean_distance(route[-1], route[0])
    return length


def swap_mutation(route, mutation_probability=0.2):
    if random.random() < mutation_probability:
        idx1, idx2 = random.sample(range(len(route)), 2)
        route[idx1], route[idx2] = route[idx2], route[idx1]
    return route


def create_new_epoch(population, mutation_probability=0.2):
    new_population = []
    for _ in range(len(population)):
        parent1 = tournament_selection(population)
        parent2 = tournament_selection(population)

        split_point = len(parent1) // 2
        child = parent1[:split_point] + [city for city in parent2 if city not in parent1[:split_point]]

        mutated_child = swap_mutation(child, mutation_probability)

        new_population.append(mutated_child)
    return new_population


def tournament_selection(population, tournament_size=7):
    selected = random.sample(population, tournament_size)
    best = None
    best_score = float('inf')
    for sol in selected:
        curr_score = calculate_route_length(sol)
        if curr_score < best_score:
            best_score = curr_score
            best = sol
    return best


def run_genetic_algorithm(cities, initial_population_size=200, epochs=200, mutation_probability=0.8):
    population = [random_algorithm(cities) for _ in range(initial_population_size)]

    best_solution = None
    best_score = float('inf')
    best_scores_per_epoch = []
    avg_scores_per_epoch = []
    worst_scores_per_epoch = []

    for epoch in range(epochs):
        population = create_new_epoch(population, mutation_probability)

        scores = [calculate_route_length(individual) for individual in population]
        best_epoch_score = min(scores)
        avg_epoch_score = sum(scores) / len(scores)
        worst_epoch_score = max(scores)

        if best_epoch_score < best_score:
            best_score = best_epoch_score
            best_solution = population[scores.index(best_epoch_score)]

        best_scores_per_epoch.append(best_epoch_score)
        avg_scores_per_epoch.append(avg_epoch_score)
        worst_scores_per_epoch.append(worst_epoch_score)

    return best_solution, best_score, best_scores_per_epoch, avg_scores_per_epoch, worst_scores_per_epoch


def main():
    tsp_file = 'berlin11.tsp'
    cities = read_tsp_file(tsp_file)

    best_solution, best_score, best_scores_per_epoch, avg_scores_per_epoch, worst_scores_per_epoch = run_genetic_algorithm(
        cities, initial_population_size=200, epochs=200, mutation_probability=0.3
    )

    print("Best Solution:", [city[0] for city in best_solution])
    print("Best Score:", best_score)

    greedy_route = greedy_algorithm(cities, 2)
    greedy_length = calculate_route_length(greedy_route)
    print("\nGreedy Route:", [city[0] for city in greedy_route])
    print("Greedy Route Length:", greedy_length)

    random_route = random_algorithm(cities)
    random_length = calculate_route_length(random_route)
    print("\nRandom Route:", [city[0] for city in random_route])
    print("Random Route Length:", random_length)

    population = [random_algorithm(cities) for _ in range(200)]
    best_tournament_route = tournament_selection(population)
    best_tournament_length = calculate_route_length(best_tournament_route)
    print("\nBest Route from Tournament Selection:", [city[0] for city in best_tournament_route])
    print("Best Route Length from Tournament Selection:", best_tournament_length)
    print()
    plt.figure(figsize=(16, 8))
    plt.plot(range(1, len(best_scores_per_epoch) + 1), best_scores_per_epoch, marker='o', label='Best Score')
    plt.plot(range(1, len(avg_scores_per_epoch) + 1), avg_scores_per_epoch, marker='x', label='Average Score')
    plt.plot(range(1, len(worst_scores_per_epoch) + 1), worst_scores_per_epoch, marker='s', label='Worst Score')
    plt.title("Table")
    plt.xlabel("Epoch")
    plt.ylabel("Score")
    plt.legend()
    plt.grid(True)
    plt.show()


if __name__ == "__main__":
    main()
