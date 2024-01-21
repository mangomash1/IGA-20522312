import pandas as pd
import numpy as np
from datetime import datetime, timedelta
import random
from deap import base, tools, creator, algorithms
import matplotlib.pyplot as plt
import matplotlib.dates as mdates

# Load data from CSV
df = pd.read_csv('GAS3 copy.csv')
df.set_index("TaskID", inplace=True)

# Exclude TaskID 28 from the dataset
df = df.drop(index=28, errors='ignore')

# Declare variables
start_date = datetime(2023, 8, 3)
manual_chromosome = [12, 21, 5, 26, 2, 3, 9, 25, 1, 0, 10, 16, 6, 23, 13, 7, 17, 4, 19, 14, 15, 8, 18, 20, 11, 24, 27, 22]

# Define genetic algorithm parameters
pop_size = 100
num_generations = 80
cx_prob = 0.7
mut_prob = 0.3

# Create a fitness class for minimizing lateness cost
creator.create("FitnessMin", base.Fitness, weights=(-1.0,))
creator.create("Individual", list, fitness=creator.FitnessMin)

# Initialize genetic algorithm operators
toolbox = base.Toolbox()
toolbox.register("indices", random.sample, range(len(df)), len(df))
toolbox.register("individual", tools.initIterate, creator.Individual, toolbox.indices)
toolbox.register("population", tools.initRepeat, list, toolbox.individual)
toolbox.register("mate", tools.cxOrdered)
toolbox.register("mutate", tools.mutShuffleIndexes, indpb=0.05)
toolbox.register("select", tools.selTournament, tournsize=3)

# Function to manually add chromosome
def add_manual_chromosome(chromosome, population):
    population.append(creator.Individual(chromosome))

def calculate_start_date(ind, df):
    machine_finish_dates = {}  # Dictionary to store the finish date for each MachineID
    product_takt_start_dates = {}  # Dictionary to store the start date for each ProductID and Takt
    product_takt_sequence = {}  # Dictionary to store the sequence for each ProductID and Takt

    task_dates = [start_date]  # Initialize task_dates with the project start date

    for i in range(1, len(ind)):
        machine_id = df['MachineID'][ind[i]]
        takt = df['Takt'][ind[i]]
        product_id = df['ProductID'][ind[i]]

        # Rule 1: If tasks have the same MachineID, they should be sequenced and cannot run at the same time
        if machine_id in machine_finish_dates:
            task_start = machine_finish_dates[machine_id]
        else:
            task_start = start_date

        # Rule 2: For each task in a group with the same ProductID and Takt, they should have the same start date
        if (product_id, takt) in product_takt_start_dates:
            task_start = max(task_start, product_takt_start_dates[(product_id, takt)])

        # Rule 3: For each task in a group with the same ProductID, all tasks in Takt 3 and 4 should have the same start date
        if takt in [3, 4] and product_id in [pt[0] for pt in product_takt_start_dates.keys() if pt[1] in [3, 4]]:
            task_start = max(task_start, max(product_takt_start_dates.get((product_id, t), start_date) for t in [3, 4]))

        # Rule 4: Tasks within the same ProductID are ordered based on their Takt
        if (product_id, takt-1) in product_takt_sequence:
            task_start = max(task_start, product_takt_sequence[(product_id, takt-1)])

        # Update the finish date for the current machine and the combination of ProductID and Takt
        finish_date = task_start + timedelta(days=int(df['ProcessingTime'][ind[i]]))
        machine_finish_dates[machine_id] = finish_date
        product_takt_start_dates[(product_id, takt)] = task_start
        product_takt_sequence[(product_id, takt)] = finish_date

        # Update the start date for the current task
        task_dates.append(task_start)

    return task_dates


def calculate_fitness(ind, df):
    task_dates = calculate_start_date(ind, df)  # Calculate the start dates for all tasks

    total_cost = 0
    for i in range(1, len(ind)):
        # Calculate the scheduled due date
        scheduled_due_date = task_dates[i] + timedelta(days=int(df['DueDate'][ind[i]]))
        
        # Calculate the finish date
        finish_date = task_dates[i] + timedelta(days=int(df['ProcessingTime'][ind[i]]))
        
        # Calculate the lateness
        lateness = abs((finish_date - scheduled_due_date).days)
        
        # Add the cost of lateness for this task to the total cost
        total_cost += df['CostOfLateness'][ind[i]] * lateness

    return total_cost,

# Register the new fitness function
toolbox.register("evaluate", calculate_fitness, df=df)

# Function to run the genetic algorithm
def run_genetic_algorithm():
    # Generate a random population
    population = toolbox.population(n=pop_size)

    # Add manual chromosome to the population
    add_manual_chromosome(manual_chromosome, population)

    # Track all generations and their fitness values
    all_generations = []
    all_fitnesses = []

    # Evaluate the entire initial population
    fitnesses = list(map(toolbox.evaluate, population))
    for ind, fit in zip(population, fitnesses):
        ind.fitness.values = fit

    # Save initial generation
    all_generations.append(population[:])
    all_fitnesses.append(fitnesses[:])

    # Run the genetic algorithm
    for gen in range(num_generations):
        offspring = algorithms.varAnd(population, toolbox, cxpb=cx_prob, mutpb=mut_prob)
        
        # Evaluate offspring
        fitnesses = list(map(toolbox.evaluate, offspring))
        for ind, fit in zip(offspring, fitnesses):
            ind.fitness.values = fit

        # Select the next generation
        population[:] = tools.selBest(population + offspring, k=pop_size)

        # Save current generation
        all_generations.append(population[:])
        all_fitnesses.append(fitnesses[:])

    # Extract the best individual from the final generation
    best_individual = tools.selBest(population, k=1)[0]
    print("Best Individual:", best_individual)
    print("Total Cost of Lateness:", best_individual.fitness.values[0])

    # Calculate the start dates for all tasks in the best individual
    task_dates = calculate_start_date(best_individual, df)

    # Create a DataFrame with the task details for the best individual
    df_result = pd.DataFrame({
        'Task': [f'Task {i}' for i in best_individual if i in df.index],
        'Start': [task_dates[i] for i in best_individual if i in df.index],
        'Finish': [task_dates[i] + timedelta(days=int(df['ProcessingTime'][i])) for i in best_individual if i in df.index],
        'MachineID': [df['MachineID'][i] for i in best_individual if i in df.index],
        'ProductID': [df['ProductID'][i] for i in best_individual if i in df.index]
    })

    # Save the DataFrame to a CSV file
    df_result.to_csv('igasch2.csv', index=False)
    # Load the result data from CSV
    df_result = pd.read_csv('igasch2.csv')
    # Convert the 'Start' and 'Finish' columns to datetime
    df_result['Start'] = pd.to_datetime(df_result['Start'])
    df_result['Finish'] = pd.to_datetime(df_result['Finish'])

    # Create a figure and a set of subplots
    fig, ax = plt.subplots()

    # Define the color map for different ProductIDs
    color_map = {1: 'tab:blue', 2: 'tab:orange'}  # Add more colors if there are more than 2 ProductIDs

    # Loop over the tasks
    for i, task in df_result.iterrows():
        # Create a bar for the task duration with transparency
        ax.broken_barh([(mdates.date2num(task['Start']), 
                         mdates.date2num(task['Finish']) - mdates.date2num(task['Start']))], 
                         (task['MachineID'], 1), facecolors = color_map[task['ProductID']], alpha=0.5)

    # Set the x-axis to display dates
    ax.xaxis_date()

    # Set the x-axis ticks and labels to display every day without the year
    ax.xaxis.set_major_locator(mdates.DayLocator(interval=1))  # Change the interval if needed
    ax.xaxis.set_major_formatter(mdates.DateFormatter('%m-%d'))  # Only display month and day
    plt.xticks(rotation='vertical', fontsize='small')

    # Set the y-axis ticks and labels to display every machine ID
    ax.yaxis.set_ticks(np.arange(1, 15))  # Change the range according to the number of machines
    ax.yaxis.set_ticklabels([f'Machine {i}' for i in range(1, 15)])

    # Set the labels for the x-axis and y-axis
    ax.set_xlabel('Date')
    ax.set_ylabel('Machine ID')

    # Set the title of the plot
    ax.set_title('Gantt Chart of Best Individual')

    # Add a legend for the ProductIDs
    handles = [plt.Rectangle((0,0),1,1, color=color_map[i], alpha=0.5) for i in color_map]
    labels = [f'Product {i}' for i in color_map]
    plt.legend(handles, labels)

    # Save the plot to a PNG file
    plt.savefig('gantt_chart.png')

    # Save all generations and fitness scores to a CSV file
    data_to_save = []
    for gen, fitness_values in enumerate(all_fitnesses):
        for ind, fitness in zip(all_generations[gen], fitness_values):
            data_to_save.append([gen, ind, fitness[0]])

    df_generations = pd.DataFrame(data_to_save, columns=['Generation', 'Individual', 'Fitness'])
    generations_csv_path = 'generation_data.csv'
    df_generations.to_csv(generations_csv_path, index=False)
    print(f'Generations data saved to {generations_csv_path}')

    def plot_fitness(all_fitnesses):
        # Extract the best fitness of each generation
        best_fitnesses = [min(fitnesses) for fitnesses in all_fitnesses]

        # Create a figure and a set of subplots
        fig, ax = plt.subplots()

        # Plot the best fitness of each generation
        ax.plot(best_fitnesses, marker='o')

        # Set the labels for the x-axis and y-axis
        ax.set_xlabel('Generation')
        ax.set_ylabel('Best Fitness')

        # Set the title of the plot
        ax.set_title('Best Fitness per Generation')

        # Save the plot to a PNG file
        plt.savefig('fitness_graph.png')

    # Call the function to plot the fitness
    plot_fitness(all_fitnesses)

# Run the genetic algorithm
run_genetic_algorithm()
