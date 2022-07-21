import numpy as np

from deap import base, algorithms
from deap import tools
from deap import creator

from config import *


class Func_max:
    def __init__(self):
        self.toolbox = self.ga_setup()
        self.stats = self.get_statistics()
        self.population = self.make_population()
        
    def make_population(self):
        """
        Создаем популяцию
        """
        creator.create('FitnessMax', base.Fitness, weights=(1.0,))
        creator.create('Individual', list, fitness=creator.FitnessMax)
        
        x = np.arange(-np.pi, np.pi, 0.1) 
        y = np.arange(-4 * np.pi, 4 * np.pi, 0.4)
        
        self.population = [creator.Individual([x, y]) for x, y in zip(x, y)]
        return self.population
    
    def one_fitness(self, individual):
        """
        Приспособленность одной хромосомы
        """
        x = individual[0]
        y = individual[1]
        return np.sin(x) * 100 / (x ** 2 + 2 * y ** 2 + 50), 
    
    def ga_setup(self):
        self.toolbox = base.Toolbox()
        self.toolbox.register('select', tools.selTournament, tournsize=3)
        self.toolbox.register('mate', tools.cxSimulatedBinary, eta=0.1)
        self.toolbox.register('mutate', tools.mutShuffleIndexes, indpb=0.1)
        # evaluate возвращает кортеж значений приспособленности отдельного индивида
        self.toolbox.register('evaluate', self.one_fitness)
        
        return self.toolbox
    
    def get_statistics(self):
        self.stats = tools.Statistics(lambda ind: ind.fitness.values)
        self.stats.register('max', np.max)
        self.stats.register('avg', np.mean)
        return self.stats
    
    def run(self):
        self.population, logbook = algorithms.eaSimple(self.population, self.toolbox, cxpb=P_CROSSOVER,
                                                       mutpb=P_MUTATION, stats=self.stats,
                                                       ngen=NUM_ITERATIONS, verbose=False)
        return logbook

  
    
if __name__ == '__main__':
    ga = Func_max()
    print(ga.run())
