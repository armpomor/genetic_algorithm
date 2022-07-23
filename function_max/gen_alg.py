from itertools import accumulate, pairwise
import numpy as np
import numpy.random as rand

from config import *


class FuncMax:
    """
    Search максимального значения функции z = sin(x) * 100 / (x ** 2 + 2 * y ** 2 + 50)
    """

    def __init__(self):
        # rand.seed(42)
        self.probabilitys = None
        self.x = np.arange(-np.pi, np.pi, 0.1)
        self.y = np.arange(-4 * np.pi, 4 * np.pi, 0.4)
        self.list_fitness = self.fitness(self.x, self.y)
        self.average_values_z = np.array(np.mean(self.list_fitness))

    @staticmethod
    def fitness(x, y):
        """
        Функция приспособленности.
        """
        return np.sin(x) * 100 / (x ** 2 + 2 * y ** 2 + 50)

    def probability_survival(self):
        """
        Список вероятностей стать родителем.
        """
        # Заменяем отрицательные значения целевой функции нулями
        no_negative_values = self.list_fitness
        no_negative_values[no_negative_values < 0] = 0

        self.probabilitys = 100 * no_negative_values / sum(no_negative_values)
        return self.probabilitys

    def proportionale_selection(self):
        """
        Отбор родителей по правилу рулетки
        """
        # Создаем диапазоны для рулетки
        p = accumulate(self.probabilitys, initial=0)
        ranges = [range(int(i[0]), int(i[1]) + 1) for i in pairwise(p)]

        # Крутим рулетку и собираем список родителей
        parents_x = np.array([])
        parents_y = np.array([])

        for i in range(len(self.x)):
            r = rand.randint(0, 99)
            index = [r in i for i in ranges].index(True)
            parents_x = np.append(parents_x, self.x[index])
            parents_y = np.append(parents_y, self.y[index])
        self.x = parents_x[:]
        self.y = parents_y[:]

    def crossing_over(self):
        """
        Перемешиваем координаты y
        """
        rand.shuffle(self.y)

    def mutation(self):
        population = np.c_[self.x, self.y]

        n = round(len(self.x) * RATIO_MUTATION)  # Количество мутантов

        index_mutants = rand.choice(len(population), size=n)  # Индексы будущих мутантов

        mutants = population[index_mutants]

        # Удаляем будущих мутантов из популяции
        population = np.delete(population, [index_mutants], 0)

        # Производим мутацию
        # mutants[:, 0] += 0.1
        mutants = np.flip(mutants, axis=0)

        # Добавляем мутантов в список популяции
        population = np.concatenate((population, mutants), axis=0)

        self.x = population[:, 0]
        self.y = population[:, 1]

    def run(self):
        for i in range(NUM_ITER):
            self.probability_survival()
            self.proportionale_selection()
            self.crossing_over()
            self.mutation()
            self.list_fitness = self.fitness(self.x, self.y)
            self.average_values_z = np.append(self.average_values_z, np.mean(self.list_fitness))


if __name__ == '__main__':
    ga = FuncMax()
    ga.run()
    print(ga.average_values_z)
