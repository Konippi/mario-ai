import numpy
from ga.individual import Individual

class CustomGA:
    def adaptive_mutation_rate(
        current_generation,
        n_generations,
        init_mutation_rate=0.1,
        final_mutatrion_rate=0.01
    ):
        return init_mutation_rate - (init_mutation_rate - final_mutatrion_rate) * (current_generation / n_generations)

class Controller:
    @classmethod
    def select(cls, individuals, probabilities):
        normalized_probabolities = numpy.array(probabilities) / sum(probabilities)
        return numpy.random.choice(individuals, size=2, replace=False, p=normalized_probabolities)

    @classmethod
    def crossover(cls, individual1, individual2, crossover_rate=0.8):
        """
        crossoverにおける変更点
        ---------------------
        - 部分的交差を導入
        - 交差率を導入
        """
        if numpy.random.rand() > crossover_rate:
            return (individual1, individual2)
        
        size = len(individual1)
        child1, child2 = [-1]*size, [-1]*size

        cx_point1 = numpy.random.randint(0, size - 1)
        cx_point2 = numpy.random.randint(0, size - 1)
        
        if cx_point1 > cx_point2:
            cx_point1, cx_point2 = cx_point2, cx_point1

        child1[cx_point1:cx_point2 + 1] = individual1[cx_point1:cx_point2 + 1]
        child2[cx_point1:cx_point2 + 1] = individual2[cx_point1:cx_point2 + 1]

        for i in range(cx_point1, cx_point2 + 1):
            if individual2[i] not in child1:
                value = individual2[i]
                while True:
                    index = individual1.index(value)
                    value = individual2[index]
                    if value not in child1:
                        break
                child1[child1.index(-1)] = value

        for i in range(cx_point1, cx_point2 + 1):
            if individual1[i] not in child2:
                value = individual1[i]
                while True:
                    index = individual2.index(value)
                    value = individual1[index]
                    if value not in child2:
                        break
                child2[child2.index(-1)] = value

        for i in range(size):
            if child1[i] == -1:
                child1[i] = individual2[i]
            if child2[i] == -1:
                child2[i] = individual1[i]
                
        child1 = Individual(data=numpy.array(child1))
        child2 = Individual(data=numpy.array(child2))

        return (child1, child2)

    @classmethod
    def mutate(cls, individuals, current_generation, n_generations):
        """
        mutateにおける変更点
        ------------------
        - Adaptive mutation rateを導入
        - ガウシアン突然変異を導入
        """
        mutation_rate = CustomGA.adaptive_mutation_rate(
            current_generation=current_generation, 
            n_generations=n_generations,
        )
        n_mutation = int(numpy.ceil(len(individuals) * mutation_rate))
        individuals_to_mutate = numpy.random.choice(individuals, size=n_mutation, replace=False)
        
        for individual in individuals_to_mutate:
            noise = numpy.random.normal(0, 1, size=len(individual.data))
            individual.data += numpy.round(noise * mutation_rate).astype(numpy.int64)
            individual.data = numpy.clip(individual.data, 0, 1)
