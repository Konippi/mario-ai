import numpy as np
from ga.individual import Individual

class CustomGA:
    @staticmethod
    def adaptive_tournament_size(
        current_generation, 
        n_generations, 
        n_individuals,
        max_tournament_ratio=0.3
    ):
        min_tournament_size, max_tournament_size = 1, int(n_individuals * max_tournament_ratio)
        return int(min_tournament_size + (max_tournament_size - min_tournament_size) * (1 - current_generation / n_generations))
    
    @staticmethod
    def adaptive_crossover_rate(
        current_generation,
        n_generations,
        init_crossover_rate=0.8, # 最初は多様性を確保したいため、比較的大きい値にする
        final_crossover_rate=0.4,
    ):
        return init_crossover_rate - (init_crossover_rate - final_crossover_rate) * (current_generation / n_generations)
    
    @staticmethod
    def adaptive_mutation_rate(
        current_generation,
        n_generations,
        init_mutation_rate=0.1, # 最初は多様性を確保したいため、比較的大きい値にする
        final_mutatrion_rate=0.001,
    ):
        return init_mutation_rate - (init_mutation_rate - final_mutatrion_rate) * (current_generation / n_generations)

class Controller:
    @classmethod
    def select(cls, individuals, probabilities, current_generation, n_generations):
        """
        selectにおける変更点
        -------------------
        - トーナメント選択を導入
            - 適応型選択圧を導入する
            - 選択圧をコントロールし、多様性を維持する
        
        selectにおける考慮点
        ------
        - learn.py(l32)で実装されているエリート選択により、最も優秀な1個体は既に次世代に受け継いでいる
            - 多様性がない状態 -> 急激な性能低下を防ぐ
        """
        adaptive_tournament_size = CustomGA.adaptive_tournament_size(current_generation, n_generations, len(individuals))
        def _select_tournament_winner():
            tournament_indices = np.random.choice(len(individuals), size=adaptive_tournament_size, replace=False)
            winner_index = max(tournament_indices, key=lambda i: probabilities[i])
            return individuals[winner_index]
        
        father = _select_tournament_winner()
        mother = _select_tournament_winner()
        
        # 同じ個体が選ばれた場合は再選択
        while mother is father:
            mother = _select_tournament_winner()
        
        return father, mother

    @classmethod
    def crossover(cls, individual1, individual2, current_generation, n_generations, crossover_rate=0.8):
        """
        crossoverにおける変更点
        ---------------------
        - 2点交叉を導入
            - 順序関係を考慮せずに交叉することで多様性を出す
        - 適応型交差率を導入
            - 局所解を回避する -> 多様性を維持する
        """
        crossover_rate = CustomGA.adaptive_crossover_rate(current_generation, n_generations)
        
        if np.random.random() > crossover_rate:
            return individual1, individual2
        
        def _two_point_crossover(individual1, individual2):
            size = len(individual1.data)
            idx0, idx1 = np.sort(np.random.choice(size, size=2, replace=False))

            child1 = individual1.data.copy()
            child2 = individual2.data.copy()

            child1[idx0:idx1], child2[idx0:idx1] = child2[idx0:idx1], child1[idx0:idx1]

            return Individual(child1), Individual(child2)
        
        def _pmx(individual1, individual2):
            size = len(individual1.data)
            idx0, idx1 = np.sort(np.random.choice(size, size=2, replace=False))

            child1 = individual1.data.copy()
            child2 = individual2.data.copy()

            child1_indices = np.zeros(size, dtype=int)
            child2_indices = np.zeros(size, dtype=int)

            for idx in range(size):
                child1_indices[child1[idx]] = idx
                child2_indices[child2[idx]] = idx

            for idx in range(idx0, idx1):
                val1, val2 = child1[idx], child2[idx]
                
                child1[idx], child1[child1_indices[val2]] = val2, val1
                child1_indices[val1], child1_indices[val2] = child1_indices[val2], idx

                child2[idx], child2[child2_indices[val1]] = val1, val2
                child2_indices[val2], child2_indices[val1] = child2_indices[val1], idx
                
            return Individual(child1), Individual(child2)
        
        # child1, child2 = _pmx(individual1, individual2)
        child1, child2 = _two_point_crossover(individual1, individual2)

        return child1, child2

    @classmethod
    def mutate(cls, individuals, current_generation, n_generations):
        """
        mutateにおける変更点
        ------------------
        - 適応型突然変異率を導入
            - 進化の進行に応じて「探索 < 搾取」になるよう調整する
        """
        mutation_rate = CustomGA.adaptive_mutation_rate(current_generation, n_generations)
        n_mutation = int(np.ceil(len(individuals) * mutation_rate))
        individuals_to_mutate = np.random.choice(individuals, size=n_mutation, replace=False)
        for individual in individuals_to_mutate:
            for _ in range(int(len(individual.data) * mutation_rate)):
                index1 = np.random.randint(0, len(individual.data) - 1)
                index2 = np.random.randint(index1, len(individual.data))
                buf1 = individual.data[index1]
                buf2 = individual.data[index2]
                individual.data[index1] = buf2
                individual.data[index2] = buf1
