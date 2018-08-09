import numpy as np
import matplotlib.pyplot as plt

def de(fobj, bounds, mut=0.8, crossp=0.7, popsize=20, its=3000):
    dimensions = len(bounds)
    pop = np.random.rand(popsize, dimensions)
    min_b, max_b = np.asarray(bounds).T
    diff = np.fabs(min_b - max_b)
    pop_denorm = min_b + pop * diff
    fitness = np.asarray([fobj(ind) for ind in pop_denorm])
    best_idx = np.argmin(fitness)
    best = pop_denorm[best_idx]
    for i in range(its):
        for j in range(popsize):
            idxs = [idx for idx in range(popsize) if idx != j]
            a, b, c = pop[np.random.choice(idxs, 3, replace = False)]
            mutant = np.clip(a + mut * (b - c), 0, 1)
            cross_points = np.random.rand(dimensions) < crossp
            if not np.any(cross_points):
                cross_points[np.random.randint(0, dimensions)] = True
            trial = np.where(cross_points, mutant, pop[j])
            trial_denorm = min_b + trial * diff
            f = fobj(trial_denorm)
            if f < fitness[j]:
                fitness[j] = f
                pop[j] = trial
                if f < fitness[best_idx]:
                    best_idx = j
                    best = trial_denorm
        yield best, fitness[best_idx]

def fobj(x):
    value=0
    for i in ramge (len(x)):
        value+=x[i]**2
        return value/len(x)

for d in [8, 16 ,32 , 64]:
 it = list(de(lambda x: sum(x**2)/d, bounds=[(-100, 100)] * d))
 x,f =zip(*it)
 plt.xlabel('ITERATION')
 plt.ylabel('FITNESS')
 plt.title('Evolution of fitness on 3000 iterations(different dimensions)')
 plt.plot(f, label='d={}'.format(d))
plt.legend()
plt.show()
