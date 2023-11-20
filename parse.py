import struct 
import sys 
import matplotlib.pyplot as plt

assert len(sys.argv) == 3

iterations = int(sys.argv[1]) 
executions = int(sys.argv[2]) 

lf = f'{iterations}L'
df = f'{iterations}d'

def load_data(fname): 
    areas = [] 
    entropies = [] 
    densities = [] 

    f = open(fname, "rb") 
    for ex in range(executions): 
        seed = struct.unpack('i', f.read(4)) 
        areas.append(struct.unpack(lf, f.read(8 * iterations))) 
        entropies.append(struct.unpack(lf, f.read(8 * iterations)))
        densities.append(struct.unpack(df, f.read(8 * iterations)))
    f.close() 
    
    area_diffs = [
        [abs(areas[ex][i+1]-areas[ex][i]) for i in range(iterations-1)]  
        for ex in range(executions)
    ]
    entropy_diffs = [ 
        [abs(entropies[ex][i+1]-entropies[ex][i]) for i in range(iterations-1)]
        for ex in range(executions) 
    ] 
    density_diffs = [
        [abs(densities[ex][i+1]-densities[ex][i]) for i in range(iterations-1)] 
        for ex in range(executions) 
    ] 

    areas_sum = [sum(area_diffs[i]) for i in range(executions)]
    entropies_sum = [sum(entropy_diffs[i]) for i in range(executions)] 
    densities_sum = [sum(density_diffs[i]) for i in range(executions)]
   
    return areas_sum, entropies_sum, densities_sum 

fnames = ['conwaylife', 'dayandnight', 'lifewithoutdeath']
data = [
    load_data(f'{fname}-{iterations}-{executions}.txt') 
    for fname in fnames
]

plt.hist([d[0] for d in data], label=fnames)
plt.title('Change in area comparison')
plt.xlabel(f'Average change in area over {iterations} iterations')
plt.ylabel('Number of automata')
plt.legend() 
plt.savefig(f'areas-change.png')
plt.clf() 

plt.hist([d[1] for d in data], label=fnames)
plt.title('Change in entropy comparison')
plt.xlabel(f'Average change in entropy over {iterations} iterations') 
plt.ylabel('Number of automata') 
plt.legend() 
plt.savefig(f'entropies-change.png')
plt.clf() 

plt.hist([d[2] for d in data], label=fnames)
plt.title('Change in density comparison')
plt.xlabel(f'Average change in density over {iterations} iterations') 
plt.ylabel('Number of automata') 
plt.legend() 
plt.savefig(f'densities-change.png')
plt.clf() 
