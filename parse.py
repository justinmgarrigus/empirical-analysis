import struct 
import sys
import os
import math 
import random 

import matplotlib.pyplot as plt
import seaborn as sns

import numpy as np 
from sklearn.linear_model import LinearRegression
from scipy.stats import ttest_ind, bootstrap 

# Execution flags
do_system_calls = False 

# Event flags
do_standard_execution = True 
do_gif_collection = True
do_exhaustive_search = True

def system(cmd):
    if not do_system_calls: 
        print('(skipped) ', end='') 
    print(cmd)
    if do_system_calls: 
        os.system(cmd)

automatons = {
    'B1357S1357':   'Replicator', 
    'B2S':          'Seeds', 
    'B3S012345678': 'Life without Death', 
    'B3S23':        'Life', 
    'B34S34':       '34 Life', 
    'B35678S5678':  'Diamoeba', 
    'B36S125':      '2x2', 
    'B36S23':       'HighLife', 
    'B3678S34678':  'Day & Night', 
    'B368S245':     'Morley', 
    'B4678S35678':  'Anneal'
}

# Reads the log file outputted from an automaton simulation and returns the 
# initial clumpiness and initial count of the starting state, as well as the 
# area, entropy, and density of each step of the simulation after the first.
def load_data(fname, iterations, executions): 
    initial_clumpinesses = [] 
    initial_counts = [] 
    areas = [] 
    entropies = [] 
    densities = [] 

    lf = f'{iterations}L'
    df = f'{iterations}d'

    f = open(fname, "rb") 
    executions = struct.unpack('L', f.read(8))[0] 
    for ex in range(executions): 
        seed = struct.unpack('L', f.read(8)) 
        initial_clumpinesses.append(struct.unpack('d', f.read(8))[0]) 
        initial_counts.append(struct.unpack('L', f.read(8))[0])
        areas.append(struct.unpack(lf, f.read(8 * iterations))) 
        entropies.append(struct.unpack(lf, f.read(8 * iterations)))
        densities.append(struct.unpack(df, f.read(8 * iterations)))
    f.close() 

    return initial_clumpinesses, initial_counts, areas, entropies, densities


def average_fig(data, statname, fname=None):
        if fname is None:   
            fname = statname

        plt.figure(figsize=(16,10))
        plt.rcParams.update({'font.size': 22}) 
        plt.boxplot(data, labels=sim_names) 
        plt.title(f'Average {statname} over {iterations} iterations for '\
                  f'{len(data[0])} executions')
        plt.xlabel('Simulation ruleset') 
        plt.ylabel(f'Average {statname}') 
        plt.xticks(rotation=90) 
        # plt.gcf().subplots_adjust(bottom=0.15)
        plt.tight_layout() 

        fname = f'figures/average-{fname}.png' 
        plt.savefig(fname)
        print('Figure saved:', fname)
        plt.clf()
        plt.close()


# Run each test and combine the results into figures. This command contains 
# the different configurable parameters we can use to change the execution 
# state.
cmd = 'gcc automata.c -o automata -lm -DITERATIONS={} -DEXECUTIONS={} ' \
      '-DROWS={} -DCOLUMNS={} -Ihashset.c hashset.c/hashset.c'

# Log files will be combined into the "logs" directory. 
if not os.path.exists('logs'): 
    os.mkdir('logs') 

# Figures will be saved to the "figures" directory. 
if not os.path.exists('figures'): 
    os.mkdir('figures') 

if do_standard_execution: 
    # Run all tests with a default configuration. 
    iterations = 32
    executions = 65536
    system(cmd.format(iterations, executions, 48, 48))
    
    sim_names = []
    sim_clumps = [] 
    sim_counts = [] 
    sim_areas = [] 
    sim_entropies = [] 
    sim_densities = [] 
    for rule, name in automatons.items():
        system(f'./automata {rule} logs/{rule}.txt')
        initial_clumpinesses, initial_counts, areas, entropies, densities = \
            load_data(f'logs/{rule}.txt', iterations, executions)
        
        sim_names.append(name + '\n' + rule) 
        sim_clumps.append(initial_clumpinesses) 
        sim_counts.append(initial_counts) 
        sim_areas.append(areas) 
        sim_entropies.append(entropies) 
        sim_densities.append(densities)

    # Find the average of each value.
    avg_areas = [
        [sum(areas) / len(areas) for areas in sim_areas[sim]] 
        for sim in range(len(sim_names))
    ]
    avg_entropies = [
        [sum(entropies) / len(entropies) for entropies in sim_entropies[sim]]
        for sim in range(len(sim_names)) 
    ]
    avg_densities = [
        [sum(densities) / len(densities) for densities in sim_densities[sim]]
        for sim in range(len(sim_names)) 
    ]
    
    average_fig(avg_areas, 'area') 
    average_fig(avg_entropies, 'entropy') 
    average_fig(avg_densities, 'density')
   
    def correlation_fig(basis, statname, shortname): 
        # Returns Pearson's correlation coefficient for the two samples. 
        def pearson_corr(x, y): 
            x_mean = sum(x) / len(x) 
            y_mean = sum(y) / len(y)

            numerator = sum(
                (x[i] - x_mean) * (y[i] - y_mean) 
                for i in range(len(x))
            )
            denominator = math.sqrt(
                sum((x[i] - x_mean) ** 2 for i in range(len(x))) * 
                sum((y[i] - y_mean) ** 2 for i in range(len(y)))
            )
            return numerator / denominator if denominator != 0 else 0 
    
        def mat(x, y): 
            return [pearson_corr(x[s], y[s]) for s in range(len(sim_names))]
       
        coefficients = [
            mat(basis, avg_areas), 
            mat(basis, avg_entropies), 
            mat(basis, avg_densities)
        ]
        fig, ax = plt.subplots(figsize=(16, 8))
        h = sns.heatmap(
            coefficients, 
            xticklabels=[
                n[:n.index('\n')].replace(' ', '\n') 
                for n in sim_names
            ], 
            yticklabels=['Area', 'Entropy', 'Density'],
            vmin=-1, 
            vmax=1,
            annot=True, 
            fmt='.2f',
            annot_kws={'rotation': 90}
        )
        h.figure.subplots_adjust(bottom=0.3)
        plt.title('Correlation between statistical averages and \n' + statname)
        
        fname = f'figures/correlation-{shortname}.png'
        plt.savefig(fname)
        print('Figure saved:', fname)
        plt.clf()
        plt.close() 

    correlation_fig(sim_counts, 'initial simulation size', 'count')
    correlation_fig(sim_clumps, 'initial simulation clumpiness', 'clump')  

    # Clumpiness was shown to correlate strongly with Life without Death.
    # We can perform linear regression on these variables with the 
    # clumpiness as the independent variable to see if we can predict it.
    def linear_regression(idx, x, y, basename, comparename, suffix=''): 
        x = x[idx] 
        y = y[idx] 
        fig, ax = plt.subplots(figsize=(16, 8))
        ax.scatter(x, y)
    
        # Split the data into training and testing sets so we can evaluate how 
        # well the model fits.
        # x, y = x.copy(), y.copy() 
        # x.shuffle() 
        # y.shuffle() 
        split = int(2/3 * len(x)) # How much of the dataset should be "train".
        x_train, x_test = x[:split], x[split:]
        y_train, y_test = y[:split], y[split:] 

        model = LinearRegression()
        def cvt(x): # Utility method to convert 1D array to 2D numpy.
            return np.array(x).reshape((-1, 1))
        
        model.fit(cvt(x_train), y_train) 
        line_x = cvt([min(x_train), max(x_train)])
        line_y = model.predict(line_x) 
    
        # Evaluate the R^2 score, which tells us how well the model fits on the
        # trained dataset. 
        y_train_mean = sum(y_train) / len(y_train) 
        rsquared_numerator = sum(
            (y_train[i] - model.predict(cvt(x_train[i]))) ** 2 
            for i in range(len(y_train))
        )
        rsquared_denominator = sum(
            (y_train[i] - y_train_mean) ** 2
            for i in range(len(y_train))
        )
        rsquared = 1 - rsquared_numerator / rsquared_denominator

        # Evaluate the root mean squared error for the points in the test set.
        rmse = math.sqrt( 1 / len(y_test) * 
            sum(
                (y_test[i] - model.predict(cvt(x_test[i]))) ** 2
                for i in range(len(y_test))
            )
        )
        
        # We'll report both the R^2 score and the root mean squared error in 
        # the title of the graph.
        simname = '"' + sim_names[idx].replace('\n', ' (') + ')"'
        title = f'Linear regression between initial {basename} and ' \
                f'average {comparename} \nfor {simname} with ' \
                f'R^2 = {round(rsquared[0], 4)} and RMSE = {round(rmse, 4)}'
        if suffix != '':
            title += ' (no outliers)' 

        ax.plot(line_x, line_y)
        plt.title(title) 
        plt.xlabel(f'Initial {basename}') 
        plt.ylabel(f'Average {comparename}') 

        fname = f'figures/linreg-{basename}-{comparename}{suffix}.png'
        plt.savefig(fname)
        print('Figure saved:', fname) 
        plt.clf()
        plt.close()

        return model

    linear_regression(5, sim_counts, avg_entropies, 'count', 'entropy')
    linear_regression(0, sim_counts, avg_densities, 'count', 'density') 
    
    # Remove the outliers from the dataset and find the correlation again.
    # The data is represented as an x-value and an associated list of y-values.
    # If any single y-value is outside of the standard deviation range, then
    # remove that x-value from all graphs.
    def remove_outliers(x, y, idx):
        # "x" is a list of integers representing independent variables. "y" is
        # a list of lists of integers representing dependent variables.
        # Get the standard deviation of each dependent variable.
        means = [sum(var) / len(var) for var in y]
        stdevs = [
            # stdev(yy) for yy in y
            math.sqrt(
                sum((var[i] - mean) ** 2 for i in range(len(var))) / len(var)
            )
            for var, mean in zip(y, means) 
        ]

        # Add items to the list only if each "y" variable is within 1 standard
        # deviations of the mean. 
        x_new = []
        y_new = [[] for _ in range(len(y))] 
        idx_list = [] 
        for i in range(len(x)):
            do_add = True 
            for idx, (mean, stdev) in enumerate(zip(means, stdevs)):
                if abs(y[idx][i] - mean) > stdev:
                    do_add = False
                    break 
            if do_add: 
                x_new.append(x[i])
                for yidx in range(len(y)): 
                    y_new[yidx].append(y[yidx][i])
                idx_list.append(i) 
        return x_new, y_new, idx_list

    remout_clumps = []
    remout_areas = [] 
    remout_entropies = [] 
    remout_densities = [] 
    idx_list = []
    for aidx in range(len(automatons)): 
        c, (a, e, d), i = remove_outliers(
            sim_clumps[aidx], 
            (avg_areas[aidx], avg_entropies[aidx], avg_densities[aidx]), aidx
        ) 
        remout_clumps.append(c)
        remout_areas.append(a)
        remout_entropies.append(e)
        remout_densities.append(d)
        idx_list.append(i)

    # Make all simulation executions the same size.
    minlen = min(len(aut) for aut in remout_clumps)
    remout_clumps = [
        remout_clumps[i][:minlen] for i in range(len(remout_clumps))
    ]
    remout_areas = [
        remout_areas[i][:minlen] for i in range(len(remout_areas))
    ]
    remout_entropies = [
        remout_entropies[i][:minlen] for i in range(len(remout_entropies))
    ]
    remout_densities = [
        remout_densities[i][:minlen] for i in range(len(remout_densities))
    ]
    idx_list = [
        idx_list[i][:minlen] for i in range(len(idx_list))
    ]

    average_fig(remout_areas, 'area (outliers removed)', 'area-out') 
    average_fig(remout_entropies, 'entropy (outliers removed)', 'entropy-out') 
    average_fig(remout_densities, 'density (outliers removed)', 'density-out')
    
    avg_areas = remout_areas 
    avg_entropies = remout_entropies 
    avg_densities = remout_densities 
    
    sim_counts = [
        [sim_counts[a][i] for i in idx_list[a]] 
        for a in range(len(automatons))
    ]
    sim_clumps = [
        [sim_clumps[a][i] for i in idx_list[a]]
        for a in range(len(automatons))
    ] 
    
    correlation_fig(
        sim_counts, 
        'initial simulation size (outliers removed)', 
        'count-out'
    )
    correlation_fig(
        sim_clumps, 
        'initial simulation clumpiness (outliers removed)', 
        'clump-out'
    )

    linear_regression(5, sim_counts, avg_entropies, 'count', 'entropy', '-out')
    linear_regression(0, sim_counts, avg_densities, 'count', 'density', '-out')
    

if do_gif_collection: 
    # Create gifs of each automaton.
    system(cmd.format(64, 1, 48, 48) + ' -DDRAW -Igifenc gifenc/gifenc.c')
    for rule, name in automatons.items(): 
        system(f'./automata {rule} /dev/null')


if do_exhaustive_search: 
    # The logs for the exhaustive search go into the "ex-logs" directory.
    directory = 'ex-logs'
    if not os.path.exists(directory):
        os.mkdir(directory) 

    iterations = 32
    executions = 65536

    sim_names = []
    sim_clumps = [] 
    sim_counts = [] 
    sim_areas = [] 
    sim_entropies = [] 
    sim_densities = []
    system(cmd.format(iterations, executions, 48, 48))
    for rule, name in automatons.items(): 
        system(f'./automata {rule} {directory}/{rule}.txt 16')
        initial_clumpinesses, initial_counts, areas, entropies, densities = \
            load_data(f'{directory}/{rule}.txt', iterations, executions)
        
        sim_names.append(name + '\n' + rule) 
        sim_clumps.append(initial_clumpinesses) 
        sim_counts.append(initial_counts) 
        sim_areas.append(areas) 
        sim_entropies.append(entropies) 
        sim_densities.append(densities)

    # Find the average of each value.
    avg_areas = [
        [sum(areas) / len(areas) for areas in sim_areas[sim]] 
        for sim in range(len(sim_names))
    ]
    avg_entropies = [
        [sum(entropies) / len(entropies) for entropies in sim_entropies[sim]]
        for sim in range(len(sim_names)) 
    ]
    avg_densities = [
        [sum(densities) / len(densities) for densities in sim_densities[sim]]
        for sim in range(len(sim_names)) 
    ] 
    
    average_fig(avg_areas, 'area (exhaustive)', 'area-ex') 
    average_fig(avg_entropies, 'entropy (exhaustive)', 'entropy-ex') 
    average_fig(avg_densities, 'density (exhaustive)', 'density-ex')

    # We want to see how similar Life (B3S23) is to HighLife (B36S23). Both of
    # their distributions seem to be equal, so we can compare them with a 
    # T-test to quantify how equivalent they are. 
    life_areas, life_entropies, life_densities = \
        avg_areas[3], avg_entropies[3], avg_densities[3] 
    highlife_areas, highlife_entropies, highlife_densities = \
        avg_areas[7], avg_entropies[7], avg_densities[7] 
    
    def multi_hist(x, labels, statname): 
        # Make all items in the list the same size.
        minlen = min(len(y) for y in x) 
        x = [y[:minlen] for y in x]
        
        t_test, p_value = ttest_ind(x[0], x[1]) 
        fig, ax = plt.subplots(figsize=(16, 7))
        plt.hist(x[0], 11, alpha=0.5, edgecolor='black', 
            histtype='barstacked', label=labels[0])
        plt.hist(x[1], 11, alpha=0.5, edgecolor='black', 
            histtype='barstacked', label=labels[1])
        plt.title(f'Comparison between average {statname} for ' + \
                  f'{"/".join(labels)}: t({round(t_test, 4)}), ' + \
                  f'p({round(p_value, 4)})')
        plt.xlabel(f'Average {statname}')
        plt.ylabel(f'Number of executions')
        plt.legend()
        fname = f'figures/{statname}-compare.png'
        plt.savefig(fname)
        print('Figure saved:', fname)
        plt.clf()
        plt.close() 

    multi_hist(
        (life_areas, highlife_areas), 
        ('life', 'highlife'), 
        'area'
    )
    multi_hist(
        (life_entropies, highlife_entropies), 
        ('life', 'highlife'), 
        'entropy'
    )
    multi_hist(
        (life_densities, highlife_densities), 
        ('life', 'highlife'), 
        'density'
    )

    def area_distribution(data, name, xaxis, keyword='', confidence=None):
        # We can show a density plot of Life to sho what the sample
        # distribution looks like. We'll compare a bootstrapped distribution to 
        # it next. 
        fig, ax = plt.subplots(figsize=(16, 7))
        kde = sns.kdeplot(data)
        ax.set_xlim(max(0, min(data)), None) 
        kde.set(title=f'Distribution of areas for {keyword} search of the ' + \
                      '\nLife (B3S23) automaton')
        kde.set(xlabel=xaxis)
        kde.set(ylabel='Density (distribution)') 
        
        # Plot a vertical line on the mean and 95% confidence intervals for the 
        # data.
        if confidence is None: 
            mean = sum(data) / len(data) 
            var = sum((x - mean) ** 2 for x in data)
            stdev = math.sqrt(var / len(data))
            low, high = mean - stdev * 2, mean + stdev * 2
            plt.axvline(x=mean, linestyle=':', label='Mean')
            plt.axvline(x=low, linestyle='--', label='95% confidence interval')
            plt.axvline(x=high, linestyle='--')
        else: 
            mean, low, high = confidence
            plt.axvline(x=mean, linestyle=':', label='Predicted mean')
            plt.axvline(
                x=low, 
                linestyle='--', 
                label='Predicted 95% confidence interval'
            )
            plt.axvline(x=high, linestyle='--') 
            plt.axvline(
                x=(sum(data) / len(data)), 
                linestyle='-.', 
                label='Calculated mean'
            )
        
        plt.legend() 
        fname = f'figures/confidence-{name}.png'
        plt.savefig(fname)
        print('Figure saved:', fname) 
        plt.close()
        
        # Returns the confidence intervals for the mean. 
        return mean, low, high

    # Bootstrap a new list of values estimating the new mean of the 
    # samples.
    bootstrap_iters = 1000
    bootstrap_distribution = [] 
    for i in range(bootstrap_iters): 
        # Randomly select a portion (subsample) of the exhaustive search
        # (sample of the underlying true population).
        subsample = [
            random.choice(life_areas) 
            for i in range(len(life_areas))
        ]

        # Get the mean of this subsample.
        mean = sum(subsample) / len(subsample) 

        # This represents a new datapoint of our bootstrapped distribution. 
        # Add it to a list so we can model it next. 
        bootstrap_distribution.append(mean) 

    _, _, _ = area_distribution(
        life_areas,  
        'sample',
        'Average area', 
        'exhaustive'
    )
    bs_mean, bs_low, bs_high = area_distribution(
        bootstrap_distribution, 
        'bootstrap', 
        'Average of average areas', 
        'bootstrapped'
    )

    # Compare how well this matches the random distribution for Life. Use the
    # same execution/iteration counts as we did for the exhaustive search. 
    system(cmd.format(iterations, executions, 48, 48))
    system(f'./automata B3S23 logs/random-B3S23.txt')
    _, _, life_random_areas, _, _ = \
        load_data(f'logs/random-B3S23.txt', iterations, executions)

    life_random_avg_areas = [sum(a) / len(a) for a in life_random_areas] 
    
    _, _, _ = area_distribution(
        life_random_avg_areas, 
        'random', 
        'Average area', 
        'random', 
        (bs_mean, bs_low, bs_high)
    )
