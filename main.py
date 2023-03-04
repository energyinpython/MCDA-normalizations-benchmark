import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
import seaborn as sns

from pyrepo_mcda.normalizations import minmax_normalization, sum_normalization, vector_normalization
from pyrepo_mcda.additions import rank_preferences
from pyrepo_mcda import weighting_methods as mcda_weights
from pyrepo_mcda import correlations as corrs
from pyrepo_mcda.mcda_methods import ARAS, TOPSIS, COCOSO
from pyrepo_mcda import normalizations as norms


def main():

    
    # ===================================================================================
    # Experiment for CoCoSo

    results_dict_rw = {
        'variable': [],
        'value': [],
        'Method': []
        }

    
    list_local = [-0.7, -0.4, -0.1, 0.2, 0.5, 0.8]
    sizes = [5, 8, 11, 14, 17, 20]

    # number of iterations: can be 1000
    iter = 100

    # loop for iterations
    for it in range(iter):
        # loop for matrix sizes
        for el, s in enumerate(sizes):

            matrix = np.random.uniform(1, 100, size = (s, s))
            types = np.ones(matrix.shape[1])
            weights = mcda_weights.critic_weighting(matrix)

            cocoso = COCOSO(normalization_method=minmax_normalization)
            pref = cocoso(matrix, weights, types)
            rank_base = rank_preferences(pref, reverse=True)

            # loop for normalization methods
            for iter, nor_met in enumerate([norms.linear_normalization, norms.max_normalization, norms.sum_normalization, norms.vector_normalization]):

                cocoso = COCOSO(normalization_method=nor_met)
                pref = cocoso(matrix, weights, types)
                rank = rank_preferences(pref, reverse=True)

                rw = corrs.weighted_spearman(rank_base, rank)

                nor_met_name = nor_met.__name__
                nor_met_name = nor_met_name.replace("_normalization", "")
                nor_met_name = nor_met_name.capitalize()

                results_dict_rw['variable'].append(sizes[el] + list_local[iter])
                results_dict_rw['value'].append(rw)
                results_dict_rw['Method'].append(nor_met_name)


    # dataframe with saved correlations
    results_pd_rw = pd.DataFrame(results_dict_rw)

    xlabels = [str(sz) + ' x ' + str(sz) for sz in sizes]

    fig, ax = plt.subplots(figsize=(9, 7))
    ax = sns.scatterplot(x="variable", y="value", data=results_pd_rw, hue="Method", s = 50, alpha = 0.1, marker = 'o')
    ax.set_xticks(sizes)
    ax.set_xticklabels(xlabels, fontsize = 14)
    ax.set_xlabel('Matrix size', fontsize = 14)
    ax.set_ylabel(r'$r_w$' + ' correlation coefficient', fontsize = 14)
    
    plt.yticks(fontsize = 14)
    ax.set_title('')
    ax.grid(True, linestyle = '--')
    ax.set_axisbelow(True)
    plt.legend(bbox_to_anchor=(0., 1.02, 1., .102), loc='lower left',
    ncol=4, mode="expand", borderaxespad=0., title = 'Normalization method', fontsize = 14)
    plt.tight_layout()
    plt.savefig('./results/scatterplot_rw_cocoso.pdf')
    plt.show()
    

    
    # ========================================================================================
    # Experiment for ARAS

    results_dict_rw = {
        'variable': [],
        'value': [],
        'Method': []
        }


    list_local = [-0.7, -0.4, -0.1, 0.2, 0.5, 0.8]
    sizes = [5, 8, 11, 14, 17, 20]

    iter = 100

    for it in range(iter):
        for el, s in enumerate(sizes):

            matrix = np.random.uniform(1, 100, size = (s, s))
            types = np.ones(matrix.shape[1])
            weights = mcda_weights.critic_weighting(matrix)

            aras = ARAS(normalization_method=sum_normalization)
            pref = aras(matrix, weights, types)
            rank_base = rank_preferences(pref, reverse=True)

            for iter, nor_met in enumerate([norms.linear_normalization, norms.max_normalization, norms.minmax_normalization, norms.vector_normalization]):

                aras = ARAS(normalization_method=nor_met)
                pref = aras(matrix, weights, types)
                rank = rank_preferences(pref, reverse=True)

                rw = corrs.weighted_spearman(rank_base, rank)

                nor_met_name = nor_met.__name__
                nor_met_name = nor_met_name.replace("_normalization", "")
                nor_met_name = nor_met_name.capitalize()

                results_dict_rw['variable'].append(sizes[el] + list_local[iter])
                results_dict_rw['value'].append(rw)
                results_dict_rw['Method'].append(nor_met_name)


    # dataframe with data
    results_pd_rw = pd.DataFrame(results_dict_rw)

    xlabels = [str(sz) + ' x ' + str(sz) for sz in sizes]

    fig, ax = plt.subplots(figsize=(9, 7))
    ax = sns.scatterplot(x="variable", y="value", data=results_pd_rw, hue="Method", s = 50, alpha = 0.1, marker = 'o')
    ax.set_xticks(sizes)
    ax.set_xticklabels(xlabels, fontsize = 14)
    ax.set_xlabel('Matrix size', fontsize = 14)
    ax.set_ylabel(r'$r_w$' + ' correlation coefficient', fontsize = 14)
    
    plt.yticks(fontsize = 14)
    ax.set_title('')
    ax.grid(True, linestyle = '--')
    ax.set_axisbelow(True)
    plt.legend(bbox_to_anchor=(0., 1.02, 1., .102), loc='lower left',
    ncol=4, mode="expand", borderaxespad=0., title = 'Normalization method', fontsize = 14)
    plt.tight_layout()
    plt.savefig('./results/scatterplot_rw_aras.pdf')
    plt.show()
    

    
    # ========================================================================================
    # Experiment for TOPSIS

    results_dict_rw = {
        'variable': [],
        'value': [],
        'Method': []
        }

    list_local = [-0.7, -0.4, -0.1, 0.2, 0.5, 0.8]
    sizes = [5, 8, 11, 14, 17, 20]

    iter = 100

    for it in range(iter):
        for el, s in enumerate(sizes):

            matrix = np.random.uniform(1, 100, size = (s, s))
            types = np.ones(matrix.shape[1])
            weights = mcda_weights.critic_weighting(matrix)

            topsis = TOPSIS(normalization_method=vector_normalization)
            pref = topsis(matrix, weights, types)
            rank_base = rank_preferences(pref, reverse=True)

            for iter, nor_met in enumerate([norms.linear_normalization, norms.max_normalization, norms.minmax_normalization, norms.sum_normalization]):

                topsis = TOPSIS(normalization_method=nor_met)
                pref = topsis(matrix, weights, types)
                rank = rank_preferences(pref, reverse=True)

                rw = corrs.weighted_spearman(rank_base, rank)

                nor_met_name = nor_met.__name__
                nor_met_name = nor_met_name.replace("_normalization", "")
                nor_met_name = nor_met_name.capitalize()

                results_dict_rw['variable'].append(sizes[el] + list_local[iter])
                results_dict_rw['value'].append(rw)
                results_dict_rw['Method'].append(nor_met_name)


    # dataframe with data
    results_pd_rw = pd.DataFrame(results_dict_rw)

    xlabels = [str(sz) + ' x ' + str(sz) for sz in sizes]

    fig, ax = plt.subplots(figsize=(9, 7))
    ax = sns.scatterplot(x="variable", y="value", data=results_pd_rw, hue="Method", s = 50, alpha = 0.1, marker = 'o')
    ax.set_xticks(sizes)
    ax.set_xticklabels(xlabels, fontsize = 14)
    ax.set_xlabel('Matrix size', fontsize = 14)
    ax.set_ylabel(r'$r_w$' + ' correlation coefficient', fontsize = 14)
    
    plt.yticks(fontsize = 14)
    ax.set_title('')
    ax.grid(True, linestyle = '--')
    ax.set_axisbelow(True)
    plt.legend(bbox_to_anchor=(0., 1.02, 1., .102), loc='lower left',
    ncol=4, mode="expand", borderaxespad=0., title = 'Normalization method', fontsize = 14)
    plt.tight_layout()
    plt.savefig('./results/scatterplot_rw_topsis.pdf')
    plt.show()
    


if __name__ == '__main__':
    main()