import numpy as np
import matplotlib
matplotlib.use('Agg')
import matplotlib.pyplot as plt
from configs import configs
import logging
import sklearn.metrics

plot_configs = configs.get_enum_return('get_plot_configs')
plot_function = plot_configs['plot_function']
plot_columns = plot_configs['plot_columns']

def r2(y_corr, y_pred):
    y_corr_mean = np.mean(y_corr)
    sstot = np.sum(np.square(y_corr-y_corr_mean))
    ssres = np.sum(np.square(y_pred-y_corr))
    return 1-ssres/sstot

def report_final_results(y_corr , y_pred, train):
    if train:
        train_string = 'train'
    else:
        train_string = 'val'
    markers = ['b.','g.', 'r.', 'c.', 'm.', 'y.', 'k.', 'b>', 'g>','r>']
    plot_var_for_legend = []
    
    for k in range(len(plot_columns)):
        this_plot, = plt.plot(plot_function(y_corr, k),plot_function(y_pred, k), markers[k])
        plot_var_for_legend.append(this_plot)
    
    plt.legend(plot_var_for_legend, plot_columns,
                scatterpoints=1,
                loc='best',
                ncol=2,
                fontsize=8)
    r2s = {}
    for k in range(len(plot_columns)):
        r2s[plot_columns[k]] = r2(y_corr = plot_function(y_corr,k), y_pred = plot_function(y_pred,k))
        #print('sklearn: ' + str(sklearn.metrics.r2_score(y_true = plot_function(y_corr,k), y_pred = plot_function(y_pred,k))))
        #print('numpy: ' + str(r2s[plot_columns[k]]))
    logging.info('r2: ' + str(r2s))
    
    ax = plt.gca()
    ylim = ax.get_ylim()
    xlim = ax.get_xlim()
    final_axis = (min(ylim[0],xlim[0]), max(ylim[1],xlim[1]))
    ax.set_ylim(final_axis)
    ax.set_xlim(final_axis)
    plt.plot([final_axis[0], final_axis[1]], [final_axis[0], final_axis[1]], 'k-', lw=1)
    plt.xlabel('Groundtruth', fontsize=10)
    plt.ylabel('Predicted', fontsize=10)
    plt.savefig('plots/' + train_string + configs['output_image_name'])
    plt.clf()
    plt.cla()
    plt.close()