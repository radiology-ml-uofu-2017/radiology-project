import numpy as np
import matplotlib
matplotlib.use('Agg')
import matplotlib.pyplot as plt
from configs import configs
import logging
import sklearn.metrics

def perf_measure(y_actual, y_hat):
    TP = 0
    FP = 0
    FN = 0
    diff = y_actual - y_hat
    fn = np.sum(np.greater(diff, 0)*diff, axis = 0)
    fp = -np.sum(np.less(diff, 0)*diff, axis = 0)
    tp = np.sum(y_actual, axis = 0)-fn
    tn = np.sum(np.equal(diff, 0), axis = 0)-tp

    return {'tp':tp,'fp': fp, 'fn': fn, 'tn':tn}
  
def sum_dictionaries_by_key(x,y):
    return {k: x.get(k, 0) + y.get(k, 0) for k in set(x) | set(y)}
  
def get_precision_recall_from_dictionary(d):
    precision = d['tp']/float(d['tp']+d['fp'])
    recall = d['tp']/float(d['tp']+d['fn'])
    f1score = 2*precision*recall/(precision+recall)
    return {'precision':precision, 'recall':recall, 'f1score':f1score }
  
def r2(y_corr, y_pred):
    y_corr_mean = np.mean(y_corr)
    sstot = np.sum(np.square(y_corr-y_corr_mean))
    ssres = np.sum(np.square(y_pred-y_corr))
    return 1-ssres/sstot

def get_copd_diagnose(fev1fvc_predrug):
    if fev1fvc_predrug>= 0.7:
        return 0
    return 1
    
def get_gold(fev1_ratio, fev1fvc_predrug):
    if fev1fvc_predrug>= 0.7:
        return 0
    if fev1_ratio>=0.8:
        return 1
    if fev1_ratio>=0.5:
        return 2
    if fev1_ratio>=0.3:
        return 3
    return 4
  
def report_final_results(y_corr , y_pred, train):
    if train:
        train_string = 'train'
    else:
        train_string = 'val'
    markers = ['b.','g.', 'r.', 'c.', 'm.', 'y.', 'k.', 'b>', 'g>','r>']
    plot_var_for_legend = []
    
    for k in range(len(configs['pft_plot_columns'])):
        this_plot, = plt.plot(configs['pft_plot_function'](y_corr, k),configs['pft_plot_function'](y_pred, k), markers[k])
        #plot_var_for_legend.append(this_plot)
        plot_var_for_legend=[this_plot]
    
        plt.legend(plot_var_for_legend, [configs['pft_plot_columns'][k]],
                    scatterpoints=1,
                    loc='best',
                    ncol=2,
                    fontsize=8)
        
        ax = plt.gca()
        ylim = ax.get_ylim()
        xlim = ax.get_xlim()
        final_axis = (min(ylim[0],xlim[0]), max(ylim[1],xlim[1]))
        ax.set_ylim(final_axis)
        ax.set_xlim(final_axis)
        plt.plot([final_axis[0], final_axis[1]], [final_axis[0], final_axis[1]], 'k-', lw=1)
        plt.xlabel('Groundtruth', fontsize=10)
        plt.ylabel('Predicted', fontsize=10)
        plt.savefig('plots/' + train_string + '_' +configs['pft_plot_columns'][k]+ '_' + configs['output_image_name'])
        plt.clf()
        plt.cla()
        plt.close()
    
    r2s = {}
    for k in range(len(configs['pft_plot_columns'])):
        r2s[configs['pft_plot_columns'][k]] = r2(y_corr = configs['pft_plot_function'](y_corr,k), y_pred = configs['pft_plot_function'](y_pred,k))
        #print('sklearn: ' + str(sklearn.metrics.r2_score(y_true = configs['pft_plot_function'](y_corr,k), y_pred = configs['pft_plot_function'](y_pred,k))))
        #print('numpy: ' + str(r2s[configs['pft_plot_columns'][k]]))
    logging.info('r2: ' + str(r2s))
    accuracies = {}
    for k in range(len(configs['get_labels_columns_copd'])):
        accuracies[configs['get_labels_columns_copd'][k]] = {}
        accuracies[configs['get_labels_columns_copd'][k]]['roc'] = sklearn.metrics.average_precision_score(y_true = y_corr[:,-k], y_score = y_pred[:,-k])
        perfs = perf_measure(y_actual = y_corr[:,-k], y_hat = (y_pred[:,-k]>0.5)*1)
        accuracies[configs['get_labels_columns_copd'][k]]['perfs'] = perfs
        accuracies[configs['get_labels_columns_copd'][k]]['scores'] = get_precision_recall_from_dictionary(perfs)
        accuracies[configs['get_labels_columns_copd'][k]]['accuracy'] = (perfs['tn']+perfs['tp'])/(perfs['tn']+perfs['tn']+perfs['fp']+perfs['fn'])
    logging.info('accuracy: : ' + str(accuracies))