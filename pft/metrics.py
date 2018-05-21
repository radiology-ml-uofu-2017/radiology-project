import numpy as np
import matplotlib
matplotlib.use('Agg')
import matplotlib.pyplot as plt
from configs import configs
import logging
import sklearn.metrics
import scipy.stats

# defining what variables are going to be ploted (plot_columns) and how they are calculated
# using the variables present in labels_to_use
def absolutes_and_important_ratios_plot_calc(values, all_true_values, name):
    if name in configs['get_labels_columns']:
        return get_plot_value(values, all_true_values, name)
    elif name=='fev1_ratio':
        return get_plot_value(values, all_true_values, 'fev1_predrug')/get_plot_value(values, all_true_values, 'fev1_pred')
    elif name=='fev1fvc_predrug':
        return get_plot_value(values, all_true_values, 'fev1_predrug')/get_plot_value(values, all_true_values,'fvc_predrug')
    else:
        raise_with_traceback(ValueError(name + ' is not a valid name value for function absolutes_and_important_ratios_plot_calc'))
      
def get_plot_value(values, all_true_values, name):
    if configs['use_true_predicted'] and name.endswith('_pred'):
        return all_true_values[:,configs['all_input_columns'].index(name)]
    return values[:,configs['get_labels_columns'].index(name)]

def plot_results(y_corr, y_pred, y_corr_all, train_string):
    markers = ['b.','g.', 'r.', 'c.', 'm.', 'y.', 'k.', 'b>', 'g>','r>']
    prettify_name = {'fvc_pred':'Predicted FVC', 'fev1_pred':'Predicted FEV1', 'fev1fvc_pred':'Predicted FEV1/FVC', 
                     'fev1_predrug':'Pre-drug FEV1', 'fvc_predrug':'Pre-drug FVC', 'fev1fvc_predrug':'Pre-drug FEV1/FVC', 
                     'fev1_ratio':'Pre-drug/Predicted FEV1', 'fvc_ratio':'Pre-drug/Predicted FVC', 'fev1fvc_ratio':'Pre-drug/Predicted FEV1/FVC'}
    
    for i, plot_list in enumerate(configs['pft_plot_columns']):
        plot_var_for_legend = []
        name_var_for_legend = []
        for k in range(len(plot_list)):
            name = plot_list[k]
            this_plot, = plt.plot(absolutes_and_important_ratios_plot_calc(y_corr, y_corr_all, name),absolutes_and_important_ratios_plot_calc(y_pred, y_corr_all, name), markers[k])
            plot_var_for_legend.append(this_plot)
            name_var_for_legend.append(prettify_name[name])
        plt.legend(plot_var_for_legend, name_var_for_legend,
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
        plt.savefig('plots/' + train_string + '_' + (str(i) if len(plot_list)>1 else plot_list[0]) + '_' + configs['output_image_name'])
        plt.clf()
        plt.cla()
        plt.close()

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
    accuracy = (d['tp']+d['tn'])/float(d['tp']+d['fp']+d['fn']+d['tn'])
    return {'precision':precision, 'recall':recall, 'f1score':f1score, 'accuracy': accuracy}
  
def r2(y_corr, y_pred):
    y_corr_mean = np.mean(y_corr)
    sstot = np.sum(np.square(y_corr-y_corr_mean))
    ssres = np.sum(np.square(y_pred-y_corr))
    return 1-ssres/sstot

def get_copd_diagnose(fev1fvc_predrug):
    return (fev1fvc_predrug< 0.7)*1
    
def get_gold(fev1_ratio, fev1fvc_predrug):
    return (fev1fvc_predrug<0.7)*(1+(fev1_ratio<0.8)+(fev1_ratio<0.5)+(fev1_ratio<0.3))

def get_accuracies(y_corr, y_pred):
    accuracies = {}
    accuracies['roc'] = sklearn.metrics.average_precision_score(y_true = y_corr, y_score = y_pred)
    perfs = perf_measure(y_actual = y_corr, y_hat = (y_pred>0.5)*1)
    accuracies['perfs'] = perfs
    accuracies['scores'] = get_precision_recall_from_dictionary(perfs)
    accuracies['accuracy'] = (perfs['tn']+perfs['tp'])/float(perfs['tn']+perfs['tp']+perfs['fp']+perfs['fn'])
    return accuracies

def report_final_results(y_corr , y_pred, y_corr_all, train):
    if train:
        train_string = 'train'
    else:
        train_string = 'val'
    logging.info('metrics for ' + train_string + ' set:' )
    
    y_corr, y_pred, y_corr_all = configs['pre_transform_labels'].apply_inverse_transform((y_corr, y_pred, y_corr_all))
        
    plot_results(y_corr, y_pred, y_corr_all, train_string)
        
    r2s = {}
    correlations = {}
    output_variables = list(set(sum(configs['pft_plot_columns'],[])))
    accuracies = {}
    for k in range(len(output_variables)):
        name = output_variables[k]
        this_y_corr = absolutes_and_important_ratios_plot_calc(y_corr, y_corr_all, name)
        this_y_pred = absolutes_and_important_ratios_plot_calc(y_pred, y_corr_all, name)
        r2s[name] = r2(y_corr = this_y_corr, y_pred = this_y_pred)
        #print('sklearn: ' + str(sklearn.metrics.r2_score(y_true = configs['pft_plot_function'](y_corr,k), y_pred = configs['pft_plot_function'](y_pred,k))))
        #print('numpy: ' + str(r2s[name]))
        pearson, pvalue = scipy.stats.pearsonr(this_y_corr, this_y_pred)
        correlations[name] = {'pearson':pearson, 'pvalue':pvalue}
        if name == 'fev1fvc_predrug':
            accuracies['copd_from_pft'] = get_accuracies(get_copd_diagnose(this_y_corr),  get_copd_diagnose(this_y_pred)) 
    logging.info('r2: ' + str(r2s))
    logging.info('correlation statistical test: ' + str(correlations))
    if len(configs['get_labels_columns_copd'])>0:
        accuracies[configs['get_labels_columns_copd'][0]] = get_accuracies(absolutes_and_important_ratios_plot_calc(y_corr, y_corr_all, 'copd') ,  absolutes_and_important_ratios_plot_calc(y_pred, y_corr_all, 'copd')) 
    logging.info('accuracy: ' + str(accuracies))