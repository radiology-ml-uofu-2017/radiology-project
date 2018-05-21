import numpy as np
import scipy.stats
np.seterr(all='raise')
class PreTransformLabels():
    def __init__(self, configs1):
        pass
        self.maxlogs = {}
        self.configs = configs1
    
    def pre_transformation(self):
        transform_dict = {'boxcox':scipy.stats.boxcox, 'none':lambda x: (x, 0), 'log': lambda x: (np.log(x),0) }
        def f(col):
            transform_result = transform_dict[self.configs['get_individual_pre_transformation'][col.name]](col)
            self.maxlogs[col.name] = transform_result[1]
            return transform_result[0]
        return f
    
    def calculate_ranges(self,all_labels):
        self.ranges_labels = {}
        for label in self.configs['get_labels_columns']:
            x = np.array(all_labels[label])
            self.ranges_labels[label] = [np.amin(x), np.amax(x)]
    
    def sigmoid_normalization(self):
        def f(col):
            if (col.name in self.configs['get_labels_columns']) and (self.configs['get_individual_output_kind'][col.name]=='sigmoid'):
                return (col-ranges_labels[col.name][0]*self.configs['sigmoid_safety_constant'][col.name][0])/ranges_labels[col.name][1]/self.configs['sigmoid_safety_constant'][col.name][1]
            else:
                return col
        return f
    
    def sigmoid_denormalization(self,np_array):
        for i in range(np_array.shape[1]):
            if (np_array.shape[1]>len(self.configs['get_labels_columns'])) and (np_array.shape[1]==len(self.configs['all_input_columns'])):
                col_name = self.configs['all_input_columns'][i]
                if col_name not in self.configs['get_labels_columns']:
                    continue
            elif np_array.shape[1]==len(self.configs['get_labels_columns']):
                col_name = self.configs['get_labels_columns'][i]
            if self.configs['get_individual_output_kind'][col_name]=='sigmoid':
                np_array[:,i] =  np_array[:,i]*self.ranges_labels[col_name][1]*self.configs['sigmoid_safety_constant'][col_name][1]+self.ranges_labels[col_name][0]*self.configs['sigmoid_safety_constant'][col_name][0]
        return np_array
        
    def inverse_pre_transform(self,np_array):
        transform_dict = {'boxcox':self.inverse_box_cox, 'none':lambda x, name: x, 'log': lambda x, name: np.exp(x) }
        for i in range(np_array.shape[1]):
            if (np_array.shape[1]>len(self.configs['get_labels_columns'])) and (np_array.shape[1]==len(self.configs['all_input_columns'])):
                col_name = self.configs['all_input_columns'][i]
                if col_name not in self.configs['get_labels_columns']:
                    continue
            elif np_array.shape[1]==len(self.configs['get_labels_columns']):
                col_name = self.configs['get_labels_columns'][i]
            b = np_array[:,i]
            np_array[:,i] = transform_dict[self.configs['get_individual_pre_transformation'][col_name]](np_array[:,i], col_name)
        return np_array
      
    def inverse_box_cox(self, y, col_name):
        lmbda = self.maxlogs[col_name]
        try:
            return np.power(lmbda*np.where(lmbda*y+1>0, y,0)+1,1/lmbda)
            return np.power(lmbda*y+1,1/lmbda)
        except:
            print(lmbda*y+1)
            print(1/lmbda)
            raise
    
    def apply_inverse_transform(self, all_labels):
        
        all_labels = [ \
            self.inverse_pre_transform(x) \
        for x in all_labels]
        all_labels = [ \
            self.sigmoid_denormalization(x) \
        for x in all_labels]
        
        return all_labels
   
        
    def apply_transform(self, all_labels):
        self.calculate_ranges(all_labels)
        all_labels[self.configs['get_labels_columns']] = all_labels[self.configs['get_labels_columns']].apply(self.sigmoid_normalization())
    
        all_labels[self.configs['get_labels_columns']] = all_labels[self.configs['get_labels_columns']].apply(self.pre_transformation())
        
        return all_labels