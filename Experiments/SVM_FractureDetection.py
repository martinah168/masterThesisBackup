import torch
import pandas as pd
from sklearn.model_selection import train_test_split
from sklearn.linear_model import LogisticRegression
from sklearn.metrics import accuracy_score, precision_score, recall_score, confusion_matrix, f1_score, balanced_accuracy_score, roc_auc_score, make_scorer
from sklearn.svm import SVC
from sklearn.model_selection import RandomizedSearchCV, GridSearchCV 
from itertools import product

def load_data(model_emb_path = '/media/DATA/martina_ma/emb_dict_3D_corpus_tsne_epoch15.pt'):
    emb_df = torch.load(model_emb_path) 
    fxclass_df = emb_df.copy()
    fxclass_df = fxclass_df[fxclass_df['fracture_grading'] != -1]
    fxclass_df = fxclass_df[fxclass_df['fracture_grading'] != 4]
    fxclass_df = fxclass_df[fxclass_df['fracture_grading'] != 1]
    fxclass_df = fxclass_df[fxclass_df['fracture_flag'] != 'U']
    class_mapping = {0: 0, 2: 1, 3: 1}
    fxclass_df['fracture_grading'] = fxclass_df['fracture_grading'].map(class_mapping)

    features_ex = fxclass_df["embeddings"].tolist()
    features_tensor_ex = torch.cat(features_ex, dim=0)
    features_array = features_tensor_ex.cpu().numpy()
    X = features_array
    y = fxclass_df['fracture_grading'].to_numpy()
    return X, y

if __name__ == '__main__':
    model2fit = '/media/DATA/martina_ma/emb_df_cleaned_epoch10_tsne.pt'#"/media/DATA/martina_ma/emb_dict_3D_corpus_tsne_epoch15.pt"
    name =model2fit.split('/')[-1]
    X, y = load_data(model2fit)
    X_train, X_val, y_train, y_val = train_test_split(X, y, test_size=0.4, random_state=42)
    # defining parameter range 
    # param_grid = {'C': [0.1, 1, 10, 100, 1000], 
    #             'gamma': [1, 0.1, 0.01, 0.001, 0.0001], 
    #             'kernel': ['rbf']} 
    param_grid = {"kernel": ['poly'], 
    "degree": [18,23,30], 'gamma': [1,0.1],  'C': [0.1, 1] #3,9,
    } 
    #auc_scorer = make_scorer(roc_auc_score)
    # degree = 9
    # gamma = 1
    # C = 1
    degrees = [3,9,18,23,30]#kernel, degree, gamma, C
    for d in degrees:# product(param_grid['kernel'], param_grid['degree'], param_grid['gamma'], param_grid['C']):
    # Your code to use the current combination of parameters goes here
            #grid = GridSearchCV(SVC(class_weight='balanced'), param_grid, scoring=auc_scorer, refit = True, verbose = 3) 
        #d = 18
        model = SVC(kernel= 'poly', degree= d, class_weight='balanced') #gamma = gamma, C = C,
        # fitting the model for grid search 
        model.fit(X, y) 

        y_pred_train = model.predict(X_train)   
        roc_auc_train = roc_auc_score(y_train,y_pred_train)
        y_pred = model.predict(X_val)
        roc_auc = roc_auc_score(y_val,y_pred)
        print('Training:', roc_auc_train)
        print('Validation:', roc_auc)
        print(f"degree: {d}")
        #print(f"kernel: {kernel}, degree: {degree}, gamma: {gamma}, C: {C}")


    
    # # print best parameter after tuning 
    # path = "/media/DATA/martina_ma/grid_search_param"+name
    # print(grid.best_params_) 
    # torch.save(param_grid, path)
    # # print how our model looks after hyper-parameter tuning
    # path = "/media/DATA/martina_ma/grid_search_estimator"+name
    # print(grid.best_estimator_) 
    # torch.save(grid.best_estimator_,path)

#  0.8775938208478019
#    degree = 3
#     gamma = 1
#     C = 1
    
#     0.9082037006713608
    # degree = 9
    # gamma = 1
    # C = 1
# 
# 
# 
# 

# Training: 0.9192053289923003
# Validation: 0.9357651518761483
# kernel: poly, degree: 9, gamma: 1, C: 0.1
        
#         Training: 0.900273419592654
# Validation: 0.9201740870753055
# kernel: poly, degree: 9, gamma: 1, C: 1
#         Training: 0.8235842525283711
# Validation: 0.8129686399711973
# degree: 18
        
# /media/DATA/martina_ma/emb_dict_3D_cleaned_balanced_tsne.pt
# Training: 0.7568825020816377
# Validation: 0.7449412562013226
# degree: 30, gamma = 0.1
# 
# Training: 0.812181556222927
#Validation: 0.8242006448846038
#degree: 9, gamma = 1
# 
# 
# '/media/DATA/martina_ma/emb_df_cleaned_epoch10_tsne.pt'
# 
# Training: 0.877495024632149
# Validation: 0.8667962220034824
# degree: 3
# degree 9 stuck with gamma 1
# 
#    
        #     Training: 0.8181755897034355
#gamma default 
Training: 0.777770708949137                                                                                                               
Validation: 0.7674872389814228                                                                                                            
degree: 3                                                                                                                                 
Training: 0.82210966472002                                                                                                                
Validation: 0.7964713139056614                                                                                                            
degree: 9  
Validation: 0.8017177587489583
degree: 18
Training: 0.7989712135547509
Validation: 0.801081570198763
degree: 23
Training: 0.8214044131240961
Validation: 0.802236244967853
degree: 30