# os.walk into each data dir 
# os.walk into each sub dir with name crbLargeT, res8 res20 res32 res50
# return all files with the name accuracy.csv from all data dir with sub dir
# cbrLargeT, res8 res20 res32 res50
# combine columns with name accuracy from all files into a seperate file
import pandas as pd

def graph_acc(df_mean):
    
    import matplotlib.pyplot as plt
    from matplotlib import cm
    import csv
    import pandas as pd
    import os

    cwd = os.getcwd()
    print('current directory: {}'.format(cwd))    
    print(df_mean)

    ax = plt.gca()

    df_mean.plot(kind='line', linestyle='-', linewidth=1, marker='o', markersize=3, x=r'$\epsilon$', y='CBR-LargeT', ax=ax, yerr='std_CBR-LargeT', capsize=2, capthick=1, logx=True, c='darkblue', fontsize=12)
    df_mean.plot(kind='line', linestyle='-', linewidth=1, marker='o', markersize=3, x=r'$\epsilon$', y='Resnet 8', ax=ax, yerr='std_Resnet 8', capsize=2, capthick=1, logx=True, c='purple', fontsize=12)
    df_mean.plot(kind='line', linestyle='-', linewidth=1, marker='o', markersize=3, x=r'$\epsilon$', y='Resnet 20', ax=ax, yerr='std_Resnet 20', capsize=2, capthick=1, logx=True, c='darkred', fontsize=12)
    df_mean.plot(kind='line', linestyle='-', linewidth=1, marker='o', markersize=3, x=r'$\epsilon$', y='Resnet 32', ax=ax, yerr='std_Resnet 32', capsize=2, capthick=1, logx=True, c='darkgreen', fontsize=12)
    df_mean.plot(kind='line', linestyle='-', linewidth=1, marker='o', markersize=3, x=r'$\epsilon$', y='Resnet 50', ax=ax, yerr='std_Resnet 50', capsize=2, capthick=1, logx=True, c='black',fontsize=12)

    ax.grid('on', which='major', axis='both', linewidth=0.4, color='gainsboro')
    
    ax.spines["top"].set_linewidth(0.4)
    ax.spines["bottom"].set_linewidth(0.4)
    ax.spines["right"].set_linewidth(0.4)
    ax.spines["left"].set_linewidth(0.4)
    
    ax.spines['left'].set_color('gainsboro')
    ax.spines['right'].set_color('gainsboro')
    ax.spines['top'].set_color('gainsboro')
    ax.spines['bottom'].set_color('gainsboro')

    ax.xaxis.set_ticks_position('none') 
    ax.yaxis.set_ticks_position('none') 
    
    plt.xlabel(r'$\epsilon$ (log scale)', fontsize=14)
    plt.ylabel('accuracy (%)', fontsize=14)
    plt.legend(fontsize=10)

    plt.savefig('/path.../experiments/fgsm/acc_graphs/pgd/xray/avg_acc_std_graph_pgd.pdf')
    plt.show()

def get_acc():

    import fnmatch
    import os
    import glob
    import pandas as pd
    import sys

    # create a list of lists to hold names of all accuracy.csv files for each architecture size (res8, res20, res32, res50)
    architecture_size = [[] for x in range(5)]

    cwd = os.getcwd()
   
    
    for root, dirs, files in os.walk(cwd):
        #print(root)
        for filename in fnmatch.filter(files, 'accuracy_pgd.csv'):
            acc_file_path = os.path.join(root, filename)

            if ('res8' in acc_file_path):
                print(acc_file_path)
                architecture_size[0].append(acc_file_path)
       
            elif ('res20' in acc_file_path):
       	       	print(acc_file_path)
                architecture_size[1].append(acc_file_path)

            elif ('res32' in acc_file_path):
       	       	print(acc_file_path)
                architecture_size[2].append(acc_file_path)

            elif ('res50' in acc_file_path):
       	       	print(acc_file_path)
                architecture_size[3].append(acc_file_path)

            elif ('five_layer_cnn' in acc_file_path):
       	       	print(acc_file_path)
                architecture_size[4].append(acc_file_path)

            else:
                continue
            '''
            elif ('cbr_tiny' in acc_file_path):
       	       	print(acc_file_path)
                architecture_size[5].append(acc_file_path)
            '''

    print('Number of Resnet 8 accuracy.csv files: {}'.format(len(architecture_size[0])))
    print('Number of Resnet 20 accuracy.csv files: {}'.format(len(architecture_size[1])))
    print('Number of Resnet 32 accuracy.csv files: {}'.format(len(architecture_size[2])))
    print('Number of Resnet 50 accuracy.csv files: {}'.format(len(architecture_size[3])))
    print('Number of cbrLargeT accuracy.csv files: {}'.format(len(architecture_size[4])))

    # Create an empty Dataframe                       
    #df = pd.DataFrame()
    acc_data = []
    import csv
    col_name = ['acc_1', 'acc_2', 'acc_3', 'acc_4', 'acc_5', 'acc_6', 'acc_7', 'acc_8', 'acc_9', 'acc_10'] 
    col_arch = ['Resnet 8', 'Resnet 20', 'Resnet 32', 'Resnet 50', 'CBR-LargeT']
    eps = [0.01,0.02,0.03,0.04,0.05,0.06,0.07,0.08,0.09,0.1,0.2,0.3,0.4,0.5,0.6,0.7,0.8,0.9,1,2,3,4,5,6,7,8,9,10]
    
    # df with 5 columns, mean values for cbrLargeT,resnet8,20,32,50
    df_mean = pd.DataFrame()
    df_std = pd.DataFrame()
    
    for i in range(len(architecture_size)):
        df = pd.DataFrame()
        print(len(architecture_size)) 
        for idx, j in enumerate(architecture_size[i]):
            print(len(architecture_size[i]))
            #print(idx)
            #print(j)
            df2 = pd.read_csv(j)
            data = df2['accuracy']
            df[col_name[idx]]=pd.Series(data)
     
        df['mean'] = df.mean(axis=1)
        mean_data = df['mean']
        df_mean[col_arch[i]]=pd.Series(mean_data)

        df['std'] = df.std(axis=1)
        std_data = df['std']
        df_mean['std_'+col_arch[i]]=pd.Series(std_data)


    
        df.to_csv( "/path.../experiments/fgsm/acc_graphs_09_22_20/pgd/xray/new_acc_{}_pgd.csv".format(i), index=False)
    df_mean[r'$\epsilon$']=pd.Series(eps)
    graph_acc(df_mean)    

    return

get_acc()

