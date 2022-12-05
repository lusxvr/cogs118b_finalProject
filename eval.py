import pandas as pd
import numpy as np
import math
#%matplotlib inline
from matplotlib import pyplot as plt
import random
#random.seed(1)
from sklearn.preprocessing import normalize

# IMPORTANT: Filepath depends on OS!
filepath = r'data/hmnist_28_28_L_pcaRed.csv'
df = pd.read_csv(filepath)
if filepath == r'data/hmnist_28_28_L_pcaRed.csv':
    df = df.iloc[:, 1:]
#print(df) # Shape (#images, h*w+1) (here: (10015, 785))

def runkmeans():
    def disp_img(data_vec):
        # Handle the cases of whether the label for an image is provided or not
        if math.sqrt(np.size(data_vec)) != int(math.sqrt(np.size(data_vec))):
            img_label = data_vec[-1]
            img_vec = data_vec[0:-1]
        else:
            img_vec = data_vec

        size = np.size(img_vec)
        dimension = int(math.sqrt(size))

        img_mat = np.reshape(np.array(img_vec), (dimension, dimension))

        plt.imshow(img_mat)
        plt.show()


    #img_number = 9869 # Index of a particular data image
    #data_vec = df.loc[img_number]

    #disp_img(data_vec)

    label = df['label']
    uniq_label, counts_label = np.unique(label, return_counts=True)

    # Each tuple will contain a label and the number of images with the same label
    # [(label, #occurences), ..]
    occurences = list(zip(uniq_label, counts_label))
    #print(occurences)
    num_cath = len(uniq_label)

    # IMPORTANT: Filepath depends on OS!
    df_meta = pd.read_csv(r'data/HAM10000_metadata.csv')
    #print(df_meta)

    cath = df_meta['dx']
    uniq_cath, counts_cath = np.unique(cath, return_counts=True)

    diagnoses = list(zip(uniq_cath, counts_cath))
    #print(diagnoses)

    # List of 7 nested lists (for each diagnostical categorie)
    # The ith nested list contains the indices of all images, which correspond to label i
    # E.g. img_index_group[2] contains all images for label 2
    img_index_group = []

    for i in range(len(uniq_label)):
        img_index_group.append([])
        for j in range(df.shape[0]):
            if df['label'][j] == i:
                img_index_group[i].append(j)

    # Output would exceed the size limit
    # print(img_index_group)

    # For debugging purposes
    # print(list((i, len(img_index_group[i])) for i in range(len(img_index_group))))

    training_split = 0.9                #Percentage of Data used for training
    img_index_group_split = []          #Same as img_index_group but every nested list is split into two lists: first is list for training, second is list for testing

    for i in range(len(img_index_group)):
        random.shuffle(img_index_group[i])
        split_point = round(len(img_index_group[i])*training_split)
        img_index_group_split.append([img_index_group[i][:split_point], img_index_group[i][split_point:]])
        
        #print(i)
        #print(len(img_index_group_split[i][0]))
        #print(len(img_index_group_split[i][1]))

    #print(img_index_group_split)

    df_train = pd.concat((df.iloc[img_index_group_split[i][0]] for i in range(num_cath)))
    #df_train.sort_index(axis=0, inplace=True)
    #print(df_train)

    def calcSqDistances(data, Kmus):
        return ((-2 * data.dot(Kmus.T) + np.sum(np.multiply(Kmus,Kmus), axis=1).T).T + np.sum(np.multiply(data, data), axis=1)).T


    def determineRnk(sqDmat):
        m = np.argmin(sqDmat, axis=1)
        return np.eye(sqDmat.shape[1])[m]


    def recalcMus(data, Rnk):
        return (np.divide(data.T.dot(Rnk), np.sum(Rnk, axis=0))).T



    classes = num_cath # Number of diagnostical categories
    max_iterations = 10000000


    images = df_train.to_numpy()
    # Remove the labels
    images = images[:, 0:-1]  # arr: Shape: (#images, #pixels)

    n = images.shape[0]
    dim = images.shape[1]

    # Initialize cluster centers by randomly picking points from the data
    rndinds = np.random.permutation(n)
    k_mus = images[rndinds[:classes]]


    for iter in range(max_iterations):
        sqDmat = calcSqDistances(images, k_mus)
        rank = determineRnk(sqDmat)
        k_mus_old = k_mus
        k_mus = recalcMus(images, rank)

        if sum(abs(k_mus_old.flatten() - k_mus.flatten())) < 1e-6:
            break


    # for i in range(classes):
    #     disp_img(k_mus[i])

    mean = []

    for i in range(num_cath):
        mean.append(np.mean(df_train.loc[img_index_group_split[i][0]].iloc[:, :-1], axis=0))

        #print(mean[i])

    MSE = np.empty([len(mean), len(k_mus)])
    for i in range(len(mean)):
        for j in range(len(k_mus)):
            MSE[j][i] = ((mean[i] - k_mus[j]).transpose().dot(mean[i] - k_mus[j]))

    df_MSE = pd.DataFrame(MSE)

    #disp_img(df_MSE, colorbar=True)

    def find_mapping(df_MSE, iter=1000):     
        mappings = {}
        mappings['Map'] = []
        mappings['Cost'] = []
        for i in range(iter):
            mapping = []
            mean_list = list(range(len(mean)))
            mus_list = list(range(len(k_mus)))

            while len(mus_list) != 0:
                i = random.choice(mus_list)
                min_idx = df_MSE.iloc[i][mean_list].idxmin()
                mapping.append(min_idx)
                mean_list.remove(min_idx)
                mus_list.remove(i)

            cost = 0
            for i in range(len(mapping)):
                cost = cost + MSE[i][mapping[i]]

            mappings['Map'].append(mapping)
            mappings['Cost'].append(cost)

        min_index = mappings['Cost'].index(min(mappings['Cost']))

        df_matching = mappings['Map'][min_index]

        return mappings['Map'][min_index]

    df_matching = find_mapping(df_MSE)

    def evaluate(): 
        right = 0
        wrong = 0

        for k in range(num_cath):
            for j in range(len(df.iloc[img_index_group_split[k][1]])):
                img = df.iloc[img_index_group_split[k][1]].iloc[j]
                img_label = img[-1]
                img_vec = img[0:-1]

                error = np.empty(num_cath)
                for i in range(num_cath):
                    error[i] = (k_mus[i] - img_vec).transpose().dot(k_mus[i] - img_vec)

                idx_mu = error.argmin()
                guess = df_matching[idx_mu]
                
                if guess == img_label:
                    right += 1
                else:
                    wrong += 1

        accuracy = right/(right+wrong)

        return accuracy

    return evaluate()

acc = []
for i in range(100):
    temp = runkmeans()
    acc.append(temp)
print(acc)
print(sum(acc))
