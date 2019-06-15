
import pandas as pd
import numpy as np

data_set = pd.read_csv("hw01_data_set_images.csv", header=None)
label_set = pd.read_csv("hw01_data_set_labels.csv", header=None)

# Slicing the train data and test data rows
train_a = np.array(data_set.loc[0:24, ])
train_b = np.array(data_set.loc[39:63, ])
train_c = np.array(data_set.loc[78:102, ])
train_d = np.array(data_set.loc[117:141, ])
train_e = np.array(data_set.loc[156:180, ])

test_a = np.array(data_set.loc[25:38, ])
test_b = np.array(data_set.loc[64:77, ])
test_c = np.array(data_set.loc[103:116, ])
test_d = np.array(data_set.loc[142:155, ])
test_e = np.array(data_set.loc[181:194, ])

# Slicing the labels of the training data and converting it to a numpy array
label_a = label_set.loc[0:24, ]
label_b = label_set.loc[39:63, ]
label_c = label_set.loc[78:102, ]
label_d = label_set.loc[117:141, ]
label_e = label_set.loc[156:180, ]



label_a = label_a.replace("A", 1)
label_b = label_b.replace("B", 2)
label_c = label_c.replace("C", 3)
label_d = label_d.replace("D", 4)
label_e = label_e.replace("E", 5)

label_all = np.array(pd.concat([label_a,label_b,label_c,label_d,label_e], axis=0))

label_a = np.array(label_a)
label_b = np.array(label_b)
label_c = np.array(label_c)
label_d = np.array(label_d)
label_e = np.array(label_e)


# slicing the labels of the test data and converting it to numpy array
label_test_a = label_set.loc[25:38, ]
label_test_b = label_set.loc[64:77, ]
label_test_c = label_set.loc[103:116, ]
label_test_d = label_set.loc[142:155, ]
label_test_e = label_set.loc[181:195, ]



label_test_a = label_test_a.replace("A", 1)
label_test_b = label_test_b.replace("B", 2)
label_test_c = label_test_c.replace("C", 3)
label_test_d = label_test_d.replace("D", 4)
label_test_e = label_test_e.replace("E", 5)

label_test_all = np.array(pd.concat([label_test_a,label_test_b,label_test_c,label_test_d,label_test_e], axis=0))

label_test_a = np.array(label_test_a)
label_test_b = np.array(label_test_b)
label_test_c = np.array(label_test_c)
label_test_d = np.array(label_test_d)
label_test_e = np.array(label_test_e)


def safe_log(array):
    """
    this function maskes all the unloglable 0 values, then refilling it with 0
    :param array: the numpy array that will be take log of.
    :return: the array that has been taken safe log of
    """
    return np.log(array + 1e-100)



def pcd(given_array):
    a = np.sum(given_array, axis=0)
    a = a / len(given_array)
    return a


pcdA = pcd(train_a)
pcdB = pcd(train_b)
pcdC = pcd(train_c)
pcdD = pcd(train_d)
pcdE = pcd(train_e)


def g_score(trained, p,label):
    log_p = safe_log(p)
    cp = np.subtract(1, p)
    log_cp = safe_log(cp)
    Xd = trained
    cXd = np.subtract(1, Xd)

    a = np.sum(np.multiply(log_p, Xd), 1)
    b = np.sum(np.multiply(log_cp, cXd), 1)
    c = np.log(len(Xd) / len(label))
    return a + b + c


def g_score_comparer(trained, label):
    gA = g_score(trained, pcdA, label)
    gB = g_score(trained, pcdB, label)
    gC = g_score(trained, pcdC, label)
    gD = g_score(trained, pcdD, label)
    gE = g_score(trained, pcdE, label)
    result = np.array(gA)
    i = 0
    while i < len(gA):
        temp = 0
        temp = max(gA[i], gB[i], gC[i], gD[i], gE[i])
        if gA[i] == temp:
            result[i] = "1"
        if gB[i] == temp:
            result[i] = "2"
        if gC[i] == temp:
            result[i] = "3"
        if gD[i] == temp:
            result[i] = "4"
        if gE[i] == temp:
            result[i] = "5"
        else:
            pass
        i += 1
    return result


hat_a = g_score_comparer(test_a, label_test_all)
hat_b = g_score_comparer(test_b, label_test_all)
hat_c = g_score_comparer(test_c, label_test_all)
hat_d = g_score_comparer(test_d, label_test_all)
hat_e = g_score_comparer(test_e, label_test_all)

labeledA = g_score_comparer(train_a, label_all)
labeledB = g_score_comparer(train_b, label_all)
labeledC = g_score_comparer(train_c, label_all)
labeledD = g_score_comparer(train_d, label_all)
labeledE = g_score_comparer(train_e, label_all)


def conf_matrix(y_hat, y_label):
    d = {1: [0, 0, 0, 0, 0], 2: [0, 0, 0, 0, 0], 3: [0, 0, 0, 0, 0],
         4: [0, 0, 0, 0, 0],
         5: [0, 0, 0, 0, 0]}
    df = pd.DataFrame(data=d)
    df.index = df.index + 1
    for val in y_hat:
        col = y_label[int(val)]
        df.loc[val, col] += 1
    return df


conf_a = conf_matrix(labeledA, label_a)
conf_b = conf_matrix(labeledB, label_b)
conf_c = conf_matrix(labeledC, label_c)
conf_d = conf_matrix(labeledD, label_d)
conf_e = conf_matrix(labeledE, label_e)

conf_all = conf_a + conf_b + conf_c + conf_d + conf_e
print("      y_train")
print(conf_all)



conf_test_a = conf_matrix(hat_a, label_test_a)
conf_test_b = conf_matrix(hat_b, label_test_b)
conf_test_c = conf_matrix(hat_c, label_test_c)
conf_test_d = conf_matrix(hat_d, label_test_d)
conf_test_e = conf_matrix(hat_e, label_test_e)

conf_test_all = conf_test_a + conf_test_b + conf_test_c + conf_test_d +conf_test_e

print("      y_test")
print(conf_test_all)
