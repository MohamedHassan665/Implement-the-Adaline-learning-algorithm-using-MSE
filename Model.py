import pandas as pd
import numpy as np
import matplotlib
import matplotlib.pyplot as plt
from sklearn.utils import shuffle
colors = ['red', 'green', 'blue']
def preprocessing(data):
  maleCnt = 0
  femaleCnt = 0
  for idx,row in data.iterrows():
    if data.at[idx, "species"] == "Adelie":
      data.at[idx, "species"] = 0
    elif data.at[idx, "species"] == "Gentoo":
      data.at[idx, "species"] = 1
    elif data.at[idx, "species"] == "Chinstrap":
      data.at[idx, "species"] = 2
    if data.at[idx,"gender"] == "male":
      data.at[idx,"gender"] = 0
      maleCnt+=1
    elif data.at[idx,"gender"] == "female":
      femaleCnt+=1
      data.at[idx,"gender"] = 1
  #print(data.isnull().sum())
  if maleCnt >= femaleCnt:
    data["gender"].fillna(0, inplace=True)
  else:
    data["gender"].fillna(1, inplace=True)
  #print(data.isnull().sum())
  minMaxDF = pd.DataFrame(columns = ["bill_length_mm", "bill_depth_mm", "flipper_length_mm", "gender", "body_mass_g"])
  minMaxDF = minMaxDF.append({'bill_length_mm':0}, ignore_index=True)
  minMaxDF = minMaxDF.append({'bill_length_mm':0}, ignore_index=True)
  #print(minMaxDF)
  for col in data.columns:
    if(col == "species"):
      continue
    maxVal = data[col].max()
    minVal = data[col].min()
    minMaxDF[col][0] = minVal
    minMaxDF[col][1] = maxVal
    data[col] = (data[col] - minVal) / (maxVal - minVal)
  return data, minMaxDF

def samplePreprocessing(sampleData, minMaxDF):
  if "gender" in sampleData.columns:
    if sampleData.at[0,"gender"] == "male":
      sampleData.at[0,"gender"] = 0
    elif sampleData.at[0,"gender"] == "female":
      sampleData.at[0,"gender"] = 1
  for col in sampleData.columns:
    sampleData[col] = (sampleData[col] - minMaxDF[col][0]) / (minMaxDF[col][1] - minMaxDF[col][0])
  return sampleData

def visualization(data):
  for col in data.columns:
    if col == "species":
      continue
    flag = 0
    for col2 in data.columns:
      if col == col2:
        flag = 1
        continue
      if flag == 1:
        plt.scatter(data[col], data[col2], c=data["species"], cmap=matplotlib.colors.ListedColormap(colors))
        plt.xlabel(col)
        plt.ylabel(col2)
        plt.show()

def dropClass(data, class1, class2):
  for idx, row in data.iterrows():
    if data["species"][idx] != class1 and data["species"][idx] != class2:
      data.drop([idx], axis=0, inplace=True)
  return data

def prepareTarget(data, class1, class2):
  for idx,row in data.iterrows():
    if data.at[idx, "species"] == class1:
      data.at[idx, "species"] = 1
    elif data.at[idx, "species"] == class2:
      data.at[idx, "species"] = -1
  return data

def trainTestSplit(data, feature1, feature2):
  trainData = data.loc[:29,[feature1, feature2, "species"]]
  trainData = trainData.append(data.loc[50:79,[feature1, feature2, "species"]])
  trainData = trainData.reset_index()
  trainData = trainData.drop("index", axis = 1)
  testData = data.loc[30:49,[feature1, feature2, "species"]]
  testData = testData.append(data.loc[80:99,[feature1, feature2, "species"]])
  testData = testData.reset_index()
  testData = testData.drop("index", axis = 1)
  #trainData = shuffle(trainData)
  #testData = shuffle(testData)
  trainData = trainData.sample(frac=1,random_state=15).reset_index(drop=True)
  testData = testData.sample(frac=1, random_state =8).reset_index(drop=True)
  return trainData, testData

def train(weights, x, data, epoches, eta, feature1, feature2, MSE_threshold):
    prev_MSE = float(0)
    MSE_cnt = 0
    while(True):
        globalCnt = 1
        cnt = 0
        MSE = float(0)
        while (cnt < 60):
            x[0] = data[feature1][cnt]
            x[1] = data[feature2][cnt]
            net = np.dot(weights, x)
            #p = np.sign(net)
            p = net
            t = data["species"][cnt]
            weights = weights.reshape(3, 1)
            weights = weights + eta * (t - p[0][0]) * x
            #MSE += 0.5*(t - p[0][0])*(t - p[0][0])
            #if globalCnt == 30:
                #cnt += 20
                #globalCnt = 0
            weights = weights.reshape(1, 3)
            cnt += 1
            #print(t, "    ", p[0][0], "\n")
            #globalCnt += 1
        cnt = 0
        while (cnt < 60):
            x[0] = data[feature1][cnt]
            x[1] = data[feature2][cnt]
            net = np.dot(weights, x)
            p = net
            t = data["species"][cnt]
            MSE += (t - p[0][0])*(t - p[0][0])
            cnt += 1
        MSE = MSE/ 2*60
        if (round(MSE,5) == round(prev_MSE,5)):
            MSE_cnt+=1
        else:
            MSE_cnt = 0
        if(MSE_cnt == 20):
            break
        prev_MSE = MSE
        print(MSE, "\n")
        if(MSE<= MSE_threshold):
            break

    return weights

def test(weights, x, data, feature1, feature2):
    tp = 0
    tn = 0
    fp = 0
    fn = 0
    succeed = 0
    #globalCnt = 0
    cnt = 0
    weights = weights.reshape(1, 3)
    predict = []
    actual = []
    while (cnt < 40):
        x[0] = data[feature1][cnt]
        x[1] = data[feature2][cnt]
        net = np.dot(weights, x)
        p = np.sign(net)

        t = data["species"][cnt]
        predict.append(p[0][0])
        actual.append(t)
        if p[0][0] == t:
            if p[0][0] == 1:
                tp += 1
            else:
                tn += 1
            succeed += 1
        else:
            if p[0][0] == 1:
                fp += 1
            else:
                fn += 1
        #globalCnt += 1
        #if globalCnt == 20:
            #cnt += 30
            #globalCnt = 0
            #continue
        cnt += 1
    confusionMatrix = [
        [tn, fp],
        [fn, tp]
    ]
    confusionMatrix = np.array(confusionMatrix)

    print("tn    fp")
    print(tn, "  ", fp)
    print("fn    tp")
    print(fn, "  ", tp)
    print("Accuracy is ", succeed / 40 * 100, "%")

def getPoints(weights, data, feature1, feature2):
  x1 = data[feature1].min()
  y1 = (-weights[0][2] - weights[0][0]*x1) / weights[0][1]
  x2 = data[feature1].max()
  y2 =(-weights[0][2] - weights[0][0]*x2) / weights[0][1]
  return x1, y1, x2, y2

def plotLine(data, feature1, feature2, x1, y1, x2, y2,class1,class2):
    color = [colors[class1], colors[class2]]
    plt.scatter(data[feature1], data[feature2], c=data["species"], cmap=matplotlib.colors.ListedColormap(color))
    plt.xlabel(feature1)
    plt.ylabel(feature2)
    x_values = [x1, x2]
    y_values = [y1, y2]
    plt.plot(x_values, y_values)
    plt.show()

def predictSample(sampleDF, weights, feature1, feature2):
    x = np.array([sampleDF[feature1][0], sampleDF[feature2][0], 1])
    x = x.reshape(3, 1)
    net = np.dot(weights, x)
    predicted = np.sign(net)

    return predicted[0][0]

def main(feature1,feature2,class1,class2,epoches,eta,bias, MSE_threshold):

    data = pd.read_csv("penguins.csv")
    data, minMaxDF = preprocessing(data)
    # print(minMaxDF)
    visualization(data)
    weights = np.array([0.01, 0.01, 1])
    weights = weights.reshape(1, 3)
    x = np.zeros([3, 1])
    if bias==1:
        x[2]=1
    else:
        weights[0][2]=0
    data = dropClass(data, class1, class2)
    data = prepareTarget(data, class1, class2)
    data = data.reset_index()
    trainData, testData = trainTestSplit(data, feature1, feature2)
    weights = train(weights, x, trainData, epoches, eta, feature1, feature2, MSE_threshold)
    print("Learned weights are: ", weights)
    test(weights, x, testData, feature1, feature2)
    x1, y1, x2, y2 = getPoints(weights, data, feature1, feature2)
    plotLine(data, feature1, feature2, x1, y1, x2, y2,class1,class2)
    return minMaxDF,feature1,feature2,weights


