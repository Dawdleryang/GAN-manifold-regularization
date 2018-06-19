# classification and regression for IC design data analysis
# yang xulei 6/6/2018

import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
from sklearn.model_selection import KFold
from sklearn.preprocessing import StandardScaler

from keras.models import Sequential
from keras.layers import LSTM, Dense
from keras.callbacks import ReduceLROnPlateau, EarlyStopping
from keras.utils import np_utils
scaler = StandardScaler()
CLASSES = 2

def confusion_matrix(y, y_pred):
    TN = 0
    FN = 0
    FP = 0
    TP = 0
    for i in range(y.shape[0]):
        if y[i] == 0 and y_pred[i] == 0:
            TN = TN + 1
        elif y[i] == 1 and y_pred[i] == 0:
            FN = FN + 1
        elif y[i] == 0 and y_pred[i] == 1:
            FP = FP + 1
        elif y[i] == 1 and y_pred[i] == 1:
            TP = TP + 1
    return TN, FN, FP, TP


def derive_metric(TN, FN, FP, TP):
    overall = float(TP + TN) / float(TP + TN + FN + FP)
    average = (TP / float(TP + FP) + TN / float(FN + TN)) / 2
    sens = TP / float(TP + FN)
    spec = TN / float(TN + FP)
    ppr = TP / float(TP + FP)

    return overall, average, sens, spec, ppr


def read_data(csv_file):
    data = pd.read_csv(csv_file)
    input = data.iloc[:, 3:8]
    input = np.array(input)
    input = scaler.fit_transform(input)

    output = data.iloc[:, 8]
    output = np.array(output)
    output[np.where(output == 'pass')] = 0
    output[np.where(output == 'near')] = 2
    output[np.where(output == 'fail')] = 1

    print("No. of pass - " + str(np.sum(output == 0)))
    print("No. of near - " + str(np.sum(output == 2)))
    print("No. of fail - " + str(np.sum(output == 1)))
    if CLASSES==2:
        X = input[np.where(output < 2)]
        Y = output[np.where(output < 2)]
    else:
        X = input
        Y = output
    # print(X.shape)
    # print(Y.shape)
    return X, Y


def build_model():
    model = Sequential()
    model.add(Dense(64, input_dim=5, activation='relu'))
    model.add(Dense(256, activation='relu'))
    model.add(Dense(256, activation='relu'))
    model.add(Dense(3, activation="sigmoid"))

    # model.compile(loss='binary_crossentropy',
    model.compile(loss='categorical_crossentropy',
                  optimizer='adam',
                  metrics=['accuracy'])

    model.summary()

    return model


def do_train(epochs=100):
    csv_file = "AnalogDataOpamp_v1.csv"
    input, output = read_data(csv_file)
    input = input.astype('float32')
    # print(output)
    output = np_utils.to_categorical(output, 3)
    # print(output)
    # exit(0)

    # do 5-fold cross validation
    folds = 4
    kf = KFold(n_splits=folds, random_state=2018, shuffle=True)
    predictions = np.zeros(len(output), np.float32)

    kfold = 0
    for train_index, test_index in kf.split(output):
        kfold = kfold + 1
        X_train, X_test = input[train_index], input[test_index]
        Y_train, Y_test = output[train_index], output[test_index]
        print("kfold={:d}".format(kfold))
        print(X_train.shape, X_test.shape, Y_train.shape, Y_test.shape)

        lr_reducer = ReduceLROnPlateau(factor=np.sqrt(0.1), cooldown=0, patience=5, min_lr=0.5e-6)
        early_stopper = EarlyStopping(monitor='val_loss', min_delta=0.0003, patience=500)
        # early_stopper = EarlyStopping(monitor='val_loss', patience=500)
        model = build_model()
        model.fit(X_train, Y_train,
                  batch_size=10,
                  nb_epoch=epochs,
                  validation_split=0.1,
                  shuffle=True,
                  callbacks=[lr_reducer, early_stopper])
        # callbacks=[lr_reducer, csv_logger])

        score = model.evaluate(X_test, Y_test, verbose=0)
        print('Test loss:', score[0])
        print('Test accuracy:', score[1])

        Y_train_pred = model.predict(X_train, batch_size=128, verbose=0)
        Y_train_pred = Y_train_pred.argmax(1)
        Y_test_pred = model.predict(X_test, batch_size=128, verbose=0)
        Y_test_pred = Y_test_pred.argmax(1)

        """
        TN_train, FN_train, FP_train, TP_train = confusion_matrix(Y_train, Y_train_pred)
        TN_test, FN_test, FP_test, TP_test = confusion_matrix(Y_test, Y_test_pred)

        overall_train, average_train, sens_train, spec_train, ppr_train = derive_metric(TN_train, FN_train, FP_train, TP_train)
        overall_test, average_test, sens_test, spec_test, ppr_test = derive_metric(TN_test, FN_test, FP_test, TP_test)

        print('Train Metrics:', overall_train, average_train, sens_train, spec_train, ppr_train)
        print('Test Metrics:', overall_test, average_test, sens_test, spec_test, ppr_test)
        """


do_train(epochs=5)
