import pandas as pd
from sklearn.model_selection import train_test_split
from python_files.preprocessing import ClassificationPreprocess
from python_files.learning_models import ClassifyModel
from python_files.visualization import *
from python_files.helper_functions import Helper

class Program:
    def __init__(self, file_name, model_id, pca_mode=False, train_size=0.8):
        self.train_size = train_size
        self.helper = Helper()
        self.label = 'Rate'                               # label, output, y
        self.classes = ['Low', 'Intermediate', 'High']

        self.data = pd.read_csv(file_name)                # data -> dataframe
        self.preprocess_obj = ClassificationPreprocess(self.label, self.classes, train_size*10) #1 for classification.
        self.classification_model = ClassifyModel(model_id, pca_mode)

    def start(self):
        if self.train_size == 1:    #Train with all data.
            data_train = self.data
            data_test = []
        else:
            data_train, data_test = train_test_split(self.data, test_size=1-self.train_size, shuffle=True)

        # Train Process.
        cleaned_data_train = self.preprocess_obj.start_for_training(data_train)
        copy_cleaned_data_train = cleaned_data_train
        y_train = copy_cleaned_data_train[self.label]                 # Train_Target
        X_train = copy_cleaned_data_train.drop([self.label], axis=1)  # Train_Input
        y_train = y_train.astype('int')                               #Converting from type 'object' to type 'int32' for models to recognize.

        self.helper.start_timer()
        train_accuracy, convMatrix_train, miss_count = self.classification_model.train(X_train, y_train)
        time_elapsed = self.helper.elapsed_time()
        print("Train Output\n", "Accuracy:", train_accuracy, '\n', convMatrix_train, '\n', "Elapsed Time:", time_elapsed )
        HeatMap.show(convMatrix_train)


        # Test Process.
        if len(data_test) > 0:
            cleaned_data_test = self.preprocess_obj.start_for_testing(data_test)
            copy_cleaned_data_test = cleaned_data_test
            y_test = copy_cleaned_data_test[self.label]                   # Test_Target
            X_test = copy_cleaned_data_test.drop([self.label], axis=1)    # Test_Input
            y_test  = y_test.astype('int')                                #Converting from type 'object' to type 'int32' for models to recognize.

            self.helper.start_timer()
            test_accuracy, convMatrix_test, miss_count  = self.classification_model.test(X_test, y_test)
            time_elapsed = self.helper.elapsed_time()
            print("Test Output\n", "Accuracy:", test_accuracy, '\n', convMatrix_test, '\n', "Elapsed Time:", time_elapsed)
            HeatMap.show(convMatrix_test)


        #graph = Graph(X_train, y_train, cleaned_data_train, self.label)
        #graph.feature_VS_feature_points()

    def final_test(self, file_name):
        final_test_data = pd.read_csv(file_name)
        cleaned_data_test = self.preprocess_obj.start_for_testing(final_test_data)  # Test

        y_test = cleaned_data_test[self.label]  # Test_Target
        X_test = cleaned_data_test.drop([self.label], axis=1)  # Test_Input

        self.helper.start_timer()
        test_accuracy, convMatrix_test, miss_count = self.classification_model.test_for_saved_model(X_test, y_test)
        time_elapsed = self.helper.elapsed_time()
        print("Test Output\n", "Accuracy:", test_accuracy, '\n', convMatrix_test, '\n', "Elapsed Time:", time_elapsed)
        HeatMap.show(convMatrix_test)
