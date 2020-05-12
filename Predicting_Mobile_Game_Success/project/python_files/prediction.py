import pandas as pd
from sklearn.model_selection import train_test_split
from python_files.preprocessing import PredictionPreprocess
from python_files.learning_models import PredictModel
from python_files.correlation import Correlation
from python_files.helper_functions import Helper

class Program:
    def __init__(self, file_name, model_id, train_size=0.8):
        self.train_size = train_size
        self.helper = Helper()
        self.label = 'Average User Rating'  #label, output, y

        self.data  = pd.read_csv(file_name) #data -> dataframe
        self.linearRegression = PredictModel(model_id) #1 for multivariate, 2 for polynomial
        self.preprocess_obj = PredictionPreprocess(self.label, train_size*10)

    def start(self):
        if self.train_size == 1:    #Train with all data.
            data_train = self.data
            data_test = []
        else:
            data_train, data_test = train_test_split(self.data, test_size=1-self.train_size, shuffle=True)

        # Train Process.
        cleaned_data_train = self.preprocess_obj.start_for_training(data_train)
        copy_cleaned_data_train = cleaned_data_train
        y_train = copy_cleaned_data_train[self.label]                  # Train_Target
        X_train = copy_cleaned_data_train.drop([self.label], axis=1)   # Train_Input

        self.helper.start_timer()
        train_error, train_r2_score = self.linearRegression.train(X_train, y_train)
        elapsed_time = self.helper.elapsed_time()
        print("Train Output\n", "MSE:", train_error, '\n', "R2 Score:", train_r2_score, "\n", "Elapsed Time:", elapsed_time)


        # Test Process.
        if len(data_test) > 0:
            cleaned_data_test = self.preprocess_obj.start_for_testing(data_test)     # Test
            copy_cleaned_data_test  = cleaned_data_test
            y_test  = copy_cleaned_data_test[self.label]                   # Test_Target
            X_test  = copy_cleaned_data_test.drop([self.label], axis=1)    # Test_Input

            self.helper.start_timer()
            test_error, test_r2_score = self.linearRegression.test(X_test, y_test)
            elapsed_time = self.helper.elapsed_time()
            print("Test Output\n", "MSE:", test_error, '\n', "R2 Score:", test_r2_score, "\n", "Elapsed Time:", elapsed_time)


        corr = Correlation(cleaned_data_train)
        corr.correlate()

    def final_test(self, file_name):
        final_test_data  = pd.read_csv(file_name)
        cleaned_data_test = self.preprocess_obj.start_for_testing(final_test_data)  # Test

        y_test = cleaned_data_test[self.label]  # Test_Target
        X_test = cleaned_data_test.drop([self.label], axis=1)  # Test_Input

        self.helper.start_timer()
        test_error, test_r2_score = self.linearRegression.test_for_saved_model(X_test, y_test)
        elapsed_time = self.helper.elapsed_time()

        print("Test Output\n", "MSE:", test_error, '\n', "R2 Score:", test_r2_score, "\n", "Elapsed Time:", elapsed_time)
