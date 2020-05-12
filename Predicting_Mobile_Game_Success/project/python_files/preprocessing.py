import numpy as np
import pandas as pd
import os.path
from python_files.helper_functions import Helper


class Preprocess:
    def __init__(self, label, to_be_dropped, to_be_encoded, to_be_filled_0,
                         to_be_filled_by_average, to_be_encoded_ind, to_be_encoded_dates, to_be_hot_encoded, file_id):
        self.file_id = file_id
        self.label = label              #label, y, output
        self.dict_min_max = {}          #Min and max value for each feature.
        self.dict_average = {}          #Average value for each feature.

        self.to_be_dropped = to_be_dropped
        self.to_be_encoded = to_be_encoded
        self.to_be_filled_0 = to_be_filled_0
        self.to_be_filled_by_average = to_be_filled_by_average
        self.to_be_encoded_ind = to_be_encoded_ind
        self.to_be_encoded_dates = to_be_encoded_dates
        self.to_be_hot_encoded = to_be_hot_encoded

    def start_for_training(self, train_data):
        self.data = train_data

        file_path = 'l1_'+str(self.file_id)+'.pickle'      #At least one exists
        if os.path.isfile(file_path):
            self.start_for_testing(train_data)        #Don't go through the preprocessing process again.
        else:
            self.feature_selection()
            self.data_cleaning()            #Row Based    -> Remove row when output field is empty.
            self.encoding_for_training()
            self.data_cleaning_part2()      #Column Based -> Calculate average for empty cells.
            self.data_scaling()

            self.data.to_csv('../output/train_output.csv')
            self.save_structures()

        return self.data
    def start_for_testing(self, test_data):
        self.data = test_data
        self.load_structures()

        self.feature_selection()
        self.data_cleaning()
        self.encoding_for_testing()
        self.update_with_average_value()         #Remove any empty cells.
        self.update_with_scaled_min_max_value()  #All values range between (0, 1).

        self.data.to_csv('../output/test_output.csv')

        for column in self.data.columns:
            if not column in self.list_all_unique:
                print(column)

        return self.data

    #Step (1) Choosing features.
    def feature_selection(self):
        # 1. Fill empty cells in 'In-app Purchases' with 0
        for feature in self.to_be_filled_0:
            self.data[feature].replace(to_replace=np.nan, value=0, inplace=True)

        # 2. Remove columns based on notice.
        self.data = self.data.drop(self.to_be_dropped, axis=1, inplace=False)  # axis = 1 -> drop by column

        # 3.Drop columns that have almost no data.
        # Remove any field that has more than 25% of the dataset missing values.
        at_lease_count = len(self.data) / 3
        self.data.dropna(axis=1, how='any', thresh=at_lease_count, inplace=True)

    #Step (2) Row Based: Remove row if output field is empty.
    def data_cleaning(self):
        # Drop rows that have any missing value in the label/output feature.
        self.data = self.data.dropna(axis=0, how="any", subset=[self.label], inplace=False)

    #Step (3) Data Cleaning by encoding. Calculating new weights.
    def encoding_for_training(self):
        self.one_hot_encoding()
        self.encoding_for_dates()
        #self.encoding_by_average()
    def encoding_for_testing(self):
        self.update_with_new_columns()
        self.encoding_for_dates()

    def one_hot_encoding(self):
        self.list_all_unique = []
        # Get New Columns Names.
        for feature in self.to_be_hot_encoded:
            unique_values = set()
            for i in range(0, len(self.data)):
                cell = self.data[feature].iloc[i]  # Equivalent to X[feature][i]
                if not pd.isnull(cell):
                    valuesList = cell.split(',')  # ex -> valuesList = [En, Fr, Da, SP]
                    for value in valuesList:
                        unique_values.add(value)

            for item in unique_values:
                self.list_all_unique.append(item)

        self.update_with_new_columns()
    def update_with_new_columns(self):
        # Add New Columns.
        updated_data = self.data
        updated_data = updated_data.reset_index(drop=True)
        for value in self.list_all_unique:
            new_column_arr = np.zeros(len(self.data))
            new_column_df = pd.Series(new_column_arr, name=value)
            updated_data = pd.concat([updated_data, new_column_df], axis=1)

        # Remove Old Columns
        for feature in self.to_be_hot_encoded:
            updated_data.drop(feature, axis=1, inplace=True)

        # Add Ones in the right columns.
        for feature in self.to_be_hot_encoded:
            for i in range(0, len(self.data)):
                cell = self.data[feature].iloc[i]
                if not pd.isnull(cell):
                    valuesList = cell.split(',')
                    for value in valuesList:
                        if value in updated_data: # Check if 'value' is seen in training operation.     #else: ignore it.
                            value_i = updated_data.columns.get_loc(value)
                            updated_data.iloc[i, value_i] = 1

        # Update.
        self.data = updated_data
    def encoding_for_dates(self):
        for date_feature in self.to_be_encoded_dates:
            for i in range(len(self.data)):
                cell = self.data[date_feature].iloc[i]
                if not pd.isnull(cell):
                    year = int(cell[-2:])
                    feature_i = self.data.columns.get_loc(date_feature)
                    self.data.iloc[i, feature_i] = year


    #Step (4) Data Cleaning. Column Based: Calculate average of column for empty cells.
    def data_cleaning_part2(self):
        self.build_dict_average()
        self.update_with_average_value()
    def build_dict_average(self):
        for feature in self.data.columns:
            sum_val = count_val = 0
            feature_i = self.data.columns.get_loc(feature)
            for i in range(len(self.data)):
                cell = self.data.iloc[i, feature_i]
                if not pd.isnull(cell):
                    sum_val += cell
                    count_val += 1

            self.dict_average[feature] = sum_val / count_val
    def update_with_average_value(self):
        for feature in self.data.columns:
            feature_i = self.data.columns.get_loc(feature)
            if feature in self.dict_average:
                average_val = self.dict_average[feature]
                for i in range(len(self.data)):
                    cell = self.data.iloc[i, feature_i]
                    if pd.isnull(cell):
                        self.data.iloc[i, feature_i] = average_val

    #Step (5) Let the whole dataset ranges between (0, 1).
    def data_scaling(self):
        self.build_dict_min_max()
        self.update_with_scaled_min_max_value()

    #Min Max Scalar Method
    def build_dict_min_max(self):
        for feature in self.data.columns:
            if not feature == self.label and feature not in self.list_all_unique: #Normalize for all but label column.
                column = self.data[feature]
                pair = (min(column), max(column))
                self.dict_min_max[feature] = pair
    def update_with_scaled_min_max_value(self):
        for feature in self.data.columns:
            if not feature == self.label and feature not in self.list_all_unique: #Normalize for all but label column.
                feature_i = self.data.columns.get_loc(feature)
                if feature in self.dict_min_max:
                    min_val, max_val = self.dict_min_max[feature]
                    if min_val == max_val: #Prevents divide by zero.
                        for i in range(len(self.data)):
                            self.data.iloc[i, feature_i] = 1
                    else:
                        for i in range(len(self.data)):
                            cell = self.data.iloc[i, feature_i]
                            self.data.iloc[i, feature_i] = (cell - min_val) / (max_val - min_val)

    def save_structures(self):
        helper = Helper()
        lists = [self.to_be_hot_encoded, self.to_be_encoded_dates, self.to_be_dropped, self.list_all_unique]

        dics = [self.dict_min_max,  self.dict_average]

        i = 1
        for item in lists:
            helper.save_structure(item, 'l' + str(i) + '_' + str(self.file_id))
            i += 1

        i = 1
        for item in dics:
            helper.save_structure(item, 'd' + str(i)+ '_' + str(self.file_id))
            i += 1

    def load_structures(self):
        helper = Helper()
        self.to_be_hot_encoded = helper.load_structure('l1' +'_'+ str(self.file_id))
        self.to_be_encoded_dates = helper.load_structure('l2'+'_'+ str(self.file_id))
        self.to_be_dropped = helper.load_structure('l3'+'_'+ str(self.file_id))
        self.list_all_unique = helper.load_structure('l4'+'_'+ str(self.file_id))

        self.dict_min_max = helper.load_structure('d1'+'_'+ str(self.file_id))
        self.dict_average = helper.load_structure('d2'+'_'+ str(self.file_id))

class PredictionPreprocess(Preprocess):
    def __init__(self, label, file_id):
        # Decision Criteria
        to_be_dropped = ['Subtitle', 'Developer', 'Primary Genre', 'Original Release Date', 'In-app Purchases', 'URL', 'ID', 'Name', 'Icon URL', 'Description']
        to_be_filled_0 = []
        to_be_filled_by_average = []
        to_be_encoded = []
        to_be_hot_encoded = ['Languages', 'Age Rating', 'Genres']
        to_be_encoded_ind = []  # Encode 'Age Rating' independently.
        to_be_encoded_dates = ['Current Version Release Date']

        super().__init__(label, to_be_dropped, to_be_encoded, to_be_filled_0,
                         to_be_filled_by_average, to_be_encoded_ind, to_be_encoded_dates, to_be_hot_encoded, file_id)

class ClassificationPreprocess(Preprocess):

    def __init__(self, label, classes, file_id):
        self.classes = classes
        # Decision Criteria
        to_be_dropped = ['Subtitle', 'Developer', 'Original Release Date', 'Primary Genre', 'In-app Purchases', 'URL', 'ID', 'Name', 'Icon URL', 'Description']
        to_be_filled_0 = []
        to_be_filled_by_average = []
        to_be_encoded = []
        to_be_encoded_ind = []  # Encode 'Age Rating' independently.
        to_be_encoded_dates = ['Current Version Release Date']
        to_be_hot_encoded = ['Languages', 'Age Rating', 'Genres']

        super().__init__(label, to_be_dropped, to_be_encoded, to_be_filled_0,
                         to_be_filled_by_average, to_be_encoded_ind, to_be_encoded_dates, to_be_hot_encoded, file_id)

    def start_for_training(self, train_data):
        code = 1
        for cls in self.classes:
            train_data[self.label].replace(to_replace=cls, value=code, inplace=True)
            code += 1

        return super().start_for_training(train_data)

    def start_for_testing(self, test_data):
        code = 1
        for cls in self.classes:
            test_data[self.label].replace(to_replace=cls, value=code, inplace=True)
            code += 1

        return super().start_for_testing(test_data)