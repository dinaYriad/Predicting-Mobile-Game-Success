from python_files import classification, prediction


def predict_main():
    print("Model: Linear Regression")
    print("\t At train size = 0.8")
    prediction1 = prediction.Program(file_name='../data/predicting_mobile_game_success_train_set.csv', model_id=1, train_size=0.8)
    prediction1.start()
    print("\t At train size = 1")
    prediction1 = prediction.Program(file_name='../data/predicting_mobile_game_success_train_set.csv', model_id=1, train_size=1)
    prediction1.start()
    print("\t Validation")
    prediction1.final_test("samples_mobile_game_success_test_set.csv")

    print()
    print()

    print("Model: Polynomial Regression")
    print("\t At train size = 0.8")
    prediction1 = prediction.Program(file_name='../data/predicting_mobile_game_success_train_set.csv', model_id=2, train_size=0.8)
    prediction1.start()
    print("\t At train size = 1")
    prediction1 = prediction.Program(file_name='../data/predicting_mobile_game_success_train_set.csv', model_id=2, train_size=1)
    #prediction1 = prediction.Program(file_name='../data/sample.csv', model_id=1, train_size=1)
    prediction1.start()

    print("\t Validation")

    prediction1.final_test("samples_mobile_game_success_test_set.csv")

    print()
    print("####################################################################")
    print()

def classify_main():
    models = ['Logistic Regression', 'SVM', 'KNN', "Decision Tree"]
    for model_i in range(4):
        print("Model:", models[model_i])
        for i in range(2):
            pca_mode = bool(i)
            print("PCA_Mode:", pca_mode)

            print("\t At train size = 0.8")
            classification1 = classification.Program(file_name='../data/appstore_games_classification.csv', model_id=model_i + 1, pca_mode=pca_mode, train_size=0.8)
            classification1.start()

            print("\t At train size = 1")
            classification1 = classification.Program(file_name='../data/appstore_games_classification.csv', model_id=model_i + 1, pca_mode=pca_mode, train_size=1)
            classification1.start()

            print("\t Validation")
            classification1.final_test('samples_predicting_mobile_game_success_test_set_classification.csv')

            print()

        print()
        print()
        print()

#Testing for prediction for now.
predict_main()
