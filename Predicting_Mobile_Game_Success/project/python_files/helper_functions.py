import pickle
import time

class Helper:
    def __init__(self):
        self.filename = 'finalized_model.sav'

    def save_model(self, model, technique, model_id):
        pickle.dump(model, open('../output_saved_models/' + str(technique) + str(model_id) + self.filename, 'wb'))

    def retreive_model(self, technique, model_id):
        loaded_model = pickle.load(open('../output_saved_models/' + str(technique) + str(model_id) + self.filename, 'rb'))
        return loaded_model

    def save_structure(self, structure, name):
        with open('../structures/' + name + '.pickle', 'wb') as handle:
            pickle.dump(structure, handle, protocol=pickle.HIGHEST_PROTOCOL)

    def load_structure(self, name):
        with open('../structures/' + name + '.pickle', 'rb') as handle:
            structure = pickle.load(handle)

        return structure

    def start_timer(self):
        self.start_time = time.time()

    def elapsed_time(self):
        end_time = time.time()
        return end_time - self.start_time