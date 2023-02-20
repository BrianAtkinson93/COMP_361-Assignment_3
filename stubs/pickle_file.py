import pickle


def load_data(file_path):
    # Load the pickle file
    cost = {}
    with open(file_path, 'rb') as picklefile:
        cost = pickle.load(picklefile)
    return cost


def dump_data(COST):
    # Write the data to a pickle file
    with open('resources/distances.pickle', 'wb') as picklefile:
        pickle.dump(COST, picklefile)
