import pickle

def handle_model(model_dir,save_or_pick,model=None):
    if save_or_pick == "save":
        with open(model_dir, 'wb') as file:
            pickle.dump(model, file)
    elif save_or_pick == "pick":
        with open(model_dir, 'rb') as file:
            model = pickle.load(file)
    return model

