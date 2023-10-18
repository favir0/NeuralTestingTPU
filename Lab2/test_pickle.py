import pickle
weights = {1: [0,1,2,3,5], 2: [0,1,2,3,5], 3: [0,1,2,3,5]}

def save_weights(weights):
    with open('dump.dat', 'wb') as dump_out:
        pickle.dump(weights, dump_out)

def load_weights():
    with open('dump.dat', 'rb') as dump_in:
        res = pickle.load(dump_in)
        return res


save_weights(weights)
res = load_weights()
for key, value in res.items():
    print(key, value)

res2 = load_weights()
for key, value in res2.items():
    print(key, value)


weights2 = {1: [0,1,2,3,5], 2: [0,1,2,3,5], 3: [0,1,2,3,5], 4: [0,1,2,3,5]}

save_weights(weights2)
res3 = load_weights()
for key, value in res3.items():
    print(key, type(key), value, type(value))