from model import *
from graph import *
from emo_utils import *
from tokenizer import *
from dataloader import *

def to_one_hot(y, num):
    Y = np.zeros((len(y), num))
    Y[np.arange(len(y)), y] = 1
    return Y

glove_50, glove_100 = read_glove_vecs('glove.6B.50d.txt'), read_glove_vecs('glove.6B.100d.txt')

words_to_index_50, _, word_to_vec_map_50 = read_glove_vecs('glove.6B.50d.txt')
words_to_index_100, _, word_to_vec_map_100 = read_glove_vecs('glove.6B.100d.txt')

roX_train, roY_train = read_csv('train_emoji.csv')
roX_test, roY_test = read_csv('test_emoji.csv')
X_train, Y_train = tokenize(roX_train), to_one_hot(roY_train, 5)
X_test, Y_test = tokenize(roX_test), to_one_hot(roY_test, 5)
X_train_50 = np.zeros((len(X_train), 15), dtype=int)
X_train_100 = np.zeros((len(X_train), 15), dtype=int)
X_test_50 = np.zeros((len(X_test), 15), dtype=int)
X_test_100 = np.zeros((len(X_test), 15), dtype=int)
for i, tokens in enumerate(X_train):
    for j, token in enumerate(tokens[:15]):
        X_train_50[i, j] = words_to_index_50[token]
        X_train_100[i, j] = words_to_index_100[token]
for i, tokens in enumerate(X_test):
    for j, token in enumerate(tokens[:15]):
        X_test_50[i, j] = words_to_index_50[token]
        X_test_100[i, j] = words_to_index_100[token]

train_set_50 = DataLoader(X_train_50, Y_train, batch_size=16, shuffle=True)
test_set_50 = DataLoader(X_test_50, Y_test, batch_size=56, shuffle=False)
train_set_100 = DataLoader(X_train_100, Y_train, batch_size=16, shuffle=True)
test_set_100 = DataLoader(X_test_100, Y_test, batch_size=56, shuffle=False)

print("Data ready")

'''
0: Vanila + SGD + 50
1: LSTM + SGD + 50
2: LSTM + ADAM + 50
3: LSTM + SGD + 100
4: LSTM + SGD + Dropout
'''
layers = [[
        Embedding(words_to_index_50, word_to_vec_map_50),
        VanilaRecurrent(input_dim=50, hidden_dim=48, return_sequences=True),
        VanilaRecurrent(input_dim=48, hidden_dim=48, return_sequences=False),
        Dense(input_dim=48, output_dim=5),
    ], [
        Embedding(words_to_index_50, word_to_vec_map_50),
        LSTM(input_dim=50,  hidden_dim=48, return_sequences=True),
        LSTM(input_dim=48,  hidden_dim=48, return_sequences=False),
        Dense(input_dim=48, output_dim=5),
    ], [
        Embedding(words_to_index_50, word_to_vec_map_50),
        LSTM(input_dim=50,  hidden_dim=48, return_sequences=True),
        LSTM(input_dim=48,  hidden_dim=48, return_sequences=False),
        Dense(input_dim=48, output_dim=5),
    ], [
        Embedding(words_to_index_100, word_to_vec_map_100),
        LSTM(input_dim=100, hidden_dim=48, return_sequences=True),
        LSTM(input_dim=48, hidden_dim=48, return_sequences=False),
        Dense(input_dim=48, output_dim=5),
    ], [
        Embedding(words_to_index_50, word_to_vec_map_50),
        LSTM(input_dim=50,  hidden_dim=48, return_sequences=True),
        Dropout(dropout_rate=0.5),
        LSTM(input_dim=48, hidden_dim=48, return_sequences=False),
        Dropout(dropout_rate=0.5),
        Dense(input_dim=48, output_dim=5),
    ]
]

models = [
    Model(layers[0], SoftmaxCrossEntropy(), SGD(lr=0.03)),
    Model(layers[1], SoftmaxCrossEntropy(), SGD(lr=0.4)),
    Model(layers[2], SoftmaxCrossEntropy(), Adam(learning_rate=0.001)),
    Model(layers[3], SoftmaxCrossEntropy(), SGD(lr=0.6)),
    Model(layers[4], SoftmaxCrossEntropy(), SGD(lr=0.6)),
]

print("Models ready")

loss_graphs = [LossGraph() for _ in range(5)]
conf_mats = [ConfusionMatrix() for _ in range(5)]

for epoch in range(250):
    for idx, model in enumerate(models):
        train_set = train_set_50
        test_set = test_set_50
        if idx == 3:
            train_set = train_set_100
            test_set = test_set_100

        total_loss, total_samples = 0.0, 0
        for Xb, Yb in train_set:
            loss = model.forward(Xb, Yb)
            model.backward()

            total_loss += loss * Xb.shape[0]
            total_samples += Xb.shape[0]

        loss_graphs[idx].train_losses.append(total_loss / total_samples)

        total_loss, total_samples = 0.0, 0
        for Xb, Yb in test_set:
            loss = model.forward(Xb, Yb, is_train=False)

            total_loss += loss * Xb.shape[0]
            total_samples += Xb.shape[0]

        loss_graphs[idx].test_losses.append(total_loss / total_samples)
        
    print(f"epoch {epoch} trained & tested")

for idx, model in enumerate(models):
    total_correct = 0
    total_samples = 0
    test_set = test_set_50
    if idx == 3:
        test_set = test_set_100
    for Xb, Yb in test_set:
        model.forward(Xb, Yb, is_train=False)
        probs = model.evaluate.cache["probs"]
        preds = probs.argmax(axis=1)

        labels = Yb.argmax(axis=1)

        total_correct += np.sum(preds == labels)
        total_samples += labels.shape[0]

        for t, p in zip(labels, preds):
            conf_mats[idx].mat[t, p] += 1

    acc = total_correct / total_samples
    print(f"Model {idx} Accuracy: {acc:.4f}")

for idx in range(5):
    loss_graphs[idx].show()
    conf_mats[idx].show()