from model import *
from graph import *
from emo_utils import *


'''
0: Vanila + SGD + 50
1: LSTM + SGD + 50
2: LSTM + ADAM + 50
3: LSTM + SGD + 100
4: LSTM + SGD + Dropout
'''
layers = [[
        Embedding(vocab_size=400001, embedding_dim=50),
        VanilaRecurrent(input_dim=50, hidden_dim=128, return_sequences=True),
        VanilaRecurrent(input_dim=128, hidden_dim=128, return_sequences=False),
        Dense(input_dim=128, output_dim=5),
    ], [
        Embedding(vocab_size=400001, embedding_dim=50),
        LSTM(input_dim=50,  hidden_dim=128, return_sequences=True),
        LSTM(input_dim=128, hidden_dim=128, return_sequences=False),
        Dense(input_dim=128, output_dim=5),
    ], [
        Embedding(vocab_size=400001, embedding_dim=50),
        LSTM(input_dim=50,  hidden_dim=128, return_sequences=True),
        LSTM(input_dim=128, hidden_dim=128, return_sequences=False),
        Dense(input_dim=128, output_dim=5),
    ], [
        Embedding(vocab_size=400001, embedding_dim=100),
        LSTM(input_dim=100, hidden_dim=128, return_sequences=True),
        LSTM(input_dim=128, hidden_dim=128, return_sequences=False),
        Dense(input_dim=128, output_dim=5),
    ], [
        Embedding(vocab_size=400001, embedding_dim=50),
        LSTM(input_dim=50,  hidden_dim=128, return_sequences=True),
        Dropout(dropout_rate=0.5),
        LSTM(input_dim=128, hidden_dim=128, return_sequences=False),
        Dropout(dropout_rate=0.5),
        Dense(input_dim=128, output_dim=5),
    ]
]

models = [
    Model(layers[0], SoftmaxCrossEntropy(), SGD(lr=0.01)),
    Model(layers[1], SoftmaxCrossEntropy(), SGD(lr=0.01)),
    Model(layers[2], SoftmaxCrossEntropy(), Adam(learning_rate=0.001)),
    Model(layers[3], SoftmaxCrossEntropy(), SGD(lr=0.01)),
    Model(layers[4], SoftmaxCrossEntropy(), SGD(lr=0.01)),
]

loss_graphs = [LossGraph() for _ in range(5)]
conf_mats = [ConfusionMatrix() for _ in range(5)]

for epoch in range(8):
    for model_idx, model in enumerate(models):
        if model_idx == 2:
            continue
        for Xb, Yb in train_set:
            loss = model.forward(Xb, Yb, is_train=True)
            model.backward()

            loss_graphs[model_idx].train_losses.append(loss)
    print("trained")
# TODO: Adam

for model_idx, model in enumerate(models):
    for Xb, Yb in test_set:
        model.forward(Xb, Yb, is_train=False)
        probs = model.evaluate.cache["probs"]
        preds = probs.argmax(axis=1)
        
        labels = Yb.argmax(axis=1)
        for t, p in zip(labels, preds):
            conf_mats[model_idx].mat[t, p] += 1
    print("tested")
# TODO: Adam

for i in range(5):
    loss_graphs[i].show()
    conf_mats[i].show()
    # overall accuracy도 숫자로 찍고 싶으면:
    acc = np.trace(conf_mats[i].mat) / np.maximum(conf_mats[i].mat.sum(), 1)
    print(f"Accuracy: {acc:.4f}")