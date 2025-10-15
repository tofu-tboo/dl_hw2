import numpy as np

def _sigmoid(u):
    return 1.0 / (1.0 + np.exp(-u))

class Layer:
    def __init__(self):
        self.params = {}
        self.grads = {}
        self.cache = {}

    def forward(self, x, is_train=True):
        pass

    def backward(self, dy):
        pass

class Embedding(Layer):
    def __init__(self, words_to_index, word_to_vec_map):
        super().__init__()
        vocab_size = len(word_to_vec_map) + 1 # 1 for padding
        self.embedding_dim = next(iter(word_to_vec_map.values())).shape[0]

        W = np.zeros((vocab_size, self.embedding_dim))
        rng = np.random.default_rng()

        for word, idx in words_to_index.items():
            vec = word_to_vec_map.get(word)
            if vec is not None:
                W[idx, :] = vec
            else: # missing => random
                W[idx, :] = 0.01 * rng.standard_normal(self.embedding_dim)

        self.params["W"] = W

    def forward(self, x, is_train=True): # x is index matrix
        self.cache["x"] = x
        return self.params["W"][x]

    def backward(self, dy):
        x = self.cache["x"]
        dW = np.zeros_like(self.params["W"])
        np.add.at(dW, x, dy)
        dW[0, :] = 0
        self.grads["W"] = dW
        return np.zeros_like(x)

class Dropout(Layer):
    def __init__(self, dropout_rate=0.5):
        super().__init__()
        self.dropout_rate = dropout_rate

    def forward(self, x, is_train=True):
        if (not is_train) or self.dropout_rate <= 0:
            self.cache["mask"] = None
            return x

        keep_prob = 1.0 - self.dropout_rate
        mask = (np.random.rand(*x.shape) < keep_prob) / keep_prob
        self.cache["mask"] = mask
        return x * mask

    def backward(self, dy):
        mask = self.cache["mask"]
        return dy if mask is None else dy * mask

class Dense(Layer):
    def __init__(self, input_dim, output_dim):
        super().__init__()

        rng = np.random.default_rng()
        self.params["W"] = np.sqrt(2.0 / input_dim) * rng.standard_normal(size=(input_dim, output_dim))
        self.params["b"] = np.zeros((1, output_dim))

    def forward(self, x, is_train=True):
        if x.shape[1] != self.params["W"].shape[0]:
            x = x.reshape(x.shape[0], -1)
        self.cache["x"] = x
        return x @ self.params["W"] + self.params["b"]

    def backward(self, dy):
        x = self.cache["x"]
        self.grads["W"] = x.T @ dy
        self.grads["b"] = dy.sum(axis=0, keepdims=True)
        return dy @ self.params["W"].T

class VanilaRecurrent(Layer):
    def __init__(self, input_dim, hidden_dim, return_sequences=False):
        super().__init__()
        self.hidden_dim = hidden_dim
        self.return_sequences = return_sequences

        rng = np.random.default_rng()
        self.params["W"] = np.sqrt(1.0/(input_dim + hidden_dim)) * rng.standard_normal(size=(input_dim + hidden_dim, hidden_dim))
        self.params["b"] = np.zeros((1, hidden_dim))

    def forward(self, x, is_train=True):
        batch, words, _ = x.shape
        weights = self.params["W"]
        bias = self.params["b"]

        hidden_prev = np.zeros((batch, self.hidden_dim))
        hidden_list = []
        self.cache["x"] = x
        self.cache["hidden_prev_list"] = []
        self.cache["pre_activation_list"] = []

        for t in range(words):
            x_t = x[:, t, :]
            concat = np.concatenate([x_t, hidden_prev], 1)
            z_t = concat @ weights + bias
            h_t = np.tanh(z_t)

            self.cache["hidden_prev_list"].append(hidden_prev)
            self.cache["pre_activation_list"].append(z_t)
            hidden_prev = h_t
            hidden_list.append(h_t)

        hidden_stack = np.stack(hidden_list, axis=1)
        self.cache["hidden_stack"] = hidden_stack
        return hidden_stack if self.return_sequences else hidden_stack[:, -1, :]

    def backward(self, dy):
        x = self.cache["x"]
        batch, words, D = x.shape
        weights = self.params["W"]

        if dy.ndim == 2:
            dy_full = np.zeros((batch, words, self.hidden_dim))
            dy_full[:, -1, :] = dy
            dy = dy_full

        dW = np.zeros_like(weights)
        db = np.zeros_like(self.params["b"])
        dx = np.zeros_like(x)
        grad_hidden_next = np.zeros((batch, self.hidden_dim))

        for t in reversed(range(words)):
            z_t = self.cache["pre_activation_list"][t]
            h_prev = self.cache["hidden_prev_list"][t]
            x_t = x[:, t, :]

            dh = dy[:, t, :] + grad_hidden_next
            h_t = np.tanh(z_t)
            dz = dh * (1.0 - h_t * h_t)

            concat = np.concatenate([x_t, h_prev], 1)
            dW += concat.T @ dz
            db += dz.sum(axis=0, keepdims=True)

            dconcat = dz @ weights.T
            dx[:, t, :] = dconcat[:, :D]
            grad_hidden_next = dconcat[:, D:]

        self.grads["W"] = dW
        self.grads["b"] = db
        return dx

class LSTM(Layer):
    def __init__(self, input_dim, hidden_dim, return_sequences=False):
        super().__init__()
        self.hidden_dim = hidden_dim
        self.return_sequences = return_sequences

        rng = np.random.default_rng()
        self.params["W"] = np.sqrt(1.0 / (input_dim + hidden_dim)) * rng.standard_normal(size=(input_dim + hidden_dim, 4 * hidden_dim)
        )
        self.params["b"] = np.zeros((1, 4 * hidden_dim))

    def forward(self, x, is_train=True):
        batch, words, _ = x.shape
        weights = self.params["W"]
        bias = self.params["b"]

        hidden_prev = np.zeros((batch, self.hidden_dim))
        cell_prev = np.zeros((batch, self.hidden_dim))

        hidden_states = []
        # 캐시 박스
        self.cache = {"x": x, "hidden_prev": [], "cell_prev": [],
             "i": [], "f": [], "o": [], "g": [], "cell": [], "hidden": []}

        for t in range(words):
            x_t = x[:, t, :]
            concat = np.concatenate([x_t, hidden_prev], axis=1)
            z = concat @ weights + bias
            z_i, z_f, z_o, z_g = np.split(z, 4, axis=1)

            i = _sigmoid(z_i)
            f = _sigmoid(z_f)
            o = _sigmoid(z_o)
            g = np.tanh(z_g)

            cell = f * cell_prev + i * g
            hidden = o * np.tanh(cell)

            self.cache["hidden_prev"].append(hidden_prev)
            self.cache["cell_prev"].append(cell_prev)
            self.cache["i"].append(i)
            self.cache["f"].append(f)
            self.cache["o"].append(o)
            self.cache["g"].append(g)
            self.cache["cell"].append(cell)
            self.cache["hidden"].append(hidden)

            hidden_prev = hidden
            cell_prev = cell
            hidden_states.append(hidden)

        hidden_stack = np.stack(hidden_states, axis=1)
        self.cache["hidden_stack"] = hidden_stack

        return hidden_stack if self.return_sequences else hidden_stack[:, -1, :]

    def backward(self, dy):
        x = self.cache["x"]
        batch, words, D = x.shape
        weights = self.params["W"]

        # upstream 정규화
        if dy.ndim == 2:
            grad_full = np.zeros((batch, words, self.hidden_dim))
            grad_full[:, -1, :] = dy
            dy = grad_full

        dW = np.zeros_like(weights)
        db = np.zeros_like(self.params["b"])
        dx = np.zeros_like(x)
        grad_hidden_next = np.zeros((batch, self.hidden_dim))
        grad_cell_next = np.zeros((batch, self.hidden_dim))

        for t in reversed(range(words)):
            i = self.cache["i"][t]
            f = self.cache["f"][t]
            o = self.cache["o"][t]
            g = self.cache["g"][t]
            cell = self.cache["cell"][t]
            cell_prev = self.cache["cell_prev"][t]
            hidden_prev = self.cache["hidden_prev"][t]
            x_t = x[:, t, :]

            grad_hidden = dy[:, t, :] + grad_hidden_next
            grad_o = grad_hidden * np.tanh(cell)
            grad_cell = grad_hidden * o * (1 - np.tanh(cell) ** 2) + grad_cell_next
            grad_i = grad_cell * g
            grad_f = grad_cell * cell_prev
            grad_g = grad_cell * i
            grad_cell_next = grad_cell * f

            dz_i = grad_i * i * (1 - i)
            dz_f = grad_f * f * (1 - f)
            dz_o = grad_o * o * (1 - o)
            dz_g = grad_g * (1 - g ** 2)
            dz = np.concatenate([dz_i, dz_f, dz_o, dz_g], axis=1)  # (N,4H)

            concat = np.concatenate([x_t, hidden_prev], axis=1)    # (N,D+H)
            dW += concat.T @ dz
            db += dz.sum(axis=0, keepdims=True)

            dconcat = dz @ weights.T
            dx[:, t, :] = dconcat[:, :D]
            grad_hidden_next = dconcat[:, D:]

        self.grads["W"] = dW
        self.grads["b"] = db
        return dx

class ActiveFunc(Layer):
    def __init__(self):
        super().__init__()

class ReLU(ActiveFunc):
    def forward(self, x, is_train=True):
        mask = x > 0
        self.cache["mask"] = mask # for backward
        return x * mask # False entry as 0

    def backward(self, dy):
        return dy * self.cache["mask"]

class Tanh(ActiveFunc):
    def forward(self, x, is_train=True):
        output = np.tanh(x)
        self.cache["output"] = output
        return output
    def backward(self, dy):
        output = self.cache["output"]
        return dy * (1.0 - output * output)

class SoftmaxCrossEntropy: # Combine Softmax and CE => Easy calculation
    def __init__(self):
        self.cache = {}

    def forward(self, y_hat, y):
        z = y_hat - y_hat.max(axis=1, keepdims=True) # prevent overflow
        exp = np.exp(z)
        probs = exp / exp.sum(axis=1, keepdims=True)
        probs = np.clip(probs, 1e-12, 1.0)
        
        self.cache["probs"] = probs
        self.cache["y"] = y

        loss = -np.mean(np.sum(y * np.log(probs), axis=1))
        return loss

    def backward(self):
        probs = self.cache["probs"]
        y = self.cache["y"]
        N = probs.shape[0]

        return (probs - y) / N

class Optimizer:
    def __init__(self, lr = 0.01):
        self.learning_rate = lr

    def step(self):
        pass

    def zero_grad(self, layers):
        for _, grads in layers:
            grads[...] = 0

class SGD(Optimizer):
    def __init__(self, lr=0.01, weight_decay=0.0):
        super().__init__(lr)
        self.weight_decay = weight_decay

    def step(self, layers):
        for params, grads in layers:
            if self.weight_decay > 0:
                grads = grads + self.weight_decay * params
            params -= self.learning_rate * grads

class Adam(Optimizer):
    def __init__(self, learning_rate=0.001, beta1=0.9, beta2=0.999, epsilon=1e-8, weight_decay=0.0):
        super().__init__(learning_rate)
        self.beta1 = beta1
        self.beta2 = beta2
        self.epsilon = epsilon
        self.weight_decay = weight_decay
        self.time_step = 0
        self.first_moment = {}
        self.second_moment = {}

    def step(self, layers):
        self.time_step += 1
        for idx, (parameter, gradient) in enumerate(layers):
            if idx not in self.first_moment:
                self.first_moment[idx] = np.zeros_like(parameter)
                self.second_moment[idx] = np.zeros_like(parameter)

            m = self.first_moment[idx]
            v = self.second_moment[idx]

            if self.weight_decay > 0:
                gradient = gradient + self.weight_decay * parameter

            m = self.beta1 * m + (1 - self.beta1) * gradient
            v = self.beta2 * v + (1 - self.beta2) * (gradient * gradient)

            m_hat = m / (1 - self.beta1 ** self.time_step)
            v_hat = v / (1 - self.beta2 ** self.time_step)

            parameter -= self.learning_rate * m_hat / (np.sqrt(v_hat) + self.epsilon)

            self.first_moment[idx] = m
            self.second_moment[idx] = v

class Model:
    def __init__(self, layers, evaluate, optimizer):
        self.layers = layers
        self.evaluate = evaluate
        self.parameters = []
        self.optimizer = optimizer

    def forward(self, x, y, is_train=True):
        for layer in self.layers:
            x = layer.forward(x, is_train)
        return self.evaluate.forward(x, y)

    def backward(self):
        grad = self.evaluate.backward()
        for layer in reversed(self.layers):
            grad = layer.backward(grad)

        self.vectorize_parameters()

        # for _, grads in self.parameters:
            # np.clip(grads, -1.0, 1.0, out=grads)

        self.optimizer.step(self.parameters)
        self.optimizer.zero_grad(self.parameters)

    def vectorize_parameters(self): # vectorize parameters and gradients
        self.parameters.clear()
        for layer in self.layers:
            for k in layer.params.keys():
                self.parameters.append((layer.params[k], layer.grads[k]))
