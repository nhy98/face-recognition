import numpy as np
import tensorflow as tf
from tensorflow.keras.layers import Dense, Lambda
from tensorflow.keras.models import Model
from tensorflow.keras.applications import VGG16
import pickle
from sklearn.metrics.pairwise import cosine_similarity
from model.model_irse import IR_50,IR_SE_50, l2_norm
import torch
from sklearn.svm import SVC

def _save_pickle(obj, file_path):
  with open(file_path, 'wb') as f:
    pickle.dump(obj, f)

def _load_pickle(file_path):
  with open(file_path, 'rb') as f:
    obj = pickle.load(f)
  return obj

def _base_network():
  model = VGG16(include_top = True, weights = None)
  dense = Dense(128)(model.layers[-4].output)
  norm2 = Lambda(lambda x: tf.math.l2_normalize(x, axis = 1))(dense)
  model = Model(inputs = [model.input], outputs = [norm2])
  return model

def _classifier_model(emb_array, labels):
  model = SVC(kernel='linear', probability=True)
  model.fit(emb_array, labels)

def _most_similarity3(embed_vecs, vec, labels):
  cos = torch.nn.CosineSimilarity(dim=1, eps=1e-6)
  max_sim, max_labels = 0, labels[0]
  for i in range(len(embed_vecs)):
    train_vec = embed_vecs[i].reshape(1, -1)
    sim = cos(l2_norm(train_vec), l2_norm(vec))
    if sim > max_sim:
      max_sim = sim
      max_labels = labels[i]

  return max_labels
def _most_similarity(embed_vecs, vec, labels):
  sim = cosine_similarity(embed_vecs, vec)
  # cos = torch.nn.CosineSimilarity(dim=1, eps=1e-6)
  # sim = cos(l2_norm(embed_vecs), l2_norm(vec))
  sim = np.squeeze(sim, axis = 1)
  print(f"sim: {sim}")
  argmax = np.argwhere(sim > 0.4)[::-1][:1]
  # argmax = np.argsort(sim.numpy())[::-1][:1]
  if len(argmax) == 0:
    return "Unknown"
  # label = [labels[idx] for idx in argmax][0]
  label = [labels[idx] for idx in argmax[0]][0]
  return label

def _most_similarity2(vec, labels):
  predictions = model2.predict_proba(vec)
  print(f"Prediction: {predictions}")
  best_class_indices = np.argwhere(predictions > 0.3)[::-1][:1]
  if len(best_class_indices) == 0:
    return "Unknown"
  label = [labels[idx] for idx in best_class_indices[0]][0]
  return label

def _normalize_image(image, epsilon=0.000001):
  means = np.mean(image.reshape(-1, 3), axis=0)
  stds = np.std(image.reshape(-1, 3), axis=0)
  image_norm = image - means
  image_norm = image_norm/(stds + epsilon)
  return image_norm

# model = _base_network()
# model.summary()

# model.load_weights("model/model_triplot_cv2.h5")
X_train = _load_pickle("data/faces1124.pkl")
y_train = _load_pickle("data/y_labels1124.pkl")

model = IR_SE_50([112,112])
# model.load_state_dict(torch.load("model/backbone_ir50_ms1m_epoch63.pth"))
model.load_state_dict(torch.load("model/model_ir_se50.pth"))
model.eval()
print("abc")

# X_train_vec = model.predict(np.stack(X_train))
X_train_ten = torch.Tensor(X_train)
print(f"Shape: {X_train_ten.shape}")
X_train_ten = X_train_ten.reshape(len(X_train), 3, 112, 112)
X_train_vec = model(X_train_ten)

with torch.no_grad():
  model2 = SVC(kernel='linear', probability=True)
  model2.fit(X_train_vec, y_train)
  print("done")