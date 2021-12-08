get_ipython().magic('load_ext autoreload')
get_ipython().magic('autoreload 2')
get_ipython().magic('matplotlib inline')

import pandas as pd
import pickle
import matplotlib.cm as cm
from fastai import *
from fastai.text import *
from fastai.callbacks import *
from pathlib import Path
from sklearn.feature_extraction.text import CountVectorizer, TfidfVectorizer
from sklearn.pipeline import Pipeline
from sklearn.svm import LinearSVC
from sklearn.naive_bayes import MultinomialNB
from sklearn.model_selection import RandomizedSearchCV
from sklearn.metrics import confusion_matrix, classification_report, hamming_loss, zero_one_loss, accuracy_score
from sklearn.model_selection import train_test_split

seed = 42
# python RNG
import random
random.seed(seed)
# pytorch RNGs
import torch
torch.manual_seed(seed)
torch.backends.cudnn.deterministic = True
if torch.cuda.is_available(): torch.cuda.manual_seed_all(seed)
# numpy RNG
import numpy as np
np.random.seed(seed)

get_ipython().system('python -m fastai.utils.show_install')

bs=32

# ## Loading Data

path = Path(".")

df = pd.read_csv(path/"data/clean/train.csv")

df

df_train = df[df["is_valid"] == False]; df_train

df_test = df[df["is_valid"] == True]; df_test

df_lm = pd.read_csv(path/'data/clean/unsup/unsup.csv'); df_lm

X, y = df_train["text"].to_list(), df_train["label"].to_list()

X_train, X_valid, y_train, y_valid = train_test_split(X, y, test_size=0.2, random_state=42, stratify=y)

df_train = pd.DataFrame()
df_valid = pd.DataFrame()

df_train["label"], df_train["text"] = y_train, X_train

df_valid["label"], df_valid["text"] = y_valid, X_valid

df_train

df_valid

df_train.label.value_counts(), df_valid.label.value_counts()

df_train["is_valid"] = False
df_valid["is_valid"] = True

df_train_val = pd.concat([df_train, df_valid])

df_train_val.to_csv(path/"data/clean/train_val.csv", index=False)

df_train_val

tknzer = path/'models'
(tknzer/'tmp').ls()

get_ipython().system('pip install sentencepiece')

data_lm = (TextList.from_df(df_lm,cols='text', processor=SPProcessor.load(tknzer))
                           .split_by_rand_pct(0.2, seed=seed)
                           .label_for_lm()
                           .databunch(bs=bs))

data_lm.show_batch()

data_lm.save('data/data_lm_export.pkl')

config = awd_lstm_lm_config.copy()
config['qrnn'] = True
config['n_hid'] = 1550 #default 1152
config['n_layers'] = 4 #default 3

perplexity = Perplexity()
f1 = FBeta(beta=1, average="weighted")

lm_fns3 = ['pt_wt_sp15_multifit', 'pt_wt_vocab_sp15_multifit']
lm_fns3_bwd = ['pt_wt_sp15_multifit_bwd', 'pt_wt_vocab_sp15_multifit_bwd']

# ## Fine-tune forward LM

learn_lm = language_model_learner(data_lm, AWD_LSTM, path=path, config=config, pretrained_fnames=lm_fns3, drop_mult=1., 
                                  metrics=[error_rate, accuracy, perplexity])

learn_lm.save_encoder("no_fine_tune_enc")

learn_lm.predict("O Governo", n_words=20)

learn_lm.freeze()
learn_lm.lr_find()

learn_lm.recorder.plot()

lr = 1e-1

learn_lm.fit_one_cycle(2, lr, wd=0.1, moms=(0.8,0.7))

learn_lm.save('fine_tune_lm')
learn_lm.save_encoder('fine_tune_enc')

learn_lm.unfreeze()
learn_lm.lr_find()

learn_lm.recorder.plot()

lr = 1e-2

learn_lm.fit_one_cycle(10, lr, wd=0.1, moms=(0.8,0.7), callbacks=[ShowGraph(learn_lm)])

learn_lm.recorder.plot_lr()

learn_lm.predict("O Governo", n_words=30)

learn_lm.save('fine_tune_lm')
learn_lm.save_encoder('fine_tune_enc')

# ## Fine-tune backward LM

data_lm = (TextList.from_df(df_lm,cols='text', processor=SPProcessor.load(tknzer))
                           .split_by_rand_pct(0.2, seed=seed)
                           .label_for_lm()
                           .databunch(bs=bs, backwards=True))

data_lm.show_batch()

data_lm.save('./data/data_lm_back.pkl')

learn_lm = language_model_learner(data_lm, AWD_LSTM, path=path, config=config, pretrained_fnames=lm_fns3_bwd, drop_mult=1., 
                                  metrics=[error_rate, accuracy, perplexity])

learn_lm.save_encoder("no_fine_tune_enc_bwd")

learn_lm.lr_find()

learn_lm.recorder.plot()

lr = 1e-1

learn_lm.fit_one_cycle(2, lr, wd=0.1, moms=(0.8,0.7))

learn_lm.save('fine_tune_lm_bwd')
learn_lm.save_encoder('fine_tune_enc_bwd')

learn_lm.unfreeze()
learn_lm.lr_find()

learn_lm.recorder.plot()

lr=1e-2

learn_lm.fit_one_cycle(10, lr, wd=0.1, moms=(0.8,0.7), callbacks=[ShowGraph(learn_lm)])

learn_lm.save('fine_tune_lm_bwd')
learn_lm.save_encoder('fine_tune_enc_bwd')

# ## Train forward classifier

bs=8

data_lm = load_data("data/", "data_lm_export.pkl", bs=bs)

data_clas = (TextList.from_df(df_train_val, path, cols='text',
                              processor=SPProcessor.load(tknzer))
                         .split_from_df(col=2)
                         .label_from_df(cols=0)
                         .databunch(bs=bs))

data_clas = load_data(path/"data", "data_clas_export.pkl")

len(data_clas.vocab.itos), len(data_lm.vocab.itos)

data_clas.save(path/'data/data_clas_export.pkl')

data_clas.show_batch()

config = awd_lstm_clas_config.copy()
config['qrnn'] = True
config['n_hid'] = 1550 #default 1152
config['n_layers'] = 4 #default 3

learn_c = text_classifier_learner(data_clas, AWD_LSTM, config=config, pretrained=False, drop_mult=0.5, 
                                  metrics=[accuracy,f1])
learn_c.path = path
learn_c.load_encoder("fine_tune_enc");

learn_c.freeze()
learn_c.lr_find()

learn_c.recorder.plot()

lr=2e-2

learn_c.fit_one_cycle(10, lr, wd=0.6, moms=(0.8,0.7))

learn_c.save("clf_fwd")

learn_c.freeze_to(-2)
learn_c.fit_one_cycle(10, slice(lr/(2.6**4),lr), wd=0.6, moms=(0.8,0.7))

learn_c.save("clf_fwd")

learn_c.freeze_to(-3)
learn_c.fit_one_cycle(10, slice(lr/2/(2.6**4),lr/2), wd=0.6, moms=(0.8,0.7))

learn_c.save("clf_fwd")

learn_c.unfreeze()
learn_c.fit_one_cycle(10, slice(lr/10/(2.6**4),lr/10), wd=0.6, moms=(0.8,0.7))

learn_c.save("clf_fwd")

# ## Evaluate Forward CLF

data_clas = load_data(path/"data", "data_clas_export.pkl")

data_lm = load_data(path/"data", "data_lm_export.pkl")

data_test = (TextList.from_df(df_test, path, cols='text',
                              processor=SPProcessor.load(tknzer))
                         .split_none()
                         .label_from_df(cols=0)
                         .databunch(bs=bs))

data_test.show_batch()

data_test.c, data_clas.c

len(data_test.vocab.itos), len(data_clas.vocab.itos)

data_test.save(path/"data/test_data.pkl")

learn_c.path = path
learn_c.load(path/"clf_fwd");

learn_c.data.valid_dl = data_test.fix_dl

preds,y,losses = learn_c.get_preds(with_loss=True)
predictions = np.argmax(preds, axis = 1)
interp = ClassificationInterpretation(learn_c, preds, y, losses)
interp.plot_confusion_matrix()

predictions[:15], y[:15], predictions.shape, y.shape

print(classification_report(y, predictions, target_names=learn_c.data.classes, digits=4))
print(accuracy_score(y, predictions))

learn_c.show_results()

txt_ci = TextClassificationInterpretation.from_learner(learn_c)

txt_ci.show_top_losses(5)

# ## Train backwards classifier

bs=8

data_lm = load_data("data/", "data_lm_back.pkl", bs=bs, backwards=True)

data_lm.show_batch()

data_clas = (TextList.from_df(df_train_val, path, cols='text',
                              processor=SPProcessor.load(tknzer))
                         .split_from_df(col=2)
                         .label_from_df(cols=0)
                         .databunch(bs=bs, backwards=True))

len(data_clas.vocab.itos), len(data_lm.vocab.itos)

data_clas.save(path/'data/data_clas_bwd.pkl')

data_clas.show_batch()

config = awd_lstm_clas_config.copy()
config['qrnn'] = True
config['n_hid'] = 1550 #default 1152
config['n_layers'] = 4 #default 3

learn_c = text_classifier_learner(data_clas, AWD_LSTM, config=config, pretrained=False, drop_mult=0.5, 
                                  metrics=[accuracy,f1])
learn_c.load_encoder("fine_tune_enc_bwd");

learn_c.freeze()
learn_c.lr_find()

learn_c.recorder.plot()

lr=3e-2

learn_c.fit_one_cycle(10, lr, wd=0.6, moms=(0.8,0.7))

learn_c.save("clf_bwd")

learn_c.freeze_to(-2)
learn_c.fit_one_cycle(10, slice(lr/(2.6**4),lr), wd=0.6, moms=(0.8,0.7))

learn_c.save("clf_bwd")

learn_c.freeze_to(-3)
learn_c.fit_one_cycle(10, slice(lr/2/(2.6**4),lr/2), wd=0.6, moms=(0.8,0.7))

learn_c.save("clf_bwd")

learn_c.unfreeze()
learn_c.fit_one_cycle(10, slice(lr/10/(2.6**4),lr/10), wd=0.6, moms=(0.8,0.7))

learn_c.save("clf_bwd")

# ## Evaluate bwd clf

data_clas = load_data(path/"data", "data_clas_bwd.pkl", backwards=True)

data_test = (TextList.from_df(df_test, path, cols='text',
                              processor=SPProcessor.load(tknzer))
                         .split_none()
                         .label_from_df(cols=0)
                         .databunch(bs=bs, backwards=True))

data_test.show_batch()

data_test.c, data_clas.c

len(data_test.vocab.itos), len(data_clas.vocab.itos)

data_test.save(path/"data/test_data_bwd.pkl")

learn_c.load(path/"clf_bwd");

learn_c.data.valid_dl = data_test.fix_dl

preds,y,losses = learn_c.get_preds(with_loss=True)
predictions = np.argmax(preds, axis = 1)
interp = ClassificationInterpretation(learn_c, preds, y, losses)
interp.plot_confusion_matrix()

predictions[:15], y[:15], predictions.shape, y.shape

print(classification_report(y, predictions, target_names=learn_c.data.classes, digits=4))
print(accuracy_score(y, predictions))

learn_c.show_results()

txt_ci = TextClassificationInterpretation.from_learner(learn_c)

txt_ci.show_top_losses(5)

# ## Bwd + Fwd

bs = 8

config = awd_lstm_clas_config.copy()
config['qrnn'] = True
config['n_hid'] = 1550 #default 1152
config['n_layers'] = 4 #default 3

data_clas = load_data(path/"data", "data_clas_export.pkl", bs=bs, num_workers=1)
learn_c = text_classifier_learner(data_clas, AWD_LSTM, config=config, drop_mult=0.5, metrics=[accuracy,f1])
learn_c.path = path
learn_c.load("clf_fwd");

data_test = load_data(path/"data", "test_data.pkl", bs=bs)

learn_c.data.valid_dl = data_test.fix_dl

preds,targs = learn_c.get_preds(ordered=True)
accuracy(preds,targs)

data_clas_bwd = load_data(path/"data", "data_clas_bwd.pkl", bs=bs, num_workers=1, backwards=True)
learn_c_bwd = text_classifier_learner(data_clas_bwd, AWD_LSTM, config=config, drop_mult=0.5, metrics=[accuracy,f1]).to_fp16()
learn_c_bwd.path = path
learn_c_bwd.load("clf_bwd");

data_test_bwd = load_data(path/"data", "test_data_bwd.pkl", bs=bs, backwards=True)

learn_c_bwd.data.valid_dl = data_test_bwd.fix_dl

preds_bwd,targs_bwd = learn_c_bwd.get_preds(ordered=True)
accuracy(preds_bwd,targs_bwd)

preds_avg = (preds+preds_bwd)/2
accuracy(preds_avg, targs)

predictions = np.argmax(preds_avg, axis = 1)

print(classification_report(targs, predictions, target_names=learn_c.data.classes, digits=4))
print(accuracy_score(targs, predictions))

# ## Here we go again - with class weights

bs = 8

data_clas = load_data(path/"data", "data_clas_export.pkl", bs=bs)

n_samples = len(data_clas.train_ds.x); n_samples

n_classes = data_clas.c; n_classes

y = data_clas.train_ds.y.items; y

class_weights = n_samples / (n_classes * np.bincount(y)); class_weights

class_weights = 1 - np.bincount(y)/n_samples; trn_weights

config = awd_lstm_clas_config.copy()
config['qrnn'] = True
config['n_hid'] = 1550 #default 1152
config['n_layers'] = 4 #default 3

learn_c = text_classifier_learner(data_clas, AWD_LSTM, config=config, pretrained=False, drop_mult=0.5, 
                                  metrics=[accuracy,f1])
learn_c.path = path
learn_c.load_encoder("fine_tune_enc");

learn_c.loss_func

loss_weights = torch.FloatTensor(class_weights).cuda()
learn_c.loss_func = FlattenedLoss(CrossEntropyFlat, weight=loss_weights)

learn_c.freeze()
learn_c.lr_find()

learn_c.recorder.plot()

lr=2e-2

learn_c.fit_one_cycle(10, lr, wd=0.6, moms=(0.8,0.7))

learn_c.save("clf_fwd_weighted")

learn_c.freeze_to(-2)
learn_c.fit_one_cycle(10, slice(lr/(2.6**4),lr), wd=0.6, moms=(0.8,0.7))

learn_c.save("clf_fwd_weighted")

learn_c.freeze_to(-3)
learn_c.fit_one_cycle(10, slice(lr/2/(2.6**4),lr/2), wd=0.6, moms=(0.8,0.7))

learn_c.save("clf_fwd_weighted")

learn_c.unfreeze()
learn_c.fit_one_cycle(10, slice(lr/10/(2.6**4),lr/10), wd=0.6, moms=(0.8,0.7))

learn_c.save("clf_fwd_weighted")

data_clas = load_data(path/"data", "data_clas_export.pkl")

data_lm = load_data(path/"data", "data_lm_export.pkl")

data_test = load_data(path/"data", "test_data.pkl")

len(data_test.vocab.itos), len(data_clas.vocab.itos)

learn_c.load(path/"clf_fwd_weighted");

learn_c.data.valid_dl = data_test.fix_dl

learn_c.loss_func = FlattenedLoss(CrossEntropyFlat, weight=loss_weights.cpu())
preds,y,losses = learn_c.get_preds(with_loss=True)
predictions = np.argmax(preds, axis = 1)
interp = ClassificationInterpretation(learn_c, preds, y, losses)
interp.plot_confusion_matrix()

predictions[:15], y[:15], predictions.shape, y.shape

print(classification_report(y, predictions, target_names=learn_c.data.classes, digits=4))
print(accuracy_score(y, predictions))

# ## Ablation Study
# ### Pre-Trained LM + Fine-tune LM (No gradual_unfreeze)

bs = 8

data_clas = load_data(path/"data", "data_clas_export.pkl", bs=bs)

len(data_clas.vocab.itos)

config = awd_lstm_clas_config.copy()
config['qrnn'] = True
config['n_hid'] = 1550 #default 1152
config['n_layers'] = 4 #default 3

learn_c = text_classifier_learner(data_clas, AWD_LSTM, config=config, pretrained=False, drop_mult=0.5, 
                                  metrics=[accuracy,f1])
learn_c.path = path
learn_c.load_encoder("fine_tune_enc");

learn_c.unfreeze()
learn_c.lr_find()

learn_c.recorder.plot()

lr=1e-2

learn_c.fit_one_cycle(40, slice(lr/(2.6**4),lr), wd=0.6, moms=(0.8,0.7),
                      callbacks=[SaveModelCallback(learn_c, name="clf_fwd_no_gradual_unfreeze")])

data_clas = load_data(path/"data", "data_clas_export.pkl")

data_lm = load_data(path/"data", "data_lm_export.pkl")

data_test = load_data(path/"data", "test_data.pkl")

len(data_test.vocab.itos), len(data_clas.vocab.itos)

learn_c.load(path/"clf_fwd_no_gradual_unfreeze");

learn_c.data.valid_dl = data_test.fix_dl

preds,y,losses = learn_c.get_preds(with_loss=True)
predictions = np.argmax(preds, axis = 1)
interp = ClassificationInterpretation(learn_c, preds, y, losses)
interp.plot_confusion_matrix()

predictions[:15], y[:15], predictions.shape, y.shape

print(classification_report(y, predictions, target_names=learn_c.data.classes, digits=4))
print(accuracy_score(y, predictions))

data_clas = load_data(path/"data", "data_clas_bwd.pkl", bs=bs, backwards=True)

config = awd_lstm_clas_config.copy()
config['qrnn'] = True
config['n_hid'] = 1550 #default 1152
config['n_layers'] = 4 #default 3

learn_c = text_classifier_learner(data_clas, AWD_LSTM, config=config, pretrained=False, drop_mult=0.5, 
                                  metrics=[accuracy,f1])
learn_c.path = path
learn_c.load_encoder("fine_tune_enc_bwd");

learn_c.unfreeze()
learn_c.lr_find()

learn_c.recorder.plot()

lr=1e-2

learn_c.fit_one_cycle(40, slice(lr/(2.6**4),lr), wd=0.6, moms=(0.8,0.7),
                      callbacks=[SaveModelCallback(learn_c, name="clf_bwd_no_gradual_unfreeze")])

data_clas = load_data(path/"data", "data_clas_bwd.pkl", bs=bs, backwards=True)

data_test = load_data(path/"data", "test_data_bwd.pkl", bs=bs, backwards=True)

len(data_test.vocab.itos), len(data_clas.vocab.itos)

learn_c.load(path/"clf_bwd_no_gradual_unfreeze");

learn_c.data.valid_dl = data_test.fix_dl

preds,y,losses = learn_c.get_preds(with_loss=True)
predictions = np.argmax(preds, axis = 1)
interp = ClassificationInterpretation(learn_c, preds, y, losses)
interp.plot_confusion_matrix()

predictions[:15], y[:15], predictions.shape, y.shape

print(classification_report(y, predictions, target_names=learn_c.data.classes, digits=4))
print(accuracy_score(y, predictions))

data_clas = load_data(path/"data", "data_clas_export.pkl", bs=bs, num_workers=1)
learn_c = text_classifier_learner(data_clas, AWD_LSTM, config=config, drop_mult=0.5, metrics=[accuracy,f1])
learn_c.path = path
learn_c.load("clf_fwd_no_gradual_unfreeze");

data_test = load_data(path/"data", "test_data.pkl", bs=bs)

learn_c.data.valid_dl = data_test.fix_dl

preds,targs = learn_c.get_preds(ordered=True)
accuracy(preds,targs)

data_clas_bwd = load_data(path/"data", "data_clas_bwd.pkl", bs=bs, num_workers=1, backwards=True)
learn_c_bwd = text_classifier_learner(data_clas_bwd, AWD_LSTM, config=config, drop_mult=0.5, metrics=[accuracy,f1]).to_fp16()
learn_c_bwd.path = path
learn_c_bwd.load("clf_bwd_no_gradual_unfreeze");

data_test_bwd = load_data(path/"data", "test_data_bwd.pkl", bs=bs, backwards=True)

learn_c_bwd.data.valid_dl = data_test_bwd.fix_dl

preds_bwd,targs_bwd = learn_c_bwd.get_preds(ordered=True)
accuracy(preds_bwd,targs_bwd)

preds_avg = (preds+preds_bwd)/2
accuracy(preds_avg, targs)

predictions = np.argmax(preds_avg, axis = 1)

print(classification_report(targs, predictions, target_names=learn_c.data.classes, digits=4))
print(accuracy_score(targs, predictions))

# ### Pre-Trained LM + Fine-tune LM  + Top Only

bs = 8

data_clas = load_data(path/"data", "data_clas_export.pkl", bs=bs)

len(data_clas.vocab.itos)

config = awd_lstm_clas_config.copy()
config['qrnn'] = True
config['n_hid'] = 1550 #default 1152
config['n_layers'] = 4 #default 3

learn_c = text_classifier_learner(data_clas, AWD_LSTM, config=config, pretrained=False, drop_mult=0.5, 
                                  metrics=[accuracy,f1])
learn_c.path = path
learn_c.load_encoder("fine_tune_enc");

learn_c.freeze()
learn_c.lr_find()

learn_c.recorder.plot()

lr=2e-2

learn_c.fit_one_cycle(40, slice(lr/(2.6**4),lr), wd=0.6, moms=(0.8,0.7),
                      callbacks=[SaveModelCallback(learn_c, name="clf_fwd_top_only")])

data_clas = load_data(path/"data", "data_clas_export.pkl")

data_lm = load_data(path/"data", "data_lm_export.pkl")

data_test = load_data(path/"data", "test_data.pkl")

len(data_test.vocab.itos), len(data_clas.vocab.itos)

learn_c.load(path/"clf_fwd_top_only");

learn_c.data.valid_dl = data_test.fix_dl

preds,y,losses = learn_c.get_preds(with_loss=True)
predictions = np.argmax(preds, axis = 1)
interp = ClassificationInterpretation(learn_c, preds, y, losses)
interp.plot_confusion_matrix()

predictions[:15], y[:15], predictions.shape, y.shape

print(classification_report(y, predictions, target_names=learn_c.data.classes, digits=4))
print(accuracy_score(y, predictions))

data_clas = load_data(path/"data", "data_clas_bwd.pkl", bs=bs, backwards=True)

config = awd_lstm_clas_config.copy()
config['qrnn'] = True
config['n_hid'] = 1550 #default 1152
config['n_layers'] = 4 #default 3

learn_c = text_classifier_learner(data_clas, AWD_LSTM, config=config, pretrained=False, drop_mult=0.5, 
                                  metrics=[accuracy,f1])
learn_c.path = path
learn_c.load_encoder("fine_tune_enc_bwd");

learn_c.freeze()
learn_c.lr_find()

learn_c.recorder.plot()

lr=2e-2

learn_c.fit_one_cycle(40, slice(lr/(2.6**4),lr), wd=0.6, moms=(0.8,0.7),
                      callbacks=[SaveModelCallback(learn_c, name="clf_bwd_top_only")])

data_clas = load_data(path/"data", "data_clas_bwd.pkl", bs=bs, backwards=True)

data_test = load_data(path/"data", "test_data_bwd.pkl", bs=bs, backwards=True)

len(data_test.vocab.itos), len(data_clas.vocab.itos)

learn_c.load(path/"clf_bwd_top_only");

learn_c.data.valid_dl = data_test.fix_dl

preds,y,losses = learn_c.get_preds(with_loss=True)
predictions = np.argmax(preds, axis = 1)
interp = ClassificationInterpretation(learn_c, preds, y, losses)
interp.plot_confusion_matrix()

predictions[:15], y[:15], predictions.shape, y.shape

print(classification_report(y, predictions, target_names=learn_c.data.classes, digits=4))
print(accuracy_score(y, predictions))

data_clas = load_data(path/"data", "data_clas_export.pkl", bs=bs, num_workers=1)
learn_c = text_classifier_learner(data_clas, AWD_LSTM, config=config, drop_mult=0.5, metrics=[accuracy,f1])
learn_c.path = path
learn_c.load("clf_fwd_top_only");

data_test = load_data(path/"data", "test_data.pkl", bs=bs)

learn_c.data.valid_dl = data_test.fix_dl

preds,targs = learn_c.get_preds(ordered=True)
accuracy(preds,targs)

data_clas_bwd = load_data(path/"data", "data_clas_bwd.pkl", bs=bs, num_workers=1, backwards=True)
learn_c_bwd = text_classifier_learner(data_clas_bwd, AWD_LSTM, config=config, drop_mult=0.5, metrics=[accuracy,f1]).to_fp16()
learn_c_bwd.path = path
learn_c_bwd.load("clf_bwd_top_only");

data_test_bwd = load_data(path/"data", "test_data_bwd.pkl", bs=bs, backwards=True)

learn_c_bwd.data.valid_dl = data_test_bwd.fix_dl

preds_bwd,targs_bwd = learn_c_bwd.get_preds(ordered=True)
accuracy(preds_bwd,targs_bwd)

preds_avg = (preds+preds_bwd)/2
accuracy(preds_avg, targs)

predictions = np.argmax(preds_avg, axis = 1)

print(classification_report(targs, predictions, target_names=learn_c.data.classes, digits=4))
print(accuracy_score(targs, predictions))

# ### Pre-Trained LM + No Fine-Tune LM + Gradual Unfreezing

bs = 8

data_clas = load_data(path/"data", "data_clas_export.pkl", bs=bs)

len(data_clas.vocab.itos)

config = awd_lstm_clas_config.copy()
config['qrnn'] = True
config['n_hid'] = 1550 #default 1152
config['n_layers'] = 4 #default 3

learn_c = text_classifier_learner(data_clas, AWD_LSTM, config=config, pretrained=False, drop_mult=0.5, 
                                  metrics=[accuracy,f1])
learn_c.path = path
learn_c.load_encoder("no_fine_tune_enc");

learn_c.freeze()
learn_c.lr_find()

learn_c.recorder.plot()

lr=1e-2

learn_c.fit_one_cycle(10, lr, wd=0.6, moms=(0.8,0.7))

learn_c.save("clf_no_lm_tune")

learn_c.freeze_to(-2)
learn_c.fit_one_cycle(10, slice(lr/(2.6**4),lr), wd=0.6, moms=(0.8,0.7))

learn_c.save("clf_no_lm_tune")

learn_c.freeze_to(-3)
learn_c.fit_one_cycle(10, slice(lr/2/(2.6**4),lr/2), wd=0.6, moms=(0.8,0.7))

learn_c.save("clf_no_lm_tune")

learn_c.unfreeze()
learn_c.fit_one_cycle(10, slice(lr/10/(2.6**4),lr/10), wd=0.6, moms=(0.8,0.7))

learn_c.save("clf_no_lm_tune")

data_clas = load_data(path/"data", "data_clas_export.pkl")

data_lm = load_data(path/"data", "data_lm_export.pkl")

data_test = load_data(path/"data", "test_data.pkl")

len(data_test.vocab.itos), len(data_clas.vocab.itos)

learn_c.load(path/"clf_no_lm_tune");

learn_c.data.valid_dl = data_test.fix_dl

preds,y,losses = learn_c.get_preds(with_loss=True)
predictions = np.argmax(preds, axis = 1)
interp = ClassificationInterpretation(learn_c, preds, y, losses)
interp.plot_confusion_matrix()

predictions[:15], y[:15], predictions.shape, y.shape

print(classification_report(y, predictions, target_names=learn_c.data.classes, digits=4))
print(accuracy_score(y, predictions))

data_clas = load_data(path/"data", "data_clas_bwd.pkl", bs=bs, backwards=True)

config = awd_lstm_clas_config.copy()
config['qrnn'] = True
config['n_hid'] = 1550 #default 1152
config['n_layers'] = 4 #default 3

learn_c = text_classifier_learner(data_clas, AWD_LSTM, config=config, pretrained=False, drop_mult=0.5, 
                                  metrics=[accuracy,f1])
learn_c.path = path
learn_c.load_encoder("no_fine_tune_enc_bwd");

learn_c.freeze()
learn_c.lr_find()

learn_c.recorder.plot()

lr=2e-2

learn_c.fit_one_cycle(10, lr, wd=0.6, moms=(0.8,0.7))

learn_c.save("clf_no_lm_tune_bwd")

learn_c.freeze_to(-2)
learn_c.fit_one_cycle(10, slice(lr/(2.6**4),lr), wd=0.6, moms=(0.8,0.7))

learn_c.save("clf_no_lm_tune_bwd")

learn_c.freeze_to(-3)
learn_c.fit_one_cycle(10, slice(lr/2/(2.6**4),lr/2), wd=0.6, moms=(0.8,0.7))

learn_c.save("clf_no_lm_tune_bwd")

learn_c.unfreeze()
learn_c.fit_one_cycle(10, slice(lr/10/(2.6**4),lr/10), wd=0.6, moms=(0.8,0.7))

learn_c.save("clf_no_lm_tune_bwd")

data_clas = load_data(path/"data", "data_clas_bwd.pkl", bs=bs, backwards=True)

data_test = load_data(path/"data", "test_data_bwd.pkl", bs=bs, backwards=True)

len(data_test.vocab.itos), len(data_clas.vocab.itos)

learn_c.load(path/"clf_no_lm_tune_bwd");

learn_c.data.valid_dl = data_test.fix_dl

preds,y,losses = learn_c.get_preds(with_loss=True)
predictions = np.argmax(preds, axis = 1)
interp = ClassificationInterpretation(learn_c, preds, y, losses)
interp.plot_confusion_matrix()

predictions[:15], y[:15], predictions.shape, y.shape

print(classification_report(y, predictions, target_names=learn_c.data.classes, digits=4))
print(accuracy_score(y, predictions))

data_clas = load_data(path/"data", "data_clas_export.pkl", bs=bs, num_workers=1)
learn_c = text_classifier_learner(data_clas, AWD_LSTM, config=config, drop_mult=0.5, metrics=[accuracy,f1])
learn_c.path = path
learn_c.load("clf_no_lm_tune");

data_test = load_data(path/"data", "test_data.pkl", bs=bs)

learn_c.data.valid_dl = data_test.fix_dl

preds,targs = learn_c.get_preds(ordered=True)
accuracy(preds,targs)

data_clas_bwd = load_data(path/"data", "data_clas_bwd.pkl", bs=bs, num_workers=1, backwards=True)
learn_c_bwd = text_classifier_learner(data_clas_bwd, AWD_LSTM, config=config, drop_mult=0.5, metrics=[accuracy,f1]).to_fp16()
learn_c_bwd.path = path
learn_c_bwd.load("clf_no_lm_tune_bwd");

data_test_bwd = load_data(path/"data", "test_data_bwd.pkl", bs=bs, backwards=True)

learn_c_bwd.data.valid_dl = data_test_bwd.fix_dl

preds_bwd,targs_bwd = learn_c_bwd.get_preds(ordered=True)
accuracy(preds_bwd,targs_bwd)

preds_avg = (preds+preds_bwd)/2
accuracy(preds_avg, targs)

predictions = np.argmax(preds_avg, axis = 1)

print(classification_report(targs, predictions, target_names=learn_c.data.classes, digits=4))
print(accuracy_score(targs, predictions))

# ### Pre-Trained LM + No Fine-Tune LM + No Gradual Unfreezing

bs = 8

data_clas = load_data(path/"data", "data_clas_export.pkl", bs=bs)

len(data_clas.vocab.itos)

config = awd_lstm_clas_config.copy()
config['qrnn'] = True
config['n_hid'] = 1550 #default 1152
config['n_layers'] = 4 #default 3

learn_c = text_classifier_learner(data_clas, AWD_LSTM, config=config, pretrained=False, drop_mult=0.5, 
                                  metrics=[accuracy,f1])
learn_c.path = path
learn_c.load_encoder("no_fine_tune_enc");

learn_c.freeze()
learn_c.lr_find()

learn_c.recorder.plot()

lr=2e-2

learn_c.fit_one_cycle(40, slice(lr/(2.6**4),lr), wd=0.6, moms=(0.8,0.7),
                      callbacks=[SaveModelCallback(learn_c, name="clf_no_lm_tune_no_gradual_unfreeze")])

data_clas = load_data(path/"data", "data_clas_export.pkl")

data_lm = load_data(path/"data", "data_lm_export.pkl")

data_test = load_data(path/"data", "test_data.pkl")

len(data_test.vocab.itos), len(data_clas.vocab.itos)

learn_c.load(path/"clf_no_lm_tune_no_gradual_unfreeze");

learn_c.data.valid_dl = data_test.fix_dl

preds,y,losses = learn_c.get_preds(with_loss=True)
predictions = np.argmax(preds, axis = 1)
interp = ClassificationInterpretation(learn_c, preds, y, losses)
interp.plot_confusion_matrix()

predictions[:15], y[:15], predictions.shape, y.shape

print(classification_report(y, predictions, target_names=learn_c.data.classes, digits=4))
print(accuracy_score(y, predictions))

data_clas = load_data(path/"data", "data_clas_bwd.pkl", bs=bs, backwards=True)

config = awd_lstm_clas_config.copy()
config['qrnn'] = True
config['n_hid'] = 1550 #default 1152
config['n_layers'] = 4 #default 3

learn_c = text_classifier_learner(data_clas, AWD_LSTM, config=config, pretrained=False, drop_mult=0.5, 
                                  metrics=[accuracy,f1])
learn_c.path = path
learn_c.load_encoder("no_fine_tune_enc_bwd");

learn_c.freeze()
learn_c.lr_find()

learn_c.recorder.plot()

lr=2e-2

learn_c.fit_one_cycle(40, slice(lr/(2.6**4),lr), wd=0.6, moms=(0.8,0.7),
                      callbacks=[SaveModelCallback(learn_c, name="clf_no_lm_tune_no_gradual_unfreeze_bwd")])

data_clas = load_data(path/"data", "data_clas_bwd.pkl", bs=bs, backwards=True)

data_test = load_data(path/"data", "test_data_bwd.pkl", bs=bs, backwards=True)

len(data_test.vocab.itos), len(data_clas.vocab.itos)

learn_c.load(path/"clf_no_lm_tune_no_gradual_unfreeze_bwd");

learn_c.data.valid_dl = data_test.fix_dl

preds,y,losses = learn_c.get_preds(with_loss=True)
predictions = np.argmax(preds, axis = 1)
interp = ClassificationInterpretation(learn_c, preds, y, losses)
interp.plot_confusion_matrix()

predictions[:15], y[:15], predictions.shape, y.shape

print(classification_report(y, predictions, target_names=learn_c.data.classes, digits=4))
print(accuracy_score(y, predictions))

data_clas = load_data(path/"data", "data_clas_export.pkl", bs=bs, num_workers=1)
learn_c = text_classifier_learner(data_clas, AWD_LSTM, config=config, drop_mult=0.5, metrics=[accuracy,f1])
learn_c.path = path
learn_c.load("clf_no_lm_tune_no_gradual_unfreeze");

data_test = load_data(path/"data", "test_data.pkl", bs=bs)

learn_c.data.valid_dl = data_test.fix_dl

preds,targs = learn_c.get_preds(ordered=True)
accuracy(preds,targs)

data_clas_bwd = load_data(path/"data", "data_clas_bwd.pkl", bs=bs, num_workers=1, backwards=True)
learn_c_bwd = text_classifier_learner(data_clas_bwd, AWD_LSTM, config=config, drop_mult=0.5, metrics=[accuracy,f1]).to_fp16()
learn_c_bwd.path = path
learn_c_bwd.load("clf_no_lm_tune_no_gradual_unfreeze_bwd");

data_test_bwd = load_data(path/"data", "test_data_bwd.pkl", bs=bs, backwards=True)

learn_c_bwd.data.valid_dl = data_test_bwd.fix_dl

preds_bwd,targs_bwd = learn_c_bwd.get_preds(ordered=True)
accuracy(preds_bwd,targs_bwd)

preds_avg = (preds+preds_bwd)/2
accuracy(preds_avg, targs)

predictions = np.argmax(preds_avg, axis = 1)

print(classification_report(targs, predictions, target_names=learn_c.data.classes, digits=4))
print(accuracy_score(targs, predictions))
