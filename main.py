from NanoGPTClassifier import NanoGPTClassifier
from tensorflow.keras.preprocessing.text import Tokenizer
from sklearn.model_selection import train_test_split
from tensorflow.keras.preprocessing import sequence
import pandas as pd
import numpy as np
import torch
from torch import nn

dets = [ 'the', 'a', 'an', 'this', 'that', 'these', 'those', 'a few', 'a little', 'much', 'many', 'a lot of', 'most', 'some', 'any,' 'enough', 
       'all', 'both', 'half', 'either', 'neither', 'each', 'every', 'other', 'another', 'such', 'what', 'rather', 'quite']
pronouns = [ 'my', 'your', 'his', 'her', 'its', 'our', 'their' ]
labels = [ 'negative', 'neutral', 'positive' ]

def parse_captions(txt):
    for det in dets:
        txt = txt.replace(' ' + det + ' ' , ' <DET> ')
    
    for pronoun in pronouns:
        txt = txt.replace(' ' + pronoun + ' ' , ' <PRON> ')
    
    split_txt = txt.split(' ')
    
    for i in range(len(split_txt)):
        token = split_txt[i]
        
        if token.startswith('@'):
            split_txt[i] = '<USER>'
        elif token.startswith('#'):
            split_txt[i] = '<HASHTAG>'
    return ' '.join(split_txt)[:-1]

def apply_preprocessing(X, y):
    X = X.apply(lambda row: parse_captions(row))
    X.head()

    # Tokenize the captions
    myTokenizer = Tokenizer(num_words=512)
    myTokenizer.fit_on_texts(X)
    X_tokens = myTokenizer.texts_to_sequences(X)

    # Pad sequences
    X_padded = sequence.pad_sequences(X_tokens, maxlen=512)

    # Label encode labels
    y_enc = np.zeros((len(y), 3))

    for l in range(len(y)):
        label = y[l]
        y_enc[l][labels.index(label)] = 1

    return X_padded, y_enc.astype(np.int)

def main():
    device = torch.device("cuda:0" if torch.cuda.is_available() else "cpu")

    dataset = pd.read_excel('LabeledText.xlsx')
    dataset = dataset[['Caption', 'LABEL']]
    X = dataset.Caption
    y = dataset.LABEL

    # Parse captions
    X, y = apply_preprocessing(X, y)

    print(X[:5])
    print(X.shape)
    print(y[:5])

    X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2)

    nano_gpt = NanoGPTClassifier(3, 5, 512, 40)
    optimizer = torch.optim.Adam(nano_gpt.parameters(), lr=0.0002) #, betas=(0.5, 0.999)
    loss_criterion = nn.CrossEntropyLoss()
    X_train = torch.from_numpy(X_train).to(device)
    y_train = torch.from_numpy(y_train).to(device)

    nano_gpt.fit(X_train, y_train, optimizer, loss_criterion, epochs=30, batch_size=95)

    test_batch_size = 100
    X_test = X_test[:900]
    y_test = y_test[:900]
    n_batches = np.round(len(X_test) / test_batch_size).astype(np.int)
    X_batched_test = torch.from_numpy(X_test)
    X_batched_test = torch.reshape(X_batched_test, (n_batches, test_batch_size, X_batched_test.shape[1]))
    preds_list = []

    with torch.no_grad():
        for i in range(n_batches):
            preds = nano_gpt(X_batched_test[i])
            enc_preds = torch.argmax(preds, dim=1).to('cpu')
            preds_list += enc_preds.tolist()

        enc_preds = torch.from_numpy(np.asarray(preds_list)).to('cpu')
        enc_y_test = torch.argmax(torch.from_numpy(y_test), dim=1).to('cpu')
        test_acc = torch.sum(enc_preds == enc_y_test) / len(y_test)
        print("Test Accuracy:", test_acc)


if __name__ == '__main__':
    main()