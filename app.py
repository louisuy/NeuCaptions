from flask import Flask, render_template, request
import cv2
from keras.models import load_model
import numpy as np
from keras.applications import ResNet50
from keras.optimizers import Adam
from keras.layers import Dense, Flatten,Input, Convolution2D, Dropout, LSTM, TimeDistributed, Embedding, Bidirectional, Activation, RepeatVector,Concatenate
from keras.models import Sequential, Model
from keras.utils import np_utils
from keras.preprocessing import image, sequence
from keras.utils import pad_sequences
from tqdm import tqdm

from keras.applications import ResNet50

resnet = ResNet50(include_top=False, weights='imagenet', input_shape=(224,224,3), pooling='avg')

print("="*50)
print("ResNet has been loaded.")

vocab = np.load('vocab.npy', allow_pickle=True)
vocab = vocab.item()

embeddingSize = 128
maxLen = 40
vocabSize = len(vocab)
inv_vocab = {v:k for k,v in vocab.items()}

imageModel = Sequential()

imageModel.add(Dense(embeddingSize, input_shape=(2048,), activation='relu'))
imageModel.add(RepeatVector(maxLen))

languageModel = Sequential()

languageModel.add(Embedding(input_dim=vocabSize, output_dim=embeddingSize, input_length=maxLen))
languageModel.add(LSTM(256, return_sequences=True))
languageModel.add(TimeDistributed(Dense(embeddingSize)))


conca = Concatenate()([imageModel.output, languageModel.output])
x = LSTM(128, return_sequences=True)(conca)
x = LSTM(512, return_sequences=False)(x)
x = Dense(vocabSize)(x)
out = Activation('softmax')(x)
model = Model(inputs=[imageModel.input, languageModel.input], outputs = out)

model.compile(loss='categorical_crossentropy', optimizer='RMSprop', metrics=['accuracy'])

model.load_weights('mine_model_weights.h5')

print("="*50)
print("Model has been loaded.")

app = Flask(__name__, static_folder='static')
app.config['SEND_MAX_FILE_AGE__DEFAULT'] = 1

@app.route('/')
def index():
    return render_template('index.html')

@app.route('/after', methods=['GET', 'POST'])
def after():
    global model, vocab, inv_vocab, resnet
    
    file = request.files['file1']
    
    file.save('static/file.jpg')
    
    image = cv2.imread('static/file.jpg')
    image = cv2.cvtColor(image, cv2.COLOR_BGR2RGB)

    image = cv2.resize(image, (224,224))

    image = np.reshape(image, (1,224,224,3))
    
    features = resnet.predict(image).reshape(1,2048)
    print("="*50)
    print("Predicting Features")


    text_in = ['startofseq']
    final = ''

    print("="*50)
    print("Acquiring Captions")

    count = 0
    last_sampled = 'x'
    while tqdm(count < 20):

        count += 1

        encoded = []
        for i in text_in:
            encoded.append(vocab[i])

        padded = pad_sequences([encoded], maxlen=maxLen, padding='post', truncating='post').reshape(1,maxLen)

        sampled_index = np.argmax(model.predict([features, padded]))

        sampled_word = inv_vocab[sampled_index]
        
        if sampled_word != 'endofseq':
            if last_sampled == sampled_word:
                print('Redundant word.')
                pass
            elif sampled_word == '.':
                final = final + sampled_word
                break
            else:
                print('Sampled word ' + sampled_word)
                final = final + ' ' + sampled_word
            last_sampled = sampled_word
            print(final)
        

        text_in.append(sampled_word)
        
    return render_template('predict.html', final=final)

if __name__ == "__main__":
    app.run(debug=True)