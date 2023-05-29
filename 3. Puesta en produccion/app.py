import wave
from flask import Flask, request, render_template, flash
from matplotlib import pyplot as plt
import tensorflow as tf
import joblib
import librosa
import numpy as np
import pylab
import cv2
import os

app = Flask(__name__)
app.config['SECRET_KEY'] = 'mi_clave_secreta'
rfc = joblib.load('../Models/modelo_rfc.pkl')
svc = joblib.load('../Models/modelo_svc.pkl')
rn_mel = tf.keras.models.load_model('../Models/modeloRN_mel.h5')
le_mel = joblib.load('../Models/label_encoder_mel.pkl')
rn_img = tf.keras.models.load_model('../Models/modeloRN_img.h5')
le_img = joblib.load('../Models/label_encoder_img.pkl')


def descargar(file ,ruta_guardado):

    nchannels = 1  # Número de canales (mono)
    sampwidth = 2  # Ancho de muestra en bytes
    framerate = 44100  # Tasa de muestreo en Hz
    nframes = len(file) // (nchannels * sampwidth)  # Número de frames

    # Abrir el archivo WAV en modo de escritura
    with wave.open(ruta_guardado, 'wb') as wav_file:
        wav_file.setnchannels(nchannels)
        wav_file.setsampwidth(sampwidth)
        wav_file.setframerate(framerate)
        wav_file.setnframes(nframes)
        wav_file.writeframes(file)

def audio_img(ruta_guardado, ruta_guardado_img):
    wav = wave.open(ruta_guardado, 'r')
    frames = wav.readframes(-1)
    sound_info = pylab.frombuffer(frames,dtype=np.int16)
    frame_rate = wav.getframerate()
    wav.close()

    pylab.specgram(sound_info, Fs=frame_rate)
    # Eliminar los ejes x e y
    pylab.xticks([])
    pylab.yticks([])
    # Configurar el tamaño de la figura
    fig = plt.gcf()
    fig.set_size_inches(1.28, 1.28)  # Tamaño de 128x128 píxeles
    # Configurar los bordes
    plt.subplots_adjust(left=0, right=1, bottom=0, top=1)  # Eliminar los bordes
    # Guardar el espectrograma con el tamaño y parámetros deseados
    fig.savefig(ruta_guardado_img, transparent=True)
    plt.close()

@app.route('/', methods=['GET', 'POST'])
def index():
    if request.method == 'POST':
        file = request.files['audio']
        if len(file.filename) > 0:
            ruta_guardado = 'archivo.wav'
            descargar(file.read(), ruta_guardado)
            audio, sample_rate = librosa.load(ruta_guardado, res_type='kaiser_fast')
            feature = librosa.feature.mfcc(y=audio, sr=sample_rate)
            scaled_feature = np.mean(feature.T, axis=0)
            array = scaled_feature.reshape(1, -1)
            pred_rfc = rfc.predict(array)
            pred_svc = svc.predict(array)
            pred_rn_mel = rn_mel.predict(array)
            pred_rn_mel = le_mel.inverse_transform(np.argmax(pred_rn_mel, axis=1))

            ruta_guardado_img = 'audio.png'
            audio_img(ruta_guardado, ruta_guardado_img)
            image = cv2.imread(ruta_guardado_img)
            image = np.expand_dims(image, axis=0)
            pred_rn_img = rn_img.predict(image)
            pred_rn_img = le_img.inverse_transform(np.argmax(pred_rn_img, axis=1))

             
            if os.path.exists(ruta_guardado):
                os.remove(ruta_guardado)

            if os.path.exists(ruta_guardado_img):
                os.remove(ruta_guardado_img)

            return render_template('index.html', resultados=True, pred_rfc=pred_rfc[0], pred_svc=pred_svc[0], pred_rn_mel=pred_rn_mel[0], pred_rn_img=pred_rn_img[0])
        else:
            # si no se sube corectamenta
            flash('Error al suber el audio intentalo de nuevo')
            return render_template('index.html')

    return render_template('index.html')


if __name__ == '__main__':
    app.run(debug=True)
