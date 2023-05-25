from flask import Flask, request, render_template, flash
import joblib
import librosa
import numpy as np

app = Flask(__name__)
app.config['SECRET_KEY'] = 'mi_clave_secreta'
rfc = joblib.load('../Models/modelo_rfc.pkl')
svc = joblib.load('../Models/modelo_svc.pkl')


@app.route('/', methods=['GET', 'POST'])
def index():
    if request.method == 'POST':
        archivo = request.files['audio']

        if len(archivo.filename) > 0:
            # si se sube corectamenta
            audio, sample_rate = librosa.load(archivo, res_type='kaiser_fast')  # Cargamos el archivo de audio
            # Calculamos características de MFCC
            feature = librosa.feature.mfcc(y=audio, sr=sample_rate)
            scaled_feature = np.mean(feature.T, axis=0)
            array = scaled_feature.reshape(1, -1)
            pred_rfc = rfc.predict(array)
            pred_svc = svc.predict(array)
            return render_template('index.html', resultados=True, pred_rfc=pred_rfc[0], pred_svc=pred_svc[0])
        else:
            # si no se sube corectamenta
            flash('Error al suber el audio intentalo de nuevo')
            return render_template('index.html')

    return render_template('index.html')


if __name__ == '__main__':
    app.run(debug=True)
