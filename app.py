from kokoro import KModel, KPipeline
from flask import Flask, request, send_file, jsonify
import io
import os
import numpy as np
import soundfile as sf
import torch

app = Flask(__name__)

API_KEY = os.environ.get('API_KEY')


def require_api_key(f):
    from functools import wraps
    @wraps(f)
    def decorated(*args, **kwargs):
        if not API_KEY:
            return jsonify({'error': 'API key not configured on server'}), 500
        if request.headers.get('X-API-Key') != API_KEY:
            return jsonify({'error': 'Unauthorized'}), 401
        return f(*args, **kwargs)
    return decorated

CUDA_AVAILABLE = torch.cuda.is_available()

_models = None
_pipelines = None


def get_models():
    global _models
    if _models is None:
        _models = {
            gpu: KModel().to('cuda' if gpu else 'cpu').eval()
            for gpu in [False] + ([True] if CUDA_AVAILABLE else [])
        }
    return _models


def get_pipelines():
    global _pipelines
    if _pipelines is None:
        _pipelines = {lang_code: KPipeline(lang_code=lang_code, model=False) for lang_code in 'ab'}
        _pipelines['a'].g2p.lexicon.golds['kokoro'] = 'kˈOkəɹO'
        _pipelines['b'].g2p.lexicon.golds['kokoro'] = 'kˈQkəɹQ'
    return _pipelines

VOICES = {
    'af_heart', 'af_bella', 'af_nicole', 'af_aoede', 'af_kore',
    'af_sarah', 'af_nova', 'af_sky', 'af_alloy', 'af_jessica', 'af_river',
    'am_michael', 'am_fenrir', 'am_puck', 'am_echo', 'am_eric',
    'am_liam', 'am_onyx', 'am_santa', 'am_adam',
    'bf_emma', 'bf_isabella', 'bf_alice', 'bf_lily',
    'bm_george', 'bm_fable', 'bm_lewis', 'bm_daniel',
}


def generate_audio(text: str, voice: str = 'af_heart', speed: float = 1.0) -> bytes:
    pipelines = get_pipelines()
    models = get_models()
    pipeline = pipelines[voice[0]]
    pack = pipeline.load_voice(voice)
    audio_segments = []

    for _, ps, _ in pipeline(text, voice, speed):
        ref_s = pack[len(ps) - 1]
        if CUDA_AVAILABLE:
            audio = models[True](ps, ref_s, speed)
        else:
            audio = models[False](ps, ref_s, speed)
        audio_segments.append(audio.numpy())

    if not audio_segments:
        raise ValueError('No audio generated for the provided text.')

    combined = np.concatenate(audio_segments)
    buf = io.BytesIO()
    sf.write(buf, combined, 24000, format='MP3')
    buf.seek(0)
    return buf

@app.route('/', methods=['GET'])
def index():
    return jsonify({'message': 'Kokoro TTS API'}), 200

@app.route('/tts', methods=['POST'])
@require_api_key
def tts():
    data = request.get_json(silent=True) or {}
    text = data.get('text', '').strip()
    voice = data.get('voice', 'af_heart')
    speed = float(data.get('speed', 1.0))

    if not text:
        return jsonify({'error': 'text is required'}), 400
    if voice not in VOICES:
        return jsonify({'error': f'unknown voice "{voice}"'}), 400
    if not (0.5 <= speed <= 2.0):
        return jsonify({'error': 'speed must be between 0.5 and 2.0'}), 400

    buf = generate_audio(text, voice, speed)
    return send_file(buf, mimetype='audio/mpeg', as_attachment=True, download_name='output.mp3')


if __name__ == '__main__':
    import os
    port = int(os.environ.get('PORT', 5000))
    app.run(host='0.0.0.0', port=port, debug=False)
