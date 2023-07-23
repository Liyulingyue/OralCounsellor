from openvino.runtime import Core, serialize, Tensor
import numpy as np
import librosa
import scipy
import soundfile as sf

def audio_to_mel(audio, sampling_rate):
    assert sampling_rate == 16000, "Only 16 KHz audio supported"
    preemph = 0.97
    preemphased = np.concatenate([audio[:1], audio[1:] - preemph * audio[:-1].astype(np.float32)])

    # Calculate the window length.
    win_length = round(sampling_rate * 0.02)

    # Based on the previously calculated window length, run short-time Fourier transform.
    spec = np.abs(librosa.core.spectrum.stft(preemphased, n_fft=512, hop_length=round(sampling_rate * 0.01),
                  win_length=win_length, center=True, window=scipy.signal.windows.hann(win_length), pad_mode='reflect'))

    # Create mel filter-bank, produce transformation matrix to project current values onto Mel-frequency bins.
    mel_basis = librosa.filters.mel(sr=sampling_rate, n_fft=512, n_mels=64, fmin=0.0, fmax=8000.0, htk=False)
    return mel_basis, spec

def mel_to_input(mel_basis, spec, padding=16):
    # Convert to a logarithmic scale.
    log_melspectrum = np.log(np.dot(mel_basis, np.power(spec, 2)) + 2 ** -24)

    # Normalize the output.
    normalized = (log_melspectrum - log_melspectrum.mean(1)[:, None]) / (log_melspectrum.std(1)[:, None] + 1e-5)

    # Calculate padding.
    remainder = normalized.shape[1] % padding
    if remainder != 0:
        return np.pad(normalized, ((0, 0), (0, padding - remainder)))[None]
    return normalized[None]

def preprocess_of_wav(wav_path):
    audio, sampling_rate = librosa.load(path=wav_path, sr=16000)
    if max(np.abs(audio)) <= 1:
        audio = (audio * (2**15 - 1))
    audio = audio.astype(np.int16)

    mel_basis, spec = audio_to_mel(audio=audio.flatten(), sampling_rate=sampling_rate)
    audio = mel_to_input(mel_basis=mel_basis, spec=spec)
    return audio

def create_asr_model():
    ie = Core()
    model = ie.read_model(
        model=f"speech2text_model/quartznet-15x5-en.xml"
    )
    model_input_layer = model.input(0)
    shape = model_input_layer.partial_shape
    shape[2] = -1
    model.reshape({model_input_layer: shape})
    compiled_model = ie.compile_model(model=model)
    output_layer_ir = compiled_model.output(0)
    return compiled_model, output_layer_ir

def asr(compiled_model, output_layer_ir, audio):
    alphabet = " abcdefghijklmnopqrstuvwxyz'~"
    character_probabilities = compiled_model([Tensor(audio)])[output_layer_ir]
    # Remove unnececery dimension
    character_probabilities = np.squeeze(character_probabilities)
    # Run argmax to pick most possible symbols
    character_probabilities = np.argmax(character_probabilities, axis=1)
    def ctc_greedy_decode(predictions):
        previous_letter_id = blank_id = len(alphabet) - 1
        transcription = list()
        for letter_index in predictions:
            if previous_letter_id != letter_index != blank_id:
                transcription.append(alphabet[letter_index])
            previous_letter_id = letter_index
        return ''.join(transcription)
    transcription = ctc_greedy_decode(character_probabilities)
    return transcription

def preprocess_of_gradio_input(audio):
    sr, data = audio
    # 降采样
    target_sr = 16000
    sf.write('input_audio.wav', data, sr, subtype='PCM_24')
    data, sr = librosa.load(path="input_audio.wav")
    data = librosa.resample(data, sr, target_sr)
    sf.write('input_audio2.wav', data, target_sr, subtype='PCM_24')

# 语音转文字
def speech2text(audio, asr_model, output_ir, file_name='input_audio2.wav'):
    preprocess_of_gradio_input(audio)
    audio = preprocess_of_wav(file_name)
    transcription = asr(asr_model, output_ir, audio)
    return transcription