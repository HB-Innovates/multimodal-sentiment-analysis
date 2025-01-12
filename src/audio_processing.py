def load_audio(file_path):
    import librosa
    audio, sr = librosa.load(file_path, sr=None)
    return audio, sr

def extract_features(audio, sr):
    import numpy as np
    features = {
        'mfcc': librosa.feature.mfcc(y=audio, sr=sr, n_mfcc=13),
        'chroma': librosa.feature.chroma_stft(y=audio, sr=sr),
        'mel': librosa.feature.melspectrogram(y=audio, sr=sr),
        'contrast': librosa.feature.spectral_contrast(y=audio, sr=sr),
        'tonnetz': librosa.feature.tonnetz(y=audio, sr=sr)
    }
    return {key: np.mean(value, axis=1) for key, value in features.items()}

def process_audio(file_path):
    audio, sr = load_audio(file_path)
    features = extract_features(audio, sr)
    return features