"""Conservative optional waveform cleaning."""


def clean_audio(waveform, *, remove_dc=False):
    if remove_dc:
        return waveform - waveform.mean()
    return waveform
