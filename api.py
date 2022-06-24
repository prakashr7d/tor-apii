# Imports used through the rest of the notebook.
import torch
import torchaudio
import torch.nn as nn
import torch.nn.functional as F

import IPython

from tortoise.api import TextToSpeech
from tortoise.utils.audio import load_audio, load_voice, load_voices

# This will download all the models used by Tortoise from the HuggingFace hub.
tts = TextToSpeech()

# This is the text that will be spoken.
text = "Joining two modalities results in a surprising increase in generalization! What would happen if we combined them all?"

# Here's something for the poetically inclined.. (set text=)
"""
Then took the other, as just as fair,
And having perhaps the better claim,
Because it was grassy and wanted wear;
Though as for that the passing there
Had worn them really about the same,"""

# Pick a "preset mode" to determine quality. Options: {"ultra_fast", "fast" (default), "standard", "high_quality"}. See docs in api.py
preset = "fast"

# Pick one of the voices from the output above
voice = 'train_dotrice'

# Load it and send it through Tortoise.
voice_samples, conditioning_latents = load_voice(voice)
gen = tts.tts_with_preset(text, voice_samples=voice_samples, conditioning_latents=conditioning_latents,
                          preset=preset)
torchaudio.save('generated.wav', gen.squeeze(0).cpu(), 24000)
IPython.display.Audio('generated.wav')