# O Julgamento — vídeo animado 9:16

Gerador procedural do vídeo (1080x1920, 30 fps, ~13m52s).

- `script.txt` — roteiro decupado em planos (`@plano`) e falas (`N` narradora, `J` Jesus, `P` ex-presidente, `Q` pergunta das famílias, `S` silêncio).
- `build_audio.py` — narração neural pt-BR (edge-tts), reverb de salão, trilha ambiente sintetizada, efeitos e `timeline.json`.
- `render.py` — arte e animação quadro a quadro (Pillow/numpy), legendas sincronizadas palavra a palavra.

```
pip install pillow numpy imageio-ffmpeg edge-tts
python3 build_audio.py
python3 render.py chunk 0 24952 video.mp4   # depois muxar com audio.wav
```

As fontes (Cormorant Garamond, Cinzel — Google Fonts) são esperadas em `../fonts/`.
