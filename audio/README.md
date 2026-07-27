# audio/

Recordings live here. **Audio files are gitignored** — only `.md`, `.json` and
`.csv` under this directory are tracked, so manifests, metadata and annotations
are versioned while the media is not.

Nothing has been recorded yet. This file fixes the conventions in advance so the
first session does not have to invent them.

## Layout

```
audio/
  session-NN/
    manifest.csv      prompt id, text, harmony class, entity labels
    clips/            NN-###.wav        (gitignored)
    meta.json         speaker + environment metadata, see below
    qc.csv            kgvoice bench qc output
    annotations/      prosody / disfluency markup
```

## Capture spec

| Field | Value |
|---|---|
| Format | WAV, 48 kHz, 24-bit PCM |
| Channels | Mono |
| Peak | ≈ −3 dBFS, no clipping |
| Integrated loudness | ≈ −16 LUFS |
| Noise floor | ≤ −60 dBFS, measured and recorded per session |
| Processing | None — no EQ, compression, or noise reduction. Trim only. |

Downsampling to 22.05 kHz for TTS training is a downstream step; capture stays at
48 kHz so the same material can serve ASR work without a second session.

## Session metadata (`meta.json`)

Recorded per speaker, because these are the variables that make the data useful
for anything beyond a single voice:

- region / dialect area
- age band
- urban or rural
- L1, and degree of Russian exposure
- microphone and signal chain
- room, and any treatment
- measured noise floor
- consent status and scope

## Consent

No recording enters this repository without recorded consent covering the
intended use, including redistribution terms. A speaker agreeing to "a research
project" has not agreed to a public dataset release.
