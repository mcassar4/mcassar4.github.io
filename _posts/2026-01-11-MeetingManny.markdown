---
layout: post
title: "MeetingManny - A macOS Menu Bar Copilot for Meetings"
img: aiNotes/image.png
date: 2026-01-11
tags: [python, macos, ai, productivity]
---

# MeetingManny: A macOS Menu Bar Copilot for Meetings

If you run a lot of calls, the busiest part of a meeting often happens *after* it ends: compiling notes, action items, and next steps. MeetingManny is a small, practical macOS menu bar app that handles the grunt work. It records your screen and mic, transcribes with Whisper, summarizes with Ollama, and can optionally ship structured notes to Notion.

## Why I built it

I wanted a lightweight, local-first workflow that felt like a menu bar utility, not a heavyweight meeting suite. I also wanted something I could *hack on*—a simple architecture where each step is replaceable (different capture method, different LLM, different sink).

MeetingManny optimizes for:

- frictionless capture (one click in the status bar)
- high-quality transcripts (Whisper, with optional diarization)
- structured summaries (title, key points, action items, decisions, next steps)
- optional Notion sync (without paying for Notion AI)

## Architecture at a glance

At a high level, MeetingManny is a small orchestrator that glues together a few well-defined components:

- **Capture**: screen recording + audio recording (optionally merged via ffmpeg)
- **Storage**: a per-session folder with stable filenames (audio/video/transcript/logs)
- **Processing**: transcription (Whisper), optional diarization, then structured extraction (Ollama)
- **Integration**: optional Notion upload

The menu bar UI is just a thin controller that triggers the pipeline and shows status; the “real work” happens in the recording + processing steps.

## Data flow (what happens when you click Stop)

From the menu bar, MeetingManny captures your screen and audio in the background. When you stop recording, it runs a simple pipeline:

1. **Finalize artifacts**: ensure the session folder has the raw audio + screen recording (and a merged recording when ffmpeg is available).
2. **Transcribe**: run Whisper to produce `transcript.txt`.
3. **(Optional) Diarize**: if enabled, run a diarization step to attribute segments to speakers.
4. **Extract structure**: send transcript (and any extra context) to an Ollama model to output a consistent schema (title, bullets, action items, decisions, next steps).
5. **(Optional) Publish**: push a structured note to Notion.

Each session is stored in a timestamped folder with the audio, video, transcript, and logs. It keeps things simple and transparent.

## Key features

- **Menu bar controls:** One-click start/stop from macOS status bar.
- **Screen + audio capture:** Video recording with optional ffmpeg merge.
- **Whisper transcription:** Reliable speech-to-text, locally.
- **Speaker diarization (optional):** Separate speakers if desired.
- **Ollama summarization:** Local or remote LLM for structured meeting notes.
- **Notion upload (optional):** Push clean notes to your workspace.

## Design choices (the “why” behind the architecture)

Some deliberate tradeoffs that make the app easier to trust and extend:

- **Local-first artifacts**: every run produces files you can inspect, edit, and archive.
- **Replaceable steps**: capture, transcription, diarization, summarization, and upload are separable units.
- **Structured output over prose**: the goal is “notes you can act on,” not a wall of text.

If you already live in Notion, the optional upload keeps your meeting notes consistent and searchable.

## Configuration surface (what you can swap)

MeetingManny’s behavior is controlled via a `.env` file, which keeps the architecture flexible without turning the UI into a settings panel. The main categories:

- **Capture**: output directory, FPS, monitor selection, codecs, ffmpeg path
- **Audio**: input device selection, sample rate, channel count
- **Transcription**: Whisper model choice
- **Diarization**: enable/disable + script path/args
- **LLM extraction**: Ollama model + host
- **Publishing**: optional Notion token + database id

That’s intentional: the menu bar stays minimal, and power users can tune the pipeline with environment variables.

## System audio capture (BlackHole as an architecture dependency)

On macOS, “microphone audio” and “system audio” are two different things. If you want your recording to include both sides of a call (or any app audio), you typically need a virtual audio device.

A popular, lightweight option is **BlackHole**, a virtual audio driver that shows up as an input/output device in macOS. In MeetingManny terms, BlackHole isn’t “a feature”—it’s an upstream dependency that makes the **Audio Input** component capable of receiving system audio at all.

### My setup (clean + reliable)

This is the approach that consistently captures **system audio + mic**, while still letting you **listen live**. It creates a “monitoring output” path and a separate “recording input” path.

**1) Multi‑Output Device (for listening)**

In **Audio MIDI Setup**:

- Create a Multi‑Output Device (e.g. `Custom Output`).
- Check `Manny’s AirPods Pro` + `BlackHole 2ch`.
- Set **Primary Device** = `AirPods`.
- Enable **Drift Correction** on `BlackHole 2ch`.
- Set **Sample Rate** = `48.0 kHz`.

Then in **System Settings → Sound → Output**, pick `Custom Output`.

**2) Aggregate Device (for recording input)**

In **Audio MIDI Setup**:

- Create an Aggregate Device (e.g. `AggIn`).
- Check `BlackHole 2ch` + `MacBook Pro Microphone` (or your AirPods mic if you insist).
- Set **Clock Source** = `Microphone`.
- Enable **Drift Correction** on `BlackHole 2ch`.
- Set **Sample Rate** = `48.0 kHz`.

**3) MeetingManny / MeetingFusion `.env`**

Point the app at the aggregate input (name or device index from `--list-audio-devices`):

```env
AUDIO_DEVICE=AggIn
AUDIO_CHANNELS=3
AUDIO_SAMPLE_RATE=48000
```

Why `3` channels? In this configuration, `AggIn` exposes **3 inputs** (2 from BlackHole + 1 from the mic). Capturing all 3 ensures you don’t lose either source; Whisper will down‑mix as needed for transcription.

### Notes and gotchas

- If you only select BlackHole as input, you’ll capture system audio but not your microphone. Use an Aggregate Device if you want both.
- Echo/feedback can happen if you accidentally route audio back into itself; keep routing minimal and verify in Audio MIDI Setup.
- If audio “disappears” after changing outputs, double-check you’re using a Multi-Output Device (or switch output back to your headphones/speakers).
- **AirPods too loud / no volume slider:** Multi‑Output Devices disable macOS’s system volume control. Workarounds: use AirPods hardware controls, or install a mixer like eqMac which is what I did.

## Output structure

Each meeting produces a folder that looks like this:

```
YYYY-MM-DD_HH-MM-SS/
  audio.wav
  screen.mp4
  recording.mp4
  transcript.txt
  meeting.log
```

This keeps a neat archive you can revisit later or move into your long-term notes system.

## Extending the pipeline

Because the system is built as a sequence of steps with clear inputs/outputs, it’s easy to evolve:

- Swap the **LLM extraction** prompt/model to match your note format.
- Add a post-step that turns action items into tasks (Reminders, Todoist, Linear, etc.).
- Add a “mixdown” step so multi-channel aggregate audio always becomes stereo/mono before archiving.

## What makes it different

MeetingManny is intentionally simple. It does not try to be a full meeting platform. Instead, it is a focused pipeline with local-first artifacts and optional integrations.

Most importantly: your recordings, transcripts, and notes live as files you can own and move.

If you want a quiet, reliable meeting recorder that turns raw audio into clear takeaways, MeetingManny is built for that.

## Try it

If you are curious, set it up locally and record a short test call. The fastest way to see the value is to experience the post-meeting flow: transcript, summary, and action items waiting for you.

---

If you want to customize the summary format, add custom tags, or integrate another system, the pipeline is straightforward to extend.
