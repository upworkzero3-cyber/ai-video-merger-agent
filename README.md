# AI Video Merger Agent

A Windows desktop app for automatically joining Gemini-generated clips. Select an input folder, choose **Start Automatic Merging**, and open the result from **History** or **Open Output Folder**. The app also accepts footage from other sources.

## Windows setup

1. Install 64-bit [Python 3.12 for Windows](https://www.python.org/downloads/windows/) and make sure the Python launcher is installed.
2. Use the [FFmpeg download page](https://ffmpeg.org/download.html#build-windows) to get a Windows build of FFmpeg 7 or later with both `ffmpeg.exe` and `ffprobe.exe`, plus the `libx264` encoder. Extract the build. Copy both executables from its `bin` folder to `D:\AI-Video-Merger\bin\`, or add the FFmpeg build's `bin` folder to PATH. If using a shared FFmpeg build, keep its DLLs alongside its executables.
3. Open PowerShell in the project directory and run:

```powershell
Set-Location D:\AI-Video-Merger
py -3.12 -m venv .venv
.\.venv\Scripts\python.exe -m pip install -r requirements.txt
.\.venv\Scripts\python.exe main.py
```

After setup, you can also double-click `Launch AI Video Merger.cmd`.

If using another FFmpeg folder, set `$env:FFMPEG_BIN = 'C:\Tools\ffmpeg\bin'` in the PowerShell window before starting. The environment variable takes precedence over the project `bin` directory and PATH.

## Processing behavior

- Import scans the selected folder for MP4, MOV, MKV, WebM, AVI, M4V, MTS, M2TS, WMV, MPEG and TS videos. Files with numbered names are sorted naturally. If any filename lacks a number, the folder is ordered by modification time from oldest to newest, with natural names as a tie-breaker.
- Duplicate detection compares time-aligned frame sequences within clips and across clips. It checks perceptual hashes, spatial similarity, aligned scene changes and changes between successive frames. Per-channel audio fingerprints and waveform correlation guard against removing changed dialogue, narration, music, or effects. A repeat is removed only if its reference footage will be kept. Static shots, ambiguous matches, and visuals with changed audio are preserved.
- Visual analysis uses eight samples per second. Cut boundaries round inward onto the export frame grid and retain a short guard at the end of a match. A small amount of repeated footage can remain at a join. Time-warped, cropped, or heavily regenerated repeats can also remain, as can repeats with alternate audio.
- Settings expose visual and audio thresholds, minimum sequence length, resolution, frame rate, encoder, and optional crossfades. Crossfades are off by default and apply only at eligible quiet source joins. The default output is 1920×1080, 30 FPS, H.264 video and AAC audio in MP4. Portrait presets are available for vertical footage. Video is fit to the selected canvas with padding as needed. The first playable video track is used. Audio tracks are retained by stream position and normalized to 48 kHz and a common channel layout. When audio outlasts video, the last frame is held for the audio tail.
- Each successful run creates a timestamped folder containing `merged.mp4`, `editing-plan.json`, and `processing-report.json`. The report includes source paths, skipped-file warnings, cut and reference times in seconds, visual confidence scores, and retained ranges. The scores are similarity measures. Clip indices are zero-based. Completed output is published only after FFprobe verifies the export and the report is written.
- All core analysis and export run locally. Original clips are opened for reading. Cancel stops active FFmpeg operations and cleans temporary work. Processing history is loaded from completed reports under the selected output folder.

## Cumulative revisions

Enable **Prefer newest cumulative revision** in Settings when the input folder contains successively longer exports of one story. The app checks each neighboring pair for a long shared visual opening and matching audio samples. If every pair passes, it exports the newest complete version and records the earlier versions it superseded. The newer version can replace an alternate ending. This option starts off. If the checks fail, the app runs its regular duplicate detector.

## Channel watermark

Open **Settings → Choose logo image** and select a PNG, JPEG, or WebP file. Use a transparent PNG to keep the background visible around your logo. Choose a corner, size and opacity, then click **OK**. The app remembers your selection for future exports. The watermark appears throughout the merged video, including across transitions. **Clear watermark** disables it.

## Optional local speech analysis

Install `requirements-speech.txt` using the same virtual environment. Obtain a [faster-whisper CTranslate2 model](https://github.com/SYSTRAN/faster-whisper) separately and select its local directory under **Settings**. Speech analysis runs on the CPU and compares recognized words in candidate repeats as an additional check. The app loads the supplied model locally.

```powershell
.\.venv\Scripts\python.exe -m pip install -r requirements-speech.txt
```

## Tests

Run the suite with both FFmpeg executables available. Tests generate deterministic videos in temporary directories.

```powershell
.\.venv\Scripts\python.exe -m pytest -ra
```

## Windows executable with PyInstaller

Build on 64-bit Windows from the virtual environment. The packaging script uses the [PyInstaller one-folder distribution](https://pyinstaller.org/en/stable/operating-mode.html#bundling-to-one-folder).

```powershell
.\.venv\Scripts\python.exe -m pip install "PyInstaller>=6,<7"
powershell -ExecutionPolicy Bypass -File .\package.ps1
```

The executable is `dist\AI Video Merger Agent\AI Video Merger Agent.exe`. Keep the entire folder together. Put FFmpeg and FFprobe in a `bin` directory beside the executable or configure PATH. For the optional speech build, install the speech requirements first and run the packaging script with `-IncludeSpeech`. Keep the local model directory available and select it in Settings.

## Troubleshooting

| Symptom | Action |
| --- | --- |
| FFmpeg or FFprobe was not found | Check the project or executable's `bin` directory. For a PATH install, run `ffmpeg -version` and `ffprobe -version` from a newly opened PowerShell window. |
| A clip was skipped | Read **Diagnostics** for its filename and decoder error. Check that the clip plays locally, then re-export or replace it if damaged. |
| Clips are in the wrong order | Prefix every filename with its sequence number, such as `01-intro.mp4` and `02-next.mp4`, then select the folder again. |
| Hardware encoding fails | Switch the encoder to **CPU · x264** and retry. GPU modes require a compatible FFmpeg build, GPU and driver. |
| Model fails to load | Install the optional dependencies and select the extracted CTranslate2 model directory, or disable speech analysis. |
| Some repetition remains | Check the processing report. The default thresholds favor preserving changed footage and audio. Lowering thresholds can increase removals, so review the resulting video. |
| Export fails or runs out of space | Check **Diagnostics** and free space in both the Windows temporary directory and the selected output folder. Retry with a writable local output folder. |

