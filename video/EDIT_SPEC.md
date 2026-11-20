# Visual Radio videos: edit spec

Footage: `C:\projects\visual-radio-video\captures\` (mp4 + json logs) and `hardware\hardware_landscape.mov`, `hardware_portrait.mov`.
Output: `C:\projects\visual-radio-video\output\`.

## Tools in this folder

- `gfx.html` + `render.js`: animated graphics. `node render.js <scene> <W> <H> <seconds> <outdir> [arg]` writes PNG frames at 30 fps
  (solid background). With `seconds` = 0 it writes one transparent `frame.png` (overlays). Needs Playwright + Chromium
  (the recorder venv has Python Playwright; for node run `npm i playwright@1.49` in this folder and `npx playwright install chromium`,
  or port render.js to Python Playwright, same logic). `peek.js` renders one still for checking.
  Fonts: Inter + JetBrains Mono must be installed (or loaded from Google Fonts in gfx.html); Fraunces loads from Google Fonts.
  - Animated scenes: `title` (5 s), `sdr` (9 s), `songid` (7 s), `pipeline` (7 s), `endcard` (7 s). Render each at 1920x1080 and 1080x1920.
  - Overlays (seconds = 0): `strip` with arg 0..6 (pipeline strip, step highlighted: 0 Antenna, 1 SDR, 2 FM/DAB+,
    3 Talk or music, 4 Isolate, 5 Whisper, 6 Song); `caption` with URL-encoded text (promo captions); `chapter` with text.
- `voiceover.json` + `tts_batch.py`: English AI voice (Kokoro, voice af_heart). Produces `<group>_<key>.wav`.

## Audio rules (all videos)

- Song audio: never more than ~8 s of one song unbroken (Content ID / Instagram muting). Talk audio can run longer.
- Under voice-over, duck the radio to about -18 dB; restore over 0.4 s after each line.
- Loudness: final mix normalised to -14 LUFS (YouTube/Instagram), true peak -1 dB. Short 0.3 s audio crossfades at cuts.
- Video: H.264 High, CRF 18, 30 fps, yuv420p, AAC 192k, `-movflags +faststart`. 0.3 s crossfades or hard cuts on the beat.

## 1. Long demo, landscape 1920x1080, about 5 minutes → `visual-radio-demo.mp4`

Voice-over keys are from `voiceover.json` group `long`. Times are source times in the capture files; check them against
the json logs and adjust by a second or two so cuts land on sentence boundaries. Pipeline strip overlay (`strip N`)
sits on top during chapters 3-8, fading in/out over 0.3 s.

| # | Chapter | Picture | Sound / VO |
|---|---|---|---|
| 1 | Cold open (~25 s) | L1_bnr_talk: ~10 s with a speaker colour change (pick from log). Hard cut to L3_538_songs_part2 ~01:08-01:24 (talk turns into "Radar Love", page flips to the rock scene). | Radio audio; VO `c01` starts ~2 s in, ducked radio; let ~6 s of the song play loud after the flip. |
| 2 | Title (~12 s) | `title` animation (5 s), then L3 part1 00:00-00:07 (Angels lyrics, cover). | VO `c02` over both; song ducked. |
| 3 | SDR (~16 s) | hardware_landscape.mov (best 5-6 s, slow), then `sdr` animation (9 s). strip 0 → 1. | VO `c03a` over hardware, `c03b` over sdr. Quiet soft room tone, no radio. |
| 4 | FM tuning (~25 s) | L4_picker_dab 00:05-00:48 sped up 1.6x where the pointer moves; keep the card scroll at 1x. Do NOT show the empty DAB+ view (cut before 00:50). Slow zoom (1.0→1.15) on the RDS "Now" line in a few seconds of L3. strip 2. | VO `c04a`, `c04b`. |
| 5 | Talk or music (~20 s) | L3 part3 ~02:18-02:40: talk turns into Armin van Buuren, page goes ELECTRONIC. Zoom 1.0→1.4 toward the status badge in the top right as it changes. strip 3. | VO `c05` over the talk part; song loud for ~5 s after the flip. |
| 6 | Isolate (~40 s) | L5_538_voice_music: 03:19-03:24, Listen 03:24, Voice 03:29-03:39, Music 03:47-03:55, Original 04:05-04:10. Small label top left via `chapter` overlay: "Original", "Voice only", "Music only". strip 4. | VO `c06` over the first part (duck); then the voice-only and music-only audio play un-ducked, 8-10 s each. |
| 7 | Whisper (~40 s) | L1_bnr_talk ~15 s with clear word-by-word highlight and at least one colour change. Then L2_bnr_english 00:04-00:17 (panel opens, EN clicked) and 00:39-00:52 (Dutch talk now English). strip 5. | VO `c07a` + `c07b` over L1, `c07c` over L2. |
| 8 | Which song (~16 s) | `songid` animation (7 s), then L3 part1 a few seconds of the cover on the spinning record. strip 6. | VO `c08`; short song audio under. |
| 9 | Scenes montage (~32 s) | Hard cuts on the beat, ~5 s each, with a `chapter` label of the genre: Pop = L3 part1 (Angels), Rock = L3 part2 ~01:25 (Radar Love), Electronic = L3 part3 ~02:40 (Armin), Jazz & Soul = L7_feelgood 00:13 (Marvin Gaye), Instrumental = L7_klassiek 00:28-00:40 (instrumental visual, do not show its "Pop" badge large; crop or skip if it is prominent). | VO `c09` at the start, then each clip's own audio loud (5 s each, so no song exceeds 8 s). |
| 10 | Engine room (~14 s) | L6_engine_room 00:04-00:20. | VO `c10`, radio ducked. |
| 11 | Anywhere (~8 s) | Composite: L3 (landscape) scaled to 70% on the left, P1 portrait clip at the same moment-ish as a phone with rounded corners on the right, on the dark gradient background. | VO `c11`. |
| 12 | Call to action (~14 s) | `pipeline` animation (7 s) then `endcard` (7 s, hold the last frame 3 s more). | VO `c12`; soft song tail ducked, fade out. |

Also export `visual-radio-demo-thumbnail.png` (1280x720): the Radar Love rock frame with "SEE what the radio says" in big Inter ExtraBold.

## 2. Promos, portrait 1080x1920 (9:16)

Captions via the `caption` overlay (text in quotes below), on screen 0.2 s before the matching VO line until the next one.
Each ends on `endcard` portrait (3 s, start at its t=1 s so it is already readable). Export both as one piece (`-short.mp4`, for YouTube Shorts)
and, for any promo over 15 s, also cut into 15 s parts for Instagram Stories (`-story-1.mp4`, `-story-2.mp4`).

### Promo A: "The drop" (15 s) → `promo-a-drop`
1. 0-4 s: P1_538_portrait_part2 just before 01:26 (talk, teal captions). Caption "Radio. But you can see it." VO `a1`.
2. 4 s: hard cut on the first beat of November Rain (~01:26): page flips to music. Then jump to the first sung lyric line of November Rain (find it in the log), 4-6 s. Caption "Live radio. Every word. Every song." VO `a2`.
3. 12-15 s: endcard portrait. VO `a3`.

### Promo B: "Every song gets a stage" (20 s) → `promo-b-stages`
Landscape clips presented in a vertical frame: a blurred, darkened, zoomed copy of the clip fills the 1080x1920 background;
the clip itself sits full-width (1080x608) in the middle with 24 px rounded corners; the genre name in big type above it.
1. 0-2 s: caption "Your radio has moods." VO `b1`. Background: first clip blurred.
2. 2-17 s: 5 clips x 3 s, hard cuts on the beat, each with its own audio: POP (L3 part1 Angels), ROCK (L3 part2 Radar Love), ELECTRONIC (L3 part3 Armin), JAZZ & SOUL (L7_feelgood Marvin Gaye), INSTRUMENTAL (L7_klassiek visual).
3. 17-20 s: endcard. VO `b2`.

### Promo C: "Pull the voice out" (25 s) → `promo-c-voice`
1. 0-3 s: P3_538_voice_portrait 00:00-00:03 (Angels lyrics, full band). Caption "Watch this." VO `c1`.
2. 3-11 s: P3 from ~00:11 (Voice tapped) to ~00:19: the band drops away. Caption "The band disappears. Just the voice." VO `c2` (duck only lightly so the isolated voice is clearly heard).
3. 11-18 s: L1_bnr_talk in the vertical frame style of promo B (a stretch with a colour change). Caption "Who's talking? It knows." VO `c3`.
4. 18-22 s: `pipeline` portrait (play at 1.75x so all steps light up in 4 s). Caption "All on your own PC. No cloud." VO `c4`.
5. 22-25 s: endcard. VO `c5`.

## Checks before delivery

- Watch every output once, start to end. Check: no song longer than ~8 s unbroken, VO never collides with a word that is highlighted on screen in a way that is confusing, captions inside the 9:16 safe area (top 250 px and bottom 400 px stay clear of important text because of the Instagram/Shorts UI), no empty DAB+ list shown, no mouse cursor except during clicks.
- Grab 3 still frames per video into `output\stills\` for review.
- Write `output\README.txt`: file, length, size, and a one-line description.
