# clipwatcher

Auto-clipper with a review studio. Two sources feed one review queue:

1. **Twitch live:** records live streams, detects moments from chat spikes, cuts clips.
   Live channels can also be **YouTube Live** (a live video link or an @channel); streamlink records both.
2. **Video import (Opus Clip style):** paste a YouTube link (any yt-dlp site) or
   upload a file → transcribe → Claude Code picks the viral moments → post-ready
   9:16 clips with face-tracking reframe and word-highlight captions.

The owner reviews clips in the web "studio". Approving a clip posts it to
YouTube Shorts, Instagram Reels and TikTok.

Owner: Verrell. He may write in Indonesian or English; reply in the language
he uses. He runs it on a Windows laptop with Docker Desktop and may later move
it to a VPS. He wants no paid Claude API usage: AI features go through the
Claude Code CLI on his subscription (see "Title suggestions").

## Run / test

```bash
pip install -r requirements.txt pytest       # Python 3.10+, needs ffmpeg with libass on PATH
python -m pytest -q tests                    # fast, no network
python scripts/seed_demo.py data-demo        # fake channels + rendered clips
CLIPWATCHER_DEMO=1 python -m clipwatcher studio --data data-demo   # http://127.0.0.1:8080
docker compose up -d --build                 # the real deployment (studio on :8080)
python -m clipwatcher replay data/chat/<ch>.jsonl --ratio 3 --min-score 8   # tune detection
```

`CLIPWATCHER_DEMO=1` swaps every publisher for a fake one: the full approve and
post flow runs, but nothing is uploaded. Use it for any UI or flow work.

## Architecture (package `clipwatcher/`)

| File | Role |
|---|---|
| `detector.py` | Pure spike logic, no I/O. Third trigger, for big always-busy chats (YouTube Live): `REACT_RE` reacting chatters ≥ `react_ratio` (2.6) × their own 5-min baseline and ≥ `react_min` (25) → reason `reactions` (calibrated on a real Windah Basudara log: volume flat at ~45/10 s, 'gg' waves 3x; roll calls like 'aku' must NOT match). Score = unique chatters in 10 s + 0.5 per chatter using hype words. Spike if score ≥ ratio × mean of the previous 5 min AND ≥ min_score, or if ≥3 people type "clip". Ignores known bots and `!commands`; cooldown 60 s; warmup 2 min. The same class runs live and in `replay`. |
| `channels.py` | `parse_channel()`: what the user pastes → `{name, platform, url, label}`. Twitch names stay as-is; YouTube becomes `yt-<videoId>` (a live video) or `yt-<handle>` / `yt-<UCid>` (a channel, recorded via `…/live` whenever it goes live). `name` is used in file names, so no `:`/`/`. `public_link()` = credit link for captions. |
| `ytchat.py` | YouTube Live chat without an API key: runs yt-dlp's own `YoutubeLiveChatFD` in-process and hooks `_append_fragment` to receive each batch in real time; message time = its `timestampUsec`. Status text in `.note` (live / not live / ended → use Import / chat off / bot check). If YouTube changes, updating yt-dlp is the fix; `tests/test_youtube_live.py` checks the hook points still exist. |
| `chat.py` | Anonymous read-only IRC (`justinfan#####`, irc.chat.twitch.tv:6667). Uses **local receive time** (same clock as the recorder) and appends to `data/chat/<ch>.jsonl`. |
| `recorder.py` | `streamlink --stdout` piped into our own file, with a **wall-clock → byte-offset index** per session. Clips are cut by copying bytes between two wall-clock times. Never a fixed offset into the file: streamlink drops Twitch ads, which leaves gaps, and the index keeps alignment exact. Offsets snap to 188-byte TS packets. |
| `editing.py` | Shared editing toolkit. `ClipConfig`; `transcribe_words` (faster-whisper, word timestamps, progress callback) → `lines_from_words` (≤4 words, breaks on sentence end/pauses) → `write_ass` (big bold captions, the spoken word highlighted amber, UPPERCASE in vertical); `face_track` (YuNet face detector, model auto-downloaded to `$HF_HOME/yunet.onnx` + sha256 check, Haar fallback on OpenCV 4) → `smooth_track` (virtual camera: median filter, dead zone, easing) → crop x via ffmpeg `sendcmd`; `render(src, out, cfg, captions, start, end)` returns the layout actually used. Layouts: `split` (default for Twitch: streamer's facecam on top, center of the gameplay below, captions on the seam; `detect_facecam` = faces that stay put across the clip, then the cam's own frame = the persistent edge with calm pixels inside and moving gameplay outside (time-averaged Sobel + temporal std); top panel height follows the cam's aspect (`split_top_height`, 540–960 px); no facecam → `fit`. Box precedence: explicit > clip's `facecam` > channel's saved `kv facecam:<channel>` > detect. The studio's facecam editor (frame from `/api/clips/{id}/frame`) saves per channel), `auto` (face-tracked 9:16, falls back to `fit` if <30% of samples have a face), `fit` (frame over blurred 9:16 copy), `crop` (center), `original`. |
| `clipper.py` | Twitch, per channel: wait → extract a LONG raw slice (`RAW_PRE` 90 s before the spike, `RAW_POST` 20 s after; `*.src.ts`, kept for re-renders/wider trims) → rough transcript of it with Whisper `base` → `moments.twitch_cut` (Claude Code reads transcript + chat around the spike and picks start of the setup / end of the payoff + title + context line; validated, else `_sentence_cut` = pre/post window moved to sentence edges) → accurate captions with the configured model on the chosen part only (`_captions`) → render with the context banner → `on_clip(meta)`. |
| `moments.py` | Moment finding for imports. Signals: YouTube "Most replayed" `heatmap` (from yt-dlp info), live-chat replay (`*.live_chat.json` parsed and run through `SpikeDetector`), loudness peaks (RMS vs the surrounding minute). `claude_moments` sends the timestamped transcript + signals to `claude -p` on **stdin** (transcripts exceed argv limits), 1 h windows for long videos, and asks for JSON `{clips:[{start,end,title,why,score}]}`. `heuristic_moments` is the no-AI fallback (rewards signals in the middle-to-late part of a window). Then `snap` (boundaries → sentence edges ±2.5 s, else word edges; enforce length), `dedupe`, top N. Lengths: short 15–35, medium 30–60, long 55–90 s. |
| `importer.py` | `ImportWorker` thread, one job at a time: `yt-dlp -J` (metadata, cached in `source.info.json`) → `download_mode()` picks `full` or `clips` (option `download`: auto/full/clips; auto = clips when duration ≥ 20 min; uploads and unknown durations are always full). **full:** download ≤1080p mp4 (`--js-runtimes node`, optional `--cookies`, live chat if `was_live`) or use the uploaded file. **clips:** download only the chat replay (`--skip-download --write-subs --sub-langs live_chat`), use YouTube captions, or download audio only (`audio.*`) to transcribe when there are none; each moment is fetched with `--download-sections "*a-b"` (exact to the frame, verified) into `section_NN.*`, which replaces slicing the full video; heat strip uses chat msgs (`chat_msgs_heat`) when there's no audio → rough full-video transcript for moment finding: YouTube's own captions (`pick_caption_track` = spoken language only, never a translation; `parse_json3`) or, if none, Whisper `base` (cached in `imports/<id>/transcript.json` with its `source`) → find moments → per moment: exact re-encoded slice with 3 s margins (`job<id>_NN.src.mp4`) → render → clip row (`source_kind='video'`, `job_id`, `why`, `layout`, trim_start/end inside the slice, heat = replay heat or loudness). Deletes the full source video when done. Cancel kills yt-dlp; restarts re-queue interrupted jobs. |
| `supervisor.py` | One `ChannelWatcher` (recorder, chat, detector loop, clipper threads) per enabled channel. `sync()` makes running watchers match the DB. Records a chat-score history (every 2 s) that becomes each clip's `heat` curve. Housekeeping deletes full recordings after `keep_recordings_hours`. |
| `db.py` | SQLite (WAL, one lock). `_migrate()` adds new columns to old DBs (the owner already has a DB, so never rename/drop columns). Tables: `jobs` (imports: status queued/uploading/downloading/transcribing/finding/rendering/done/failed/cancelled, step text, progress 0..1, options JSON, finder claude/heuristic), `channels`, `clips` (status pending/approved/rejected, trim_start/trim_end relative to the raw slice, captions/heat/chat_sample as JSON), `posts` (one row per clip × platform: queued/uploading/posted/failed), `kv` (settings `setting:*`, app creds `cred:<p>`, tokens `token:<p>`, OAuth states, a secret). `DEFAULT_SETTINGS` lists every setting. |
| `publishers.py` | `YouTube`, `Instagram`, `TikTok` classes: OAuth URL, code exchange, refresh, `publish()`. `PostWorker` thread drains queued posts one at a time. `Fake` is used in demo mode. |
| `reviewer.py` | AI reviewer: `ReviewWorker` thread scores pending clips 0-10 with a separate `claude -p` call that only sees the final cut (trimmed transcript relative to the cut, title, context line, chat sample), never the picker's reasoning. Also gets `frames.jpg`, a 3x2 contact sheet of 6 stills from the rendered clip (`contact_sheet`, ~1k image tokens): `run_claude(files=...)` copies it into the empty cwd and enables only `--tools Read --allowedTools Read(./frames.jpg)`. Model `CLAUDE_REVIEW_MODEL` (default `haiku`). JSON `{interesting, funny, clear, score, verdict, fix, saw_frames}` → `clips.ai_score` / `ai_review`. A bad answer increments `ai_tries` (gives up at 2); a Claude failure pauses the worker 5 min without counting. Re-trimming resets the score; a result for an outdated trim is dropped. Setting `ai_review`. |
| usage | Every `run_claude` call passes `task=` (cut, moments, review, titles, check); the studio sets `moments.USAGE_HOOK = db.add_usage`, which sums the CLI's `usage`/`total_cost_usd` into table `ai_usage` per day+task+model. `GET /api/ai-usage`, shown in Settings → Claude Code. |
| `web.py` | FastAPI app + JSON API (`/api/clips/{id}/render` with layout/trim/facecam, `/api/clips/rerender` = background bulk re-render of clips waiting for review; `/api/imports*` for jobs, `/api/imports/upload` streams multipart to disk). Optional HTTP Basic auth (`STUDIO_PASSWORD`, applies to everything except `/public/`). `/public/<hmac>/<file>` serves clip files without login (Instagram fetches videos from there). `/auth/<p>/start` and `/auth/<p>/callback` handle OAuth. |
| `static/index.html` | The whole studio UI: one file, vanilla JS, no build step. It polls `/api/state` every 4 s and the clip list every 6 s. |
| `__main__.py` | CLI: `studio`, `run` (old headless single-channel mode, writes `.json` sidecars), `replay`. |

Data dir: `studio.db`, `recordings/*.ts`, `clips/*.mp4|.jpg|.src.ts|.src.mp4`, `chat/*.jsonl`, `imports/<job>/` (info, transcript, chat replay), `.models/` (Whisper + YuNet, via `HF_HOME`), `.claude/` (Claude Code login).

Context line: one short sentence per clip (`clips.context`, written by Claude for Twitch cuts and imports, editable in review) burned in by `write_ass` as an amber banner at the top for `context_secs` (3.5 s); setting `context_overlay` turns it off.

Caption template variables: `{title}`, `{channel}`, `{link}` (Twitch channel or source video URL with `t=`), `{tag}` (channel as a hashtag). For uploads (no link) the lines with `{link}`/`{channel}` are dropped.

## Platform rules (enforced by the platforms, verified Nov 2026)

- **YouTube:** uploads from a Google Cloud project that hasn't passed the YouTube API audit can be locked to private. OAuth apps in "Testing" mode get refresh tokens that expire after 7 days. Scopes: `youtube.upload` + `youtube.readonly`. Uses a resumable upload; adds `#Shorts` to the title for vertical layouts.
- **TikTok:** unaudited apps can only Direct Post as `SELF_ONLY`, only to private accounts, max 5 users per 24 h. So the default is **inbox mode** (`/v2/post/publish/inbox/video/init/`, scope `video.upload`): the video lands in the user's TikTok inbox as a draft. Direct mode queries `creator_info` first. Upload chunking: whole file if ≤64 MB, otherwise 10 MB chunks with the remainder in the last chunk. TikTok needs an HTTPS redirect URI.
- **Instagram:** uses "Instagram API with Instagram Login" (graph.instagram.com, `IG_GRAPH_VERSION` env, default v25.0). Business or Creator account required. Reels are created from a **public `video_url`**, so the studio needs `PUBLIC_URL` (VPS, or the Cloudflare tunnel profile in docker-compose). Long-lived token valid 60 days; refreshed when <10 days are left.

## Claude Code usage (moment picking + title suggestions)

Both go through `moments.run_claude()` / the same pattern in `web.py`:
`claude -p <instructions> --output-format json --tools "" --model $CLAUDE_MODEL`
in an empty temp dir, with `ANTHROPIC_API_KEY` **stripped from the env**, so it
always uses the subscription. Auth: `CLAUDE_CODE_OAUTH_TOKEN` (from `claude setup-token`)
or a `/login` persisted in `~/.claude`. The UI button only shows when
`claude_available()` is true. **Do not add direct Anthropic API calls.**

## Conventions

- Copy in the UI: plain sentence case. Name actions by what happens ("Approve and post to 3", "Try again"). Errors say what went wrong and how to fix it.
- UI palette and type: aubergine graphite `#1e1b24`, amber "tally" `#f2b33d` for approve/active, red `#e5484d` for live/spike, Bricolage Grotesque + Instrument Sans. Clips are 9:16 tiles. The signature element is the chat heat strip under the player (trim handles live on it); keep it.
- Always pass Whisper a decoded array from `editing.load_audio()`, never a file path: faster-whisper's own decoder calls `av.open(metadata_errors=...)`, which newer PyAV removed.
- Captions never block a clip: failures are logged and the clip renders without captions.
- Re-renders always start from `src` (the raw slice), never from the last mp4.
- Credit the streamer: the default caption template links twitch.tv/<channel>. Keep that default.
- Keep `detector.py` free of I/O so `replay` and the tests keep working. Run `pytest` after touching detector, chat or recorder.

## Status

Tested in a Linux sandbox (`pytest`: 14 tests): detector (synthetic chat), byte-index cutting across
a simulated ad gap (frame-accurate), rendering/layouts/caption burn-in, the
studio UI end to end in demo mode (desktop + mobile screenshots), OAuth
redirect URLs, the public-link guard, the Claude Code suggest path (success
+ auth error), and the import pipeline end to end via upload (fake transcript,
both the Claude Code path with a stand-in CLI and the heuristic fallback),
YuNet face tracking + reframe on a synthetic moving face, DB migration of an
old database.

**Never tested against the real services:** live Twitch (streamlink + IRC were
unreachable from the sandbox), real YouTube downloads via yt-dlp (blocked in
the sandbox), real Whisper transcription (model download blocked), real Claude
Code moment picking on a long transcript, real uploads to YouTube/Instagram/TikTok, OAuth
callbacks with real codes, the Docker image build, and running on Windows/Docker
Desktop. Expect the first real run to surface bugs there.

Known rough edges / ideas:
- `@app.on_event` is deprecated in FastAPI (still works); could move to lifespan.
- Whisper runs on CPU (`int8`). `medium`/`large-v3` are slow on a laptop.
- The TikTok post URL is built as `/@/video/<id>` because we don't store the username.
- No per-clip layout is persisted (re-render uses the chosen one, the UI defaults to the global setting).
- - Whisper on CPU is slow (`small` ran ~0.75× realtime on the owner's laptop in Docker), hence the two-stage design: never run the big model over a whole video. Whisper uses `cpu_threads=os.cpu_count()`.
- yt-dlp breaks whenever YouTube changes things: `pip install -U "yt-dlp[default]"` is the usual fix.
- Possible next features: pick-best-clips ranking with Claude Code, scheduled posting times, per-platform caption overrides.
