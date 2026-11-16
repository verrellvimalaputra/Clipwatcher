# clipwatcher studio

Two ways to get clips, one place to review and post them:

- **Twitch live:** watches channels, records them while they're live, reads
  chat, and cuts a clip whenever chat spikes.
- **Import a video:** paste a YouTube link (or upload a file). It transcribes
  the video, finds the moments most likely to go viral, and turns each one into a
  post-ready vertical clip with a crop that follows the speaker's face and captions
  that highlight each word as it's spoken.

Clips wait in a web studio where you review them. Pressing **Approve** posts a
clip to YouTube Shorts, Instagram Reels and TikTok.

```
┌ Streamers ──────┬ Clips ─────────────────────┬ Review ───────────────────┐
│ ● papaplatte  2 │ [To review] Approved  Bin  │  ▶ video (9:16)           │
│   live · ~~~^~  │ ▯ ▯ ▯ ▯                    │  chat heat strip + trim   │
│ ○ trymacs       │ vertical thumbnails        │  why it clipped           │
│ ● windah      1 │                            │  title · caption          │
│ + add channel   │                            │  ☑YT ☑IG ☑TT              │
│ Settings        │                            │  [Reject] [Approve+post]  │
└─────────────────┴────────────────────────────┴───────────────────────────┘
```

## Quick start

```bash
# 1. cp .env.example .env, then set STUDIO_PASSWORD (and later PUBLIC_URL) in .env
docker compose up -d --build
# 2. open http://localhost:8080 (any username + your password)
# 3. add a Twitch channel in the sidebar
```

Want to try the whole approve-and-post flow first without really posting?
Set `CLIPWATCHER_DEMO=1` in `.env`.

Without Docker: Python 3.10+, ffmpeg with libass, then
`pip install -r requirements.txt && python -m clipwatcher studio`.

## Using the studio

- **Streamers:** add as many as you like. A red dot means live. The yellow line
  is chat activity and the dashed line is the spike threshold. **Edit** pauses a
  channel or changes its sensitivity (for big channels, raise "Min").
- **Review:** the heat strip under the video shows chat activity across the
  clip: the red line is the spike, the dashed line is normal chat. Drag the white
  handles to trim and pick a layout (**Fit**: whole frame on a blurred 9:16
  background; **Crop**: center 9:16; **16:9**), then press **Apply**. Trimming
  always starts from the raw footage, so you can widen a trim again later.
- **Keys:** `J`/`K` next/previous, `A` approve, `X` reject, `Space` play/pause.
- **Approve:** the clip is queued and uploaded in the background. The Approved
  tab shows each platform's status, a link once posted, and **Try again** if it failed.
- **Suggest titles** (optional): Claude suggests 3 titles from what was said
  plus the chat reaction. This runs through **Claude Code on your Claude
  subscription** (Pro/Max), not the paid API. See below.

## Importing videos

Open **Import a video**, paste a link, pick clip length / count / layout and
press **Find clips**. The job list shows each step (downloading, transcribing,
finding moments, editing). When it's done, **Review N clips** opens them.

- **How moments are picked:** with Claude Code set up (see below), Claude reads
  the whole transcript and picks moments that start on a hook and end on a
  payoff, and titles each one. YouTube's "Most replayed" graph and, for past
  livestreams, the chat replay are given to it as hints. Without Claude Code,
  a fallback picks from those signals plus loudness.
- **Layouts:** *Auto* follows the speaker's face in a 9:16 crop (great for
  podcasts and talking heads) and falls back to *Fit* when there are no faces,
  e.g. gameplay.
- **Speed:** transcription runs on your CPU and is the slow part. For long
  videos use Settings → Clips → Caption accuracy "Fast" or "Balanced".
- **If YouTube downloads fail** with "Sign in to confirm you're not a bot",
  export your YouTube cookies (a browser extension like "Get cookies.txt
  LOCALLY") to `data/cookies.txt` and set `/data/cookies.txt` under
  Settings → Imports. When downloads start failing after a YouTube change,
  update yt-dlp: `docker compose exec studio pip install -U "yt-dlp[default]"`.
- **Rights:** importing works best on **your own videos**, or ones you have
  permission to clip. Captions include the source link by default.

## Connecting accounts

Settings → Accounts walks you through each one. Every platform needs a free
developer app of your own. Copy the redirect URL the studio shows into the app,
paste the app's ID and secret, then press Connect. **Open the studio at the
same address you registered** (e.g. don't mix `localhost` and `127.0.0.1`).

What to expect from each platform. These rules are set by the platforms
themselves, not by this app:

| | Works right away | Limit until the platform reviews your app |
|---|---|---|
| **YouTube** | Uploads as Shorts (vertical, with #Shorts) | Projects that haven't passed the YouTube API audit may have uploads locked to private. Apps in "Testing" mode: tokens expire after 7 days, so reconnect weekly (or publish the consent screen). |
| **Instagram** | Posts Reels publicly | Account must be Business or Creator. Instagram downloads the video from your studio, so it needs `PUBLIC_URL`. |
| **TikTok** | **Inbox mode** (default): the clip lands in your TikTok inbox as a draft and you tap Post in the app | **Direct** posting stays "only me" and needs a private account until TikTok audits your app. TikTok also requires an HTTPS redirect, so set `PUBLIC_URL` first. |

### Claude Code: moment picking + title suggestions (no API key)

The Docker image includes the Claude Code CLI. Log it in to your subscription
once:

1. On any computer with a browser and Claude Code installed, run
   `claude setup-token`. It opens a browser login and prints a long-lived token
   (valid about a year).
2. Paste it into `CLAUDE_CODE_OAUTH_TOKEN` in `.env`, then run
   `docker compose up -d`.

Once Claude Code is logged in, imports use it to pick moments and the "Suggest
titles" button appears. Both use your plan's usage limit (an import of a 1-hour
video is roughly one long message), with no extra bill. The studio
deliberately strips `ANTHROPIC_API_KEY` from Claude Code's environment, so it
can never switch to API billing by accident.

If the token gets rejected (some Claude Code versions have had bugs with
`setup-token` tokens), log in directly inside the container instead. The
login is kept in `data/.claude`:

```bash
docker compose exec -it studio claude   # then type /login and follow the link
```

### PUBLIC_URL

Instagram and TikTok need to reach the studio over HTTPS. Options:

- **VPS with a domain:** put Caddy or nginx with HTTPS in front of port 8080 and set `PUBLIC_URL=https://studio.yourdomain.com`.
- **Running at home:** `docker compose --profile tunnel up -d` starts a free
  Cloudflare quick tunnel. `docker compose logs tunnel` shows the
  `https://….trycloudflare.com` address. Note that it changes on every restart,
  so a named tunnel with your own domain is better long term.

Only approved clip files are exposed publicly, through unguessable links.
Everything else stays behind your password.

## Data

```
data/
  studio.db                 channels, clips, posting status, settings, tokens
  recordings/*.ts           full recordings (deleted after 48 h by default)
  clips/*.mp4 *.jpg         finished clips + thumbnails
  clips/*.src.ts            raw footage per clip (used for re-trimming)
  chat/<channel>.jsonl      all chat, for tuning
```

Tuning a channel offline:
`python -m clipwatcher replay data/chat/<channel>.jsonl --ratio 3 --min-score 8`.
Aim for about 5–15 spikes per stream hour.

The single-channel headless mode from before still works:
`python -m clipwatcher run CHANNEL --vertical`.

## Be fair to the streamer

Every caption credits and links the streamer by default (edit the template in
Settings). Ask before posting, or join their clipping program, especially if the
account makes money. Their content is theirs.
