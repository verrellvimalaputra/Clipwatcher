FROM python:3.12-slim

RUN apt-get update \
 && apt-get install -y --no-install-recommends ffmpeg fonts-dejavu-core ca-certificates libglib2.0-0 \
 && rm -rf /var/lib/apt/lists/*

# Node: yt-dlp needs a JavaScript runtime for YouTube, and Claude Code runs on it
COPY --from=node:22-bookworm-slim /usr/local/bin/node /usr/local/bin/node
COPY --from=node:22-bookworm-slim /usr/local/lib/node_modules/npm /usr/local/lib/node_modules/npm
RUN ln -s /usr/local/lib/node_modules/npm/bin/npm-cli.js /usr/local/bin/npm

# Claude Code CLI: moment picking + title suggestions on your Claude subscription (no API key).
# Build with --build-arg WITH_CLAUDE=0 to leave it out.
ARG WITH_CLAUDE=1
RUN if [ "$WITH_CLAUDE" = "1" ]; then npm install -g @anthropic-ai/claude-code && npm cache clean --force; fi

WORKDIR /app
COPY requirements.txt .
RUN pip install --no-cache-dir -r requirements.txt
COPY clipwatcher ./clipwatcher

# Whisper + face models are cached in the data volume, so they download once
ENV HF_HOME=/data/.models PYTHONUNBUFFERED=1 DATA_DIR=/data HOST=0.0.0.0 PORT=8080
VOLUME /data
EXPOSE 8080

ENTRYPOINT ["python", "-m", "clipwatcher"]
CMD ["studio"]
