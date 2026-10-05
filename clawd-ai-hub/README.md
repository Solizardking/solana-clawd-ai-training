# Clawd AI Hub custom domain

Serves the official `solanaclawd/clawd-ai` Hugging Face Space at
https://hub.onchainai.fund using an exact copy of the Space's static files.
`source.json` records the deployed Hub revision. The catalog still refreshes
live from the Hugging Face API every 60 seconds.

Deploy from this directory:

```sh
python3 sync_space.py
vercel --prod --scope clawd-c4b28c7e
```

The Vercel project is `clawd-ai-hub`. Its production custom domain is
`hub.onchainai.fund`. Run the sync and deployment commands after publishing UI
changes to the Space. No Hub token is needed for these public static files.
