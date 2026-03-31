#!/usr/bin/env python3
"""
One-time OAuth authorization for Google Tasks (headless/remote-friendly).

Extracts client_id and client_secret from the existing credentials.json,
then runs a manual redirect flow:
  1. Prints a URL to open in any browser
  2. You paste back the redirect URL (localhost — won't load, that's OK)
  3. Saves token_tasks.json
"""

import os
import sys
import json

sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))

from config import TASKS_CREDS_FILE, TASKS_TOKEN_FILE, TASKS_SCOPES

REDIRECT_URI = "http://localhost:8085"


def main():
    if not os.path.exists(TASKS_CREDS_FILE):
        print(f"Error: {TASKS_CREDS_FILE} not found.")
        sys.exit(1)

    with open(TASKS_CREDS_FILE) as f:
        existing = json.load(f)

    client_id = existing.get("client_id")
    client_secret = existing.get("client_secret")
    if not client_id or not client_secret:
        print("Error: credentials.json missing client_id or client_secret.")
        sys.exit(1)

    if os.path.exists(TASKS_TOKEN_FILE):
        print(f"Token already exists: {TASKS_TOKEN_FILE}")
        resp = input("Overwrite? [y/N] ").strip().lower()
        if resp != "y":
            print("Aborted.")
            sys.exit(0)

    from urllib.parse import urlencode, urlparse, parse_qs
    import requests

    scopes_str = " ".join(TASKS_SCOPES)
    auth_params = {
        "client_id": client_id,
        "redirect_uri": REDIRECT_URI,
        "response_type": "code",
        "scope": scopes_str,
        "access_type": "offline",
        "prompt": "consent",
    }
    auth_url = "https://accounts.google.com/o/oauth2/auth?" + urlencode(auth_params)

    print()
    print("Open this URL in any browser and authorize:")
    print()
    print(auth_url)
    print()
    print(f"After granting access, you'll be redirected to {REDIRECT_URI}/?code=...")
    print("The page won't load — that's expected. Copy the FULL URL from the address bar.")
    print()
    redirect_url = input("Paste the redirect URL here: ").strip()

    parsed = urlparse(redirect_url)
    qs = parse_qs(parsed.query)
    code = qs.get("code", [None])[0]

    if not code:
        print("Error: could not extract authorization code from URL.")
        print("Make sure you pasted the full redirect URL including ?code=...")
        sys.exit(1)

    token_resp = requests.post(
        "https://oauth2.googleapis.com/token",
        data={
            "code": code,
            "client_id": client_id,
            "client_secret": client_secret,
            "redirect_uri": REDIRECT_URI,
            "grant_type": "authorization_code",
        },
    )

    if token_resp.status_code != 200:
        print(f"Token exchange failed: {token_resp.status_code}")
        print(token_resp.text)
        sys.exit(1)

    token_data = token_resp.json()

    creds_data = {
        "token": token_data["access_token"],
        "refresh_token": token_data.get("refresh_token"),
        "token_uri": "https://oauth2.googleapis.com/token",
        "client_id": client_id,
        "client_secret": client_secret,
        "scopes": TASKS_SCOPES,
    }

    with open(TASKS_TOKEN_FILE, "w") as f:
        json.dump(creds_data, f, indent=2)

    print(f"\nToken saved to {TASKS_TOKEN_FILE}")
    print("Google Tasks plugin is now ready to use.")


if __name__ == "__main__":
    main()
