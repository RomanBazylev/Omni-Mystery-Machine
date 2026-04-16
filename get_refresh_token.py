"""
One-time helper: generates a YouTube refresh token via local OAuth flow.

Usage:
  1. pip install requests
  2. Set env vars or pass as args:
       YT_CLIENT_ID, YT_CLIENT_SECRET
  3. Run:  python get_refresh_token.py
  4. Browser opens → sign in → authorize → token is printed
  5. Paste the refresh token into GitHub repo secret YT_REFRESH_TOKEN

Optional: auto-update GitHub secret (requires GITHUB_TOKEN with repo scope):
  python get_refresh_token.py --update-secret RomanBazylev/Omni-Mystery-Machine
"""

import http.server
import json
import os
import sys
import threading
import urllib.parse
import webbrowser

import requests

SCOPES = "https://www.googleapis.com/auth/youtube.upload https://www.googleapis.com/auth/youtube.readonly"
AUTH_URL = "https://accounts.google.com/o/oauth2/v2/auth"
TOKEN_URL = "https://oauth2.googleapis.com/token"
REDIRECT_PORT = 8976
REDIRECT_URI = f"http://localhost:{REDIRECT_PORT}"


def _get_credentials():
    client_id = os.getenv("YT_CLIENT_ID") or os.getenv("YOUTUBE_CLIENT_ID", "")
    client_secret = os.getenv("YT_CLIENT_SECRET") or os.getenv("YOUTUBE_CLIENT_SECRET", "")
    if not client_id:
        client_id = input("Enter YT_CLIENT_ID: ").strip()
    if not client_secret:
        client_secret = input("Enter YT_CLIENT_SECRET: ").strip()
    if not client_id or not client_secret:
        print("ERROR: Both CLIENT_ID and CLIENT_SECRET are required.")
        sys.exit(1)
    return client_id, client_secret


def _capture_auth_code() -> str:
    """Start a local HTTP server and wait for Google's redirect with the auth code."""
    auth_code = None
    error = None

    class Handler(http.server.BaseHTTPRequestHandler):
        def do_GET(self):
            nonlocal auth_code, error
            params = urllib.parse.parse_qs(urllib.parse.urlparse(self.path).query)
            if "code" in params:
                auth_code = params["code"][0]
                self.send_response(200)
                self.send_header("Content-Type", "text/html")
                self.end_headers()
                self.wfile.write(b"<html><body><h2>Authorization successful!</h2>"
                                 b"<p>You can close this tab and return to the terminal.</p></body></html>")
            elif "error" in params:
                error = params["error"][0]
                self.send_response(400)
                self.send_header("Content-Type", "text/html")
                self.end_headers()
                self.wfile.write(f"<html><body><h2>Error: {error}</h2></body></html>".encode())
            else:
                self.send_response(400)
                self.end_headers()

        def log_message(self, format, *args):
            pass  # Suppress server logs

    server = http.server.HTTPServer(("localhost", REDIRECT_PORT), Handler)
    server.timeout = 120  # 2 minute timeout

    thread = threading.Thread(target=server.handle_request, daemon=True)
    thread.start()
    thread.join(timeout=120)
    server.server_close()

    if error:
        print(f"ERROR: Authorization denied: {error}")
        sys.exit(1)
    if not auth_code:
        print("ERROR: Timed out waiting for authorization (120s).")
        sys.exit(1)
    return auth_code


def _exchange_code(client_id: str, client_secret: str, auth_code: str) -> dict:
    """Exchange authorization code for access + refresh tokens."""
    resp = requests.post(TOKEN_URL, data={
        "client_id": client_id,
        "client_secret": client_secret,
        "code": auth_code,
        "grant_type": "authorization_code",
        "redirect_uri": REDIRECT_URI,
    }, timeout=30)
    if not resp.ok:
        print(f"ERROR: Token exchange failed ({resp.status_code}): {resp.text}")
        sys.exit(1)
    data = resp.json()
    if "refresh_token" not in data:
        print("ERROR: No refresh_token in response. Try revoking access at")
        print("  https://myaccount.google.com/permissions")
        print("and running this script again.")
        sys.exit(1)
    return data


def _update_github_secret(repo: str, secret_name: str, secret_value: str):
    """Update a GitHub repo secret using the GitHub API."""
    gh_token = os.getenv("GITHUB_TOKEN", "")
    if not gh_token:
        gh_token = input("Enter GitHub personal access token (repo scope): ").strip()
    if not gh_token:
        print("SKIP: No GITHUB_TOKEN, cannot auto-update secret.")
        return False

    # Get repo public key for secret encryption
    headers = {"Authorization": f"Bearer {gh_token}", "Accept": "application/vnd.github+json"}
    key_resp = requests.get(f"https://api.github.com/repos/{repo}/actions/secrets/public-key",
                            headers=headers, timeout=15)
    if not key_resp.ok:
        print(f"ERROR: Failed to get repo public key ({key_resp.status_code})")
        return False

    key_data = key_resp.json()
    key_id = key_data["key_id"]
    public_key = key_data["key"]

    # Encrypt secret using libsodium sealed box
    try:
        from base64 import b64encode
        from nacl import encoding, public as nacl_public
        pub = nacl_public.PublicKey(public_key.encode("utf-8"), encoding.Base64Encoder())
        sealed = nacl_public.SealedBox(pub).encrypt(secret_value.encode("utf-8"))
        encrypted = b64encode(sealed).decode("utf-8")
    except ImportError:
        print("SKIP: Install PyNaCl to auto-update secrets: pip install pynacl")
        return False

    put_resp = requests.put(
        f"https://api.github.com/repos/{repo}/actions/secrets/{secret_name}",
        headers=headers,
        json={"encrypted_value": encrypted, "key_id": key_id},
        timeout=15,
    )
    if put_resp.ok:
        print(f"SUCCESS: Updated {secret_name} in {repo}")
        return True
    else:
        print(f"ERROR: Failed to update secret ({put_resp.status_code}): {put_resp.text}")
        return False


def main():
    import argparse
    parser = argparse.ArgumentParser(description="Generate YouTube OAuth2 refresh token")
    parser.add_argument("--update-secret", metavar="OWNER/REPO",
                        help="Auto-update YT_REFRESH_TOKEN in the given GitHub repo")
    args = parser.parse_args()

    print("=" * 60)
    print("  YouTube OAuth2 Refresh Token Generator")
    print("=" * 60)

    client_id, client_secret = _get_credentials()

    # Build authorization URL
    auth_params = urllib.parse.urlencode({
        "client_id": client_id,
        "redirect_uri": REDIRECT_URI,
        "response_type": "code",
        "scope": SCOPES,
        "access_type": "offline",
        "prompt": "consent",  # Force consent to always get refresh_token
    })
    auth_url = f"{AUTH_URL}?{auth_params}"

    print(f"\nOpening browser for Google sign-in...")
    print(f"If browser doesn't open, visit:\n  {auth_url}\n")
    webbrowser.open(auth_url)

    print("Waiting for authorization...")
    auth_code = _capture_auth_code()
    print("Authorization code received!")

    print("Exchanging for tokens...")
    tokens = _exchange_code(client_id, client_secret, auth_code)

    refresh_token = tokens["refresh_token"]
    print("\n" + "=" * 60)
    print("  REFRESH TOKEN (copy this to GitHub Secrets as YT_REFRESH_TOKEN):")
    print("=" * 60)
    print(refresh_token)
    print("=" * 60)

    if args.update_secret:
        print(f"\nUpdating YT_REFRESH_TOKEN in {args.update_secret}...")
        _update_github_secret(args.update_secret, "YT_REFRESH_TOKEN", refresh_token)

    print("\nDone! Token is valid indefinitely (if OAuth app is in Production mode).")
    print("If your app is in Testing mode, tokens expire in 7 days.")
    print("Check: Google Cloud Console → OAuth consent screen → Publishing status")


if __name__ == "__main__":
    main()
