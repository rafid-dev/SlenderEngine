# Sharing the web app over Tailscale

Use the chat page from your phone, a laptop or another PC, wherever they are, without opening a port on your router
or putting the model on the internet. [Tailscale](https://tailscale.com) connects your own devices in a private,
encrypted network (a "tailnet"). Only devices signed in to your tailnet can reach the page.

The setup below keeps the server listening on `127.0.0.1` only. Tailscale answers on an HTTPS address inside your
tailnet and passes the requests on to it. Nothing changes on your local network, your firewall or your router.

## One-time setup

1. Install Tailscale on the PC that runs the model and on every device you want to use: https://tailscale.com/download.
   Sign in to the same account on all of them.
2. In the [Tailscale admin console](https://login.tailscale.com/admin/dns), under **DNS**: turn on **MagicDNS**, and
   under **HTTPS Certificates** click **Enable**. This gives the PC a name like `your-pc.your-tailnet.ts.net` with a
   real certificate.

## Share

Start the server as you always do. Then, in the repository folder:

```bat
TAILSCALE-SHARE.bat on --config path\to\your-run-config.json
```

On Linux or a Mac: `python3 tools/tailscale_share.py on --config path/to/your-run-config.json`.

`--config` is the run config the server starts with (the `--config` file of `serve/server.py`, or the
`strata-<model>.json` that `strata launch` uses). The tool shows what it will change and asks first:

- In the run config: a new random `"api_key"` (unless there is one), your PC's tailnet name in `"allowed_hosts"`, and
  the shared address in `"trusted_origins"`. A copy of the old config is kept as `<name>.before-tailscale.json`.
- `tailscale serve --bg --https=8443 http://127.0.0.1:8080`: the page at `https://your-pc.your-tailnet.ts.net:8443/`.

Then:

1. **Restart the server.** It reads the key and the host name only when it starts.
2. On your phone, with Tailscale connected, open the address the tool printed.
3. The page asks for the API key: paste it in **Settings > Connection**. The tool printed it once; it is also in the
   run config.
4. Optional: add the page to your home screen.

The default HTTPS port is 8443, so the tool never takes over a service you already share on 443. `--https-port 443`
uses the plain address when 443 is free. `--port` is the server's port if it is not 8080. `--dry-run` shows the plan
and changes nothing.

## Check and stop

```bat
TAILSCALE-SHARE.bat
```

With no command it shows what is set up: Tailscale, MagicDNS, HTTPS certificates, what this PC shares, whether the
server runs and has a key. It changes nothing.

```bat
TAILSCALE-SHARE.bat off
```

stops sharing. Add `--forget` to also take the tailnet name out of the run config (the API key stays). The tool only
turns off the port it set up; another `tailscale serve` mapping on this PC is left alone.

## Sharing with someone else

- **One machine for one person:** in the admin console, **Machines > your PC > Share**. The other person gets this PC
  in their own tailnet and nothing else of yours. Send them the address and the API key.
- **Someone in your tailnet:** invite them, then limit what they reach with an access rule. For example, give the PC
  the tag `tag:ai-pc` and allow only port 8443:

  ```json
  "acls": [
    {"action": "accept", "src": ["friend@example.com"], "dst": ["tag:ai-pc:8443"]}
  ]
  ```

To take access away: remove the share or the rule, and change the key. Put a new `"api_key"` in the run config and
restart the server; every device then needs the new key.

Anyone with access and the key can use your GPU, the model, and the web app's tools: web search, reading web pages,
and the documents attached on this server. See [WEB_APP.md](WEB_APP.md#security).

## Not recommended: the public internet

`tailscale funnel` puts a `tailscale serve` address on the public internet. The tool never does this. If you do it
yourself, keep the API key, and know that anyone who learns the key can run the model and its tools on your PC.

## The other way: `strata launch --access tailscale`

`strata launch --access tailscale` makes the server listen on your PC's tailnet address itself, as
`http://your-pc.your-tailnet.ts.net:8080` ([TERMINAL_LAUNCH.md](TERMINAL_LAUNCH.md)). The traffic is still encrypted
by Tailscale, but the page is plain HTTP, Windows asks to allow Python on the network, and the server listens beyond
`127.0.0.1`. The `tailscale serve` setup above avoids all three.

## When something does not work

| You see | Why | What to do |
|---|---|---|
| `403` and a message about the host name | The server was not restarted after the config change, or the name is missing from `"allowed_hosts"` | Restart the server; run `TAILSCALE-SHARE.bat` to check the config |
| `401` or "This server needs its API key" | The device has no key or an old one | Settings > Connection > API key |
| `tailscale serve failed` and a link | HTTPS certificates are off for the tailnet | Admin console > DNS > HTTPS Certificates > Enable |
| A certificate warning on the first visit | The certificate is made on the first request | Wait a few seconds and reload |
| The page does not load on the phone | Tailscale is not connected on the phone | Open the Tailscale app and connect |
| `HTTPS port 443 ... shares something else` | Another service uses that port | Keep the default 8443, or pick another `--https-port` |
| "access denied" on Linux | Your user may not change Tailscale's settings | `sudo tailscale set --operator=$USER`, once |
| No chats on the phone | Chats are kept per browser and per address | Settings > Data > Export on one, Import on the other |
