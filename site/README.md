# biopb.org — static landing page

A plain static site (no build step) introducing the biopb project to users.
Served directly by nginx.

## Files
- `index.html` — the entire page
- `styles.css` — all styling (single accent color in the `--accent` CSS variable)
- `assets/logo.png` — nav/footer logo (extracted from `biopb-mcp/logo.icns`)
- `assets/favicon.ico` — site favicon

## Local preview
```bash
cd site && python3 -m http.server 8080   # → http://localhost:8080
```

## Deploy (nginx)
Copy the contents of `site/` to the biopb.org document root, e.g.:
```nginx
server {
    server_name biopb.org;
    root /var/www/biopb.org;      # contains index.html, styles.css, assets/
    index index.html;
    location / { try_files $uri $uri/ =404; }
}
```
`install.sh` (the `curl … | bash` target) is served by the existing GitHub
redirect and is independent of these files.

## Updating screenshots
The component rows use placeholder panels (`<div class="shot__ph">`). To use a
real screenshot, replace that `<div>` with `<img src="assets/your-shot.png"
alt="…">` — the `.shot img` rule already styles it to the right size.
