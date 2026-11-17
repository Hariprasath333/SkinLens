/* SkinLens service worker.
 * Caches ONLY the app shell (CSS, JS, icons, manifest, offline page) so the app opens quickly
 * and shows a friendly page when there is no connection.
 * Never cached: pages (HTML), API responses, patient data, uploaded images, heat-maps, PDFs.
 * Those requests are not touched by this worker at all, except that a page which fails to
 * load because the phone is offline is replaced by the offline page. */
const VERSION = "skinlens-shell-v2";   // v2: new logo and icons (raise this whenever icons change)
const SHELL = [
  "/offline",
  "/static/css/style.css",
  "/static/js/app.js",
  "/static/img/logo.svg",
  "/static/img/app-icon.svg",
  "/static/icons/icon-192.png",
  "/static/icons/icon-512.png",
  "/static/icons/icon-maskable-512.png",
  "/static/icons/apple-touch-icon.png",
  "/static/icons/favicon-16.png",
  "/static/icons/favicon-32.png",
  "/static/icons/favicon-48.png",
  "/static/icons/favicon.ico",
  "/manifest.webmanifest",
];

self.addEventListener("install", (event) => {
  event.waitUntil(caches.open(VERSION).then((c) => c.addAll(SHELL)).then(() => self.skipWaiting()));
});

self.addEventListener("activate", (event) => {
  event.waitUntil(
    caches.keys()
      .then((keys) => Promise.all(keys.filter((k) => k !== VERSION).map((k) => caches.delete(k))))
      .then(() => self.clients.claim())
  );
});

self.addEventListener("fetch", (event) => {
  const req = event.request;
  const url = new URL(req.url);
  if (req.method !== "GET" || url.origin !== self.location.origin) return;

  // Pages: always from the network (they contain patient data), never stored.
  if (req.mode === "navigate") {
    event.respondWith(fetch(req).catch(() => caches.match("/offline")));
    return;
  }
  // App shell: network first (so updates arrive at once), cached copy when offline.
  if (SHELL.includes(url.pathname)) {
    event.respondWith(
      fetch(req)
        .then((res) => {
          if (res.ok) {
            const copy = res.clone();
            caches.open(VERSION).then((c) => c.put(url.pathname, copy));
          }
          return res;
        })
        .catch(() => caches.match(url.pathname))
    );
  }
  // Everything else (images, heat-maps, PDFs, JSON): not intercepted -> normal network request.
});
