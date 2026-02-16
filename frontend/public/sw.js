const STATIC_CACHE = "signflow-static-v4";
const DICTIONARY_CACHE = "signflow-dictionary-v4";
const STATIC_ASSETS = ["/", "/index.html", "/manifest.json", "/icons/icon-192.svg", "/icons/icon-512.svg"];

self.addEventListener("install", (event) => {
  event.waitUntil(
    caches.open(STATIC_CACHE).then((cache) => cache.addAll(STATIC_ASSETS)).then(() => self.skipWaiting())
  );
});

self.addEventListener("activate", (event) => {
  event.waitUntil(
    caches.keys().then((keys) =>
      Promise.all(
        keys
          .filter((key) => key !== STATIC_CACHE && key !== DICTIONARY_CACHE)
          .map((key) => caches.delete(key))
      )
    )
  );
  self.clients.claim();
});

function isDictionaryRequest(url) {
  return (
    url.pathname.includes("/api/v1/dictionary") ||
    url.pathname.includes("/api/v1/signs") ||
    url.pathname.includes("/api/v1/stats/signs-per-category")
  );
}

function isApiRequest(url) {
  return url.pathname.startsWith("/api/");
}

async function staleWhileRevalidate(request) {
  const cache = await caches.open(DICTIONARY_CACHE);
  const cached = await cache.match(request);

  const networkPromise = fetch(request)
    .then((response) => {
      if (response && response.ok) {
        cache.put(request, response.clone());
      }
      return response;
    })
    .catch(() => cached);

  return cached || networkPromise;
}

async function cacheFirst(request) {
  const cached = await caches.match(request);
  if (cached) {
    return cached;
  }
  const response = await fetch(request);
  if (response && response.ok) {
    const cache = await caches.open(STATIC_CACHE);
    cache.put(request, response.clone());
  }
  return response;
}

self.addEventListener("fetch", (event) => {
  const request = event.request;
  if (request.method !== "GET") return;

  const url = new URL(request.url);
  if (url.origin !== self.location.origin) return;

  if (isDictionaryRequest(url)) {
    event.respondWith(staleWhileRevalidate(request));
    return;
  }

  if (isApiRequest(url)) {
    event.respondWith(fetch(request));
    return;
  }

  event.respondWith(cacheFirst(request));
});
