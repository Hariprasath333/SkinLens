// ---------------------------------------------------------------- menu (top hamburger + bottom "More")
document.addEventListener("click", (e) => {
  const btn = e.target.closest(".menu-btn, .menu-btn-bottom");
  if (btn) { document.querySelector(".nav")?.classList.toggle("open"); return; }
  const nav = document.querySelector(".nav.open");
  if (nav && !e.target.closest(".nav")) nav.classList.remove("open");
});

// ---------------------------------------------------------------- PWA: service worker + install button
if ("serviceWorker" in navigator && window.isSecureContext) {
  window.addEventListener("load", () => navigator.serviceWorker.register("/sw.js", { scope: "/" }).catch(() => {}));
}
(() => {
  let deferred = null;
  const buttons = () => document.querySelectorAll(".install-btn");
  window.addEventListener("beforeinstallprompt", (e) => {
    e.preventDefault();               // show our own button instead of the browser mini-bar
    deferred = e;
    buttons().forEach((b) => (b.hidden = false));
  });
  document.addEventListener("click", async (e) => {
    if (!e.target.closest(".install-btn") || !deferred) return;
    deferred.prompt();
    await deferred.userChoice.catch(() => null);
    deferred = null;
    buttons().forEach((b) => (b.hidden = true));
  });
  window.addEventListener("appinstalled", () => buttons().forEach((b) => (b.hidden = true)));
  const ios = /iphone|ipad|ipod/i.test(navigator.userAgent);
  const standalone = window.matchMedia("(display-mode: standalone)").matches || navigator.standalone;
  if (ios && !standalone) document.querySelectorAll(".ios-hint").forEach((el) => (el.hidden = false));
})();

// ---------------------------------------------------------------- photo picker: preview + resize to 1024 px
async function resizeImage(file, maxSide) {
  let src;
  try {
    src = await createImageBitmap(file, { imageOrientation: "from-image" });   // respects phone rotation
  } catch (e) {
    src = await new Promise((ok, bad) => {
      const img = new Image();
      img.onload = () => ok(img);
      img.onerror = bad;
      img.src = URL.createObjectURL(file);
    });
  }
  const w = src.width, h = src.height;
  const s = Math.min(1, maxSide / Math.max(w, h));
  if (s === 1 && /jpe?g/i.test(file.type)) return { blob: file, w, h, resized: false, src };
  const canvas = document.createElement("canvas");
  canvas.width = Math.round(w * s);
  canvas.height = Math.round(h * s);
  canvas.getContext("2d").drawImage(src, 0, 0, canvas.width, canvas.height);
  const blob = await new Promise((ok) => canvas.toBlob(ok, "image/jpeg", 0.9));
  return { blob, w: canvas.width, h: canvas.height, resized: true, ow: w, oh: h, src };
}

// ---------------------------------------------------------------- photo quality check (same rules as webapp/quality.py)
const QUALITY = { side: 512, minSide: 300, blurMin: 10, darkMean: 60, brightMean: 225, brightClip: 0.25,
                  centreMax: 0.25, skinMin: 0.35, stripesMin: 0.60, saturationMin: 140 };
const QUALITY_TEXT = {
  small: ["The photo is too small", "Use the camera's normal setting or move a little closer (at least 300 px)."],
  blurry: ["The photo looks blurry", "Hold the phone still and tap the spot on the screen to focus before taking the photo."],
  dark: ["The photo is too dark", "Move to daylight or a bright room. Avoid shadows over the spot."],
  bright: ["The photo is too bright", "Avoid the flash and direct sunlight; a little indirect light works best."],
  no_spot: ["No clear spot was found", "Make sure the spot is visible and fills a good part of the picture."],
  off_centre: ["The spot is not in the centre", "Place the spot in the middle of the picture."],
  not_skin: ["This does not look like skin", "Take a close photo of the skin spot itself (about 10 cm away)."],
};

function pixels(src, w, h) {
  const c = document.createElement("canvas");
  c.width = w; c.height = h;
  const ctx = c.getContext("2d", { willReadFrequently: true });
  ctx.drawImage(src, 0, 0, w, h);
  return ctx.getImageData(0, 0, w, h).data;
}

function otsu(gray) {
  const hist = new Array(256).fill(0);
  gray.forEach((v) => hist[v]++);
  const total = gray.length;
  let sum = 0; for (let i = 0; i < 256; i++) sum += i * hist[i];
  let sumB = 0, wB = 0, best = 0, thr = 0;
  for (let t = 0; t < 256; t++) {
    wB += hist[t]; if (!wB) continue;
    const wF = total - wB; if (!wF) break;
    sumB += t * hist[t];
    const mB = sumB / wB, mF = (sum - sumB) / wF, between = wB * wF * (mB - mF) ** 2;
    if (between > best) { best = between; thr = t; }
  }
  return thr;
}

function qualityCheck(src, ow, oh) {
  const s = QUALITY.side / Math.max(ow, oh);
  const w = Math.max(1, Math.round(ow * s)), h = Math.max(1, Math.round(oh * s));
  const d = pixels(src, w, h);
  const n = w * h, gray = new Uint8Array(n);
  let sum = 0, clipped = 0, skin = 0, sat = 0;
  for (let i = 0; i < n; i++) {
    const r = d[4 * i], g = d[4 * i + 1], b = d[4 * i + 2];
    const mx = Math.max(r, g, b), mn = Math.min(r, g, b);
    sat += mx ? 255 * (mx - mn) / mx : 0;           // HSV saturation, as OpenCV (0-255)
    const y = 0.299 * r + 0.587 * g + 0.114 * b;
    gray[i] = Math.round(y); sum += y; if (y >= 250) clipped++;
    const cr = (r - y) * 0.713 + 128, cb = (b - y) * 0.564 + 128;
    if (cr >= 133 && cr <= 180 && cb >= 70 && cb <= 135) skin++;
  }
  // variance of the Laplacian (4-neighbour kernel, as OpenCV's default)
  let ls = 0, ls2 = 0, ln = 0;
  for (let yy = 1; yy < h - 1; yy++) for (let xx = 1; xx < w - 1; xx++) {
    const i = yy * w + xx;
    const L = gray[i - 1] + gray[i + 1] + gray[i - w] + gray[i + w] - 4 * gray[i];
    ls += L; ls2 += L * L; ln++;
  }
  // stripes in one direction (Sobel structure tensor coherence over the whole picture), e.g. wood grain
  let jxx = 0, jyy = 0, jxy = 0;
  for (let yy = 1; yy < h - 1; yy++) for (let xx = 1; xx < w - 1; xx++) {
    const i = yy * w + xx;
    const gx = (gray[i - w + 1] + 2 * gray[i + 1] + gray[i + w + 1]) - (gray[i - w - 1] + 2 * gray[i - 1] + gray[i + w - 1]);
    const gy = (gray[i + w - 1] + 2 * gray[i + w] + gray[i + w + 1]) - (gray[i - w - 1] + 2 * gray[i - w] + gray[i - w + 1]);
    jxx += gx * gx; jyy += gy * gy; jxy += gx * gy;
  }
  const m = { width: ow, height: oh, brightness: sum / n, clipped: clipped / n, skin: skin / n,
              sharpness: ln ? ls2 / ln - (ls / ln) ** 2 : 0, saturation: sat / n,
              stripes: Math.sqrt((jxx - jyy) ** 2 + 4 * jxy ** 2) / (jxx + jyy + 1e-6) };
  // spot: darker-than-skin pixels (Otsu) on a 128 x 128 copy, centre of mass
  const q = pixels(src, 128, 128), g2 = new Uint8Array(128 * 128);
  for (let i = 0; i < g2.length; i++) g2[i] = Math.round(0.299 * q[4 * i] + 0.587 * q[4 * i + 1] + 0.114 * q[4 * i + 2]);
  const thr = otsu(g2);
  let cnt = 0, sx = 0, sy = 0;
  for (let i = 0; i < g2.length; i++) if (g2[i] <= thr) { cnt++; sx += i % 128; sy += Math.floor(i / 128); }
  const share = cnt / g2.length;
  m.spot_found = share > 0.02 && share < 0.9;
  if (m.spot_found) m.spot_offset = Math.hypot(sx / cnt / 128 - 0.5, sy / cnt / 128 - 0.5);
  const codes = [];
  if (Math.min(ow, oh) < QUALITY.minSide) codes.push("small");
  if (m.brightness < QUALITY.darkMean) codes.push("dark");
  else if (m.brightness > QUALITY.brightMean || m.clipped > QUALITY.brightClip) codes.push("bright");
  const skinLikeSurface = m.stripes >= QUALITY.stripesMin && m.saturation >= QUALITY.saturationMin;
  const notSkin = (m.skin < QUALITY.skinMin || skinLikeSurface) && !codes.includes("dark") && !codes.includes("bright");
  if (m.sharpness < QUALITY.blurMin && !codes.includes("dark") && !notSkin) codes.push("blurry");
  if (notSkin) codes.push("not_skin");
  else if (!m.spot_found) codes.push("no_spot");
  else if (m.spot_offset > QUALITY.centreMax) codes.push("off_centre");
  return { ok: !codes.length, metrics: m,
           issues: codes.map((c) => ({ code: c, title: QUALITY_TEXT[c][0], fix: QUALITY_TEXT[c][1] })) };
}
window.SkinLensQuality = qualityCheck;

document.querySelectorAll(".photo-picker").forEach((picker) => {
  const inputs = [...picker.querySelectorAll(".pick-input")];
  const box = picker.querySelector(".photo-preview");
  const img = box.querySelector("img");
  const info = box.querySelector(".preview-info");
  const maxSide = parseInt(picker.dataset.maxSide || "1024", 10);
  const form = picker.closest("form");
  const qbox = picker.querySelector(".quality-box");
  let job = null, chosen = null;

  const showQuality = (q) => {
    if (!qbox) return;
    picker.dataset.quality = JSON.stringify(q);
    if (q.ok) { qbox.hidden = true; return; }
    qbox.querySelector(".quality-list").innerHTML = q.issues
      .map((i) => `<li><b>${i.title}.</b> ${i.fix}</li>`).join("");
    qbox.hidden = false;
  };
  qbox?.querySelector(".quality-retake")?.addEventListener("click", () => {
    const again = chosen;
    inputs.forEach((o) => (o.value = ""));
    box.hidden = true; qbox.hidden = true; chosen = null; job = null;
    again?.click();                                 // opens the camera (or gallery) again
  });
  qbox?.querySelector(".quality-ignore")?.addEventListener("click", () => {
    qbox.hidden = true;
    info.textContent += " · photo quality warning ignored";
  });

  inputs.forEach((input) => input.addEventListener("change", () => {
    const file = input.files && input.files[0];
    if (!file) return;
    inputs.forEach((o) => { if (o !== input) o.value = ""; });   // only one photo is sent
    chosen = input;
    img.src = URL.createObjectURL(file);
    box.hidden = false;
    info.textContent = "Preparing photo…";
    job = resizeImage(file, maxSide).then((r) => {
      const kb = Math.round(r.blob.size / 1024);
      info.textContent = r.resized
        ? `Ready: made smaller from ${r.ow}×${r.oh} to ${r.w}×${r.h} px (${kb} KB)`
        : `Ready: ${r.w}×${r.h} px (${kb} KB)`;
      try { showQuality(qualityCheck(r.src, r.ow || r.w, r.oh || r.h)); } catch (err) { /* check is optional */ }
      return r;
    }).catch(() => { info.textContent = "Ready (sent as it is)"; return null; });
    box.scrollIntoView({ behavior: "smooth", block: "nearest" });
  }));

  form.addEventListener("submit", async (e) => {
    if (form.dataset.sending) return;           // second pass: our own form.submit()
    e.preventDefault();
    if (!chosen || !chosen.files.length) {
      info.textContent = "";
      alert("Please take a photo or choose one from the gallery first.");
      return;
    }
    showBusy(form);
    const r = await job;
    if (r && r.resized && typeof DataTransfer !== "undefined") {
      try {
        const dt = new DataTransfer();
        dt.items.add(new File([r.blob], "photo.jpg", { type: "image/jpeg" }));
        chosen.files = dt.files;
      } catch (err) { /* browser cannot replace the file: send the original */ }
    }
    form.dataset.sending = "1";
    form.submit();
  });
});

function showBusy(form) {
  document.querySelector(".busy-overlay")?.removeAttribute("hidden");
  const b = form.querySelector("button[type=submit]");
  if (b) b.disabled = true;
}
// coming back with the browser's Back button: hide the overlay again
window.addEventListener("pageshow", () => {
  document.querySelector(".busy-overlay")?.setAttribute("hidden", "");
  document.querySelectorAll("form[data-sending]").forEach((f) => {
    delete f.dataset.sending;
    const b = f.querySelector("button[type=submit]");
    if (b) b.disabled = false;
  });
});

// ---------------------------------------------------------------- swipeable result tabs
document.querySelectorAll(".tabs[data-swipe]").forEach((tabs) => {
  const swipe = document.querySelector(tabs.dataset.swipe);
  if (!swipe) return;
  const btns = [...tabs.querySelectorAll(".tab")];
  const setActive = (i) => btns.forEach((b, k) => {
    b.classList.toggle("active", k === i);
    b.setAttribute("aria-selected", k === i ? "true" : "false");
  });
  btns.forEach((b, i) => b.addEventListener("click", () => {
    swipe.scrollTo({ left: i * swipe.clientWidth, behavior: "smooth" });
    setActive(i);
  }));
  let t = null;
  swipe.addEventListener("scroll", () => {
    clearTimeout(t);
    t = setTimeout(() => setActive(Math.round(swipe.scrollLeft / Math.max(1, swipe.clientWidth))), 60);
  });
});

// ---------------------------------------------------------------- patient summary PDF: data URI -> file download
document.querySelectorAll('a[download][href^="data:application/pdf"]').forEach((a) => {
  try {
    const b64 = a.getAttribute("href").split(",")[1];
    const bytes = Uint8Array.from(atob(b64), (c) => c.charCodeAt(0));
    a.href = URL.createObjectURL(new Blob([bytes], { type: "application/pdf" }));
  } catch (e) { /* keep the data URI */ }
});

// ---------------------------------------------------------------- result page: wait for LIME (computed in the background)
(() => {
  const box = document.querySelector("[data-lime-url]");
  if (!box) return;
  const poll = async () => {
    try {
      const r = await fetch(box.dataset.limeUrl, { credentials: "same-origin", cache: "no-store" });
      const d = await r.json();
      if (d.status === "done") {
        box.outerHTML = '<img src="' + d.url + '?t=' + Date.now() + '" alt="LIME explanation">';
        const s = document.querySelector("#lime-summary");
        if (s) { s.textContent = d.summary; s.hidden = false; }
        return;
      }
      if (d.status !== "pending") { box.innerHTML = "LIME explanation could not be computed for this image."; return; }
    } catch (e) { /* retry */ }
    setTimeout(poll, 3000);
  };
  poll();
})();

// ---------------------------------------------------------------- feedback form: "correct diagnosis" only when disagreeing
(() => {
  const form = document.querySelector(".fb-form");
  if (!form) return;
  const wrap = document.getElementById("correct-wrap");
  const select = document.getElementById("correct_class");
  const update = () => {
    const v = (form.querySelector("input[name=agreement]:checked") || {}).value;
    wrap.hidden = v !== "disagree";
    select.required = v === "disagree";
  };
  form.addEventListener("change", update);
  update();
})();

// ---------------------------------------------------------------- patient result: LIME picture (computed on request, nothing stored)
(() => {
  const box = document.getElementById("p-lime");
  if (!box) return;
  const photo = document.querySelector(box.dataset.photoSrc);
  fetch("/check/lime", {
    method: "POST", headers: { "Content-Type": "application/json" }, cache: "no-store",
    body: JSON.stringify({ photo: photo ? photo.getAttribute("src") : "", token: box.dataset.token }),
  }).then((r) => r.ok ? r.json() : Promise.reject(r.status))
    .then((d) => { box.outerHTML = '<img src="' + d.image + '" alt="Highlighted areas (LIME)">'; })
    .catch(() => { box.innerHTML = "<div>The second picture could not be made. The heat-map on the left still shows where the app looked.</div>"; });
})();

// ---------------------------------------------------------------- "Show full details" / "Hide full details"
document.querySelectorAll("details.full-details").forEach((d) => {
  const s = d.querySelector("summary");
  d.addEventListener("toggle", () => { s.textContent = d.open ? "Hide full details" : "Show full details"; });
});
