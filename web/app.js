'use strict';

const $ = selector => document.querySelector(selector);
const input = $('#file-input');
const metadata = $('#metadata-input');
const demoButtons = document.querySelectorAll('[data-demo]');
const preview = $('#preview');
const routePage = Boolean($('#map'));
const sharedRun = new URLSearchParams(location.search).get('run');
const typeNames = { odn: 'ОДН', manhole: 'Кабельный колодец' };
const classNames = { pole: 'Опора', antenna: 'Антенна', rru: 'RRU', microwave: 'РРЛ', manhole: 'Люк' };
const classColors = { pole: '#7ee4bf', antenna: '#7eaaff', rru: '#daabff', microwave: '#ffce7b', manhole: '#ff5e85' };
const statusNames = { repair: 'Ремонт', review: 'Проверка', unassessed: 'Не оценивалось', normal: 'Норма' };
const recommendations = {
  repair: 'Передать в ремонт после подтверждения',
  review: 'Проверить по кадру или на месте',
  unassessed: 'Оценить состояние вручную',
  normal: 'Действий не требуется',
};
const coordinateSources = { csv: 'CSV', exif: 'EXIF кадра', panorama: 'Панорама', video_gps: 'Позиция камеры · GPS' };
const isVideo = file => /\.(mp4|mov|webm)$/i.test(file.name);

let current;
let demoRoutes = [];
let routeModel = '';
let panoramaArchive;
let archiveFilter = '';
let archiveSelection;
let map;
let layers;
let tiles;
let roadLabels;
let activeSite;
let activeRoute;
const demoMarkers = new Map();
let routeFilter = '';
let demoKindFilter = '';
let mapMode = 'demo';
let busy = false;
let modelsReady = true;
let previewFrame;
let galleryObserver;
let detectionsByFrame = new Map();
let observationsByFrame = new Map();
let framesByName = new Map();

function element(tag, text, css) {
  const node = document.createElement(tag);
  if (text !== undefined) node.textContent = text;
  if (css) node.className = css;
  return node;
}

async function jsonRequest(url, options) {
  const response = await fetch(url, options);
  const data = await response.json();
  if (!response.ok) {
    throw new Error(typeof data.detail === 'string' ? data.detail : 'Проверьте входные данные');
  }
  return data;
}

function updateBusy(value = busy) {
  busy = value;
  $('#analyze-button').disabled = busy || !modelsReady || !input.files.length;
  demoButtons.forEach(button => { button.disabled = busy || !modelsReady; });
  $('#progress').hidden = !busy;
  $('#upload-form').setAttribute('aria-busy', String(busy));
}

function showError(message) {
  $('#error').textContent = message;
  $('#error').hidden = !message;
}

function updateFiles() {
  const list = $('#file-list');
  list.replaceChildren(...Array.from(input.files).slice(0, 6).map(file => element('li', file.name)));
  if (input.files.length > 6) list.append(element('li', 'и ещё ' + (input.files.length - 6)));
  list.hidden = !input.files.length;
  const video = Array.from(input.files).some(isVideo);
  $('#metadata-label').textContent = video ? 'GPS-трек CSV' : 'Метаданные CSV';
  $('#metadata-template').href = video ? '/static/video-gps-template.csv' : '/static/metadata-template.csv';
  updateBusy();
}

function detectionName(item) {
  const name = classNames[item.class] || item.class;
  return item.track_id == null ? name : name + ' #' + item.track_id;
}

async function runAnalysis(url, body) {
  if (busy) return;
  showError('');
  updateBusy(true);
  try {
    const tile = $('#tile-input').checked;
    current = await jsonRequest(url + '?tile=' + tile, { method: 'POST', body });
    renderResult();
    history.replaceState(null, '', '/?run=' + encodeURIComponent(current.run_id) + '#registry');
    $('#registry').scrollIntoView({ block: 'start' });
  } catch (error) {
    showError(error.message || 'Не удалось выполнить анализ');
    $('#upload').scrollIntoView({ block: 'start' });
  } finally {
    updateBusy(false);
  }
}

function initUpload() {
  input.addEventListener('change', updateFiles);
  metadata.addEventListener('change', () => {
    const file = metadata.files[0];
    $('#metadata-name').textContent = file?.name || '';
    $('#clear-metadata').hidden = !file;
  });
  $('#clear-metadata').addEventListener('click', () => {
    metadata.value = '';
    metadata.dispatchEvent(new Event('change'));
  });

  const dropzone = $('#dropzone');
  for (const eventName of ['dragenter', 'dragover', 'dragleave', 'drop']) {
    dropzone.addEventListener(eventName, event => {
      event.preventDefault();
      dropzone.classList.toggle('dragover', eventName === 'dragenter' || eventName === 'dragover');
      if (eventName === 'drop') {
        input.files = event.dataTransfer.files;
        updateFiles();
      }
    });
  }

  $('#upload-form').addEventListener('submit', event => {
    event.preventDefault();
    if (!input.files.length || busy) return;
    if (Array.from(input.files).some(isVideo)) {
      if (input.files.length !== 1) {
        showError('Для одного анализа выберите один ролик или набор фотографий');
        return;
      }
      const body = new FormData();
      body.append('file', input.files[0]);
      if (metadata.files.length) body.append('gps', metadata.files[0]);
      runAnalysis('/api/analyze/video', body);
      return;
    }
    if (input.files.length > 100) {
      showError('За один анализ можно загрузить до 100 кадров');
      return;
    }
    const body = new FormData();
    Array.from(input.files).forEach(file => body.append('files', file));
    if (metadata.files.length) body.append('metadata', metadata.files[0]);
    runAnalysis('/api/analyze', body);
  });
  demoButtons.forEach(button => button.addEventListener('click', () => {
    $('#upload').scrollIntoView({ block: 'start' });
    runAnalysis('/api/demo');
  }));
  $('#overlay-select').addEventListener('change', updateOverlays);
  $('#confidence-filter').addEventListener('change', updateOverlays);
}

for (const link of document.querySelectorAll('.nav a[href^="#"]')) {
  link.addEventListener('click', () => {
    document.querySelector('.nav [aria-current]')?.removeAttribute('aria-current');
    link.setAttribute('aria-current', 'location');
  });
}

function updateHealth(data) {
  modelsReady = Boolean(data.model_available);
  $('#health').className = 'health ' + (modelsReady ? 'ready' : 'failed');
  $('#health span').textContent = modelsReady ? data.model + ' · ' + deviceLabel(data) : 'Детектор недоступен';
  const condition = $('#condition-state');
  if (!condition) return;
  if (data.condition_config_error) {
    condition.textContent = 'Ошибка настроек модели состояния';
    condition.title = data.condition_config_error;
  } else if (!data.condition_model_available) {
    condition.textContent = data.manhole_model_available ? 'Детекция готова · веса состояния ожидаются' : 'Готов только детектор ОДН';
  } else {
    const modules = [];
    if (data.condition_models.classifier) modules.push('классификация');
    if (data.condition_models.segmenter) modules.push('сегментация');
    condition.textContent = 'Подключено: ' + modules.join(' и ');
  }
  updateBusy();
}

function deviceLabel(data) {
  if (data.device === 'cpu') return 'CPU';
  if (data.backend === 'rocm') return 'GPU · ROCm';
  if (data.backend === 'mps') return 'GPU · MPS';
  return 'GPU';
}

function filterRows() {
  const kind = $('#kind-filter').value;
  const status = $('#status-filter').value;
  return (current?.registry || []).filter(row => (!kind || row.kind === kind) && (!status || row.status === status));
}

function evidenceButton(frame, label = 'Открыть кадр') {
  const action = element('button', label, 'plain');
  action.type = 'button';
  action.addEventListener('click', () => openFrame(framesByName.get(frame)));
  return action;
}

function renderRegistry() {
  const rows = filterRows();
  const body = $('#registry-body');
  const rendered = rows.map(row => {
    const line = element('tr');
    const object = element('td');
    const identity = row.object_id || row.observation_id;
    object.append(element('strong', typeNames[row.kind]), element('small', identity));
    if (row.source === 'metadata') object.append(element('small', 'Тип из CSV'));
    if (row.source === 'equipment_only') object.append(element('small', 'Кандидат по оборудованию'));
    if (row.source === 'equipment_cluster') object.append(element('small', 'Группа оборудования в видео'));

    const status = element('td');
    status.append(element('span', statusNames[row.status], 'pill ' + row.status));
    if (row.defect) status.append(element('small', row.defect));
    if (row.condition_confidence !== null) {
      status.append(element('small', 'Уверенность ' + Math.round(row.condition_confidence * 100) + '%'));
    }

    const geo = element('td');
    if (row.latitude !== null) {
      geo.append(element('span', row.latitude.toFixed(5) + ', ' + row.longitude.toFixed(5)));
      geo.append(element('small', coordinateSources[row.coordinate_source] || 'Координаты наблюдения'));
      if (row.coordinate_source === 'video_gps' && row.coordinate_frame !== row.frame) {
        const source = framesByName.get(row.coordinate_frame);
        if (source) geo.append(element('small', source.label));
      }
    } else {
      geo.append(element('small', 'Без координат'));
    }

    const frame = element('td');
    frame.append(evidenceButton(row.frame, 'Смотреть'));
    if (row.observations > 1) frame.append(element('small', 'Наблюдений: ' + row.observations));
    line.append(object, status, geo, element('td', recommendations[row.status], 'table-action'), frame);
    return line;
  });
  if (!rendered.length) {
    const line = element('tr');
    const blank = element('td', 'По выбранным фильтрам объектов нет');
    blank.colSpan = 5;
    line.append(blank);
    rendered.push(line);
  }
  body.replaceChildren(...rendered);
}

function byFrame(items) {
  const grouped = new Map();
  for (const item of items) {
    if (!grouped.has(item.frame)) grouped.set(item.frame, []);
    grouped.get(item.frame).push(item);
  }
  return grouped;
}

function drawFrame(canvas, frame, maxWidth) {
  const image = new Image();
  image.addEventListener('load', () => {
    const scale = Math.min(1, maxWidth / image.naturalWidth);
    canvas.width = Math.round(image.naturalWidth * scale);
    canvas.height = Math.round(image.naturalHeight * scale);
    const ctx = canvas.getContext('2d');
    ctx.drawImage(image, 0, 0, canvas.width, canvas.height);
    const mode = $('#overlay-select')?.value || 'all';
    const threshold = Number($('#confidence-filter')?.value || 0) / 100;

    if (mode === 'all' || mode === 'masks') {
      for (const item of observationsByFrame.get(frame.name) || []) {
        for (const mask of item.masks) {
          if (mask.confidence < threshold) continue;
          ctx.beginPath();
          for (const [index, [x, y]] of mask.polygon.entries()) {
            if (index === 0) ctx.moveTo(x * scale, y * scale);
            else ctx.lineTo(x * scale, y * scale);
          }
          ctx.closePath();
          ctx.fillStyle = '#f61c6790';
          ctx.strokeStyle = '#ffd2df';
          ctx.lineWidth = 2;
          ctx.fill();
          ctx.stroke();
        }
      }
    }

    if (mode === 'all' || mode === 'boxes') {
      const font = Math.max(11, Math.round(canvas.width / 85));
      ctx.font = `600 ${font}px "Source Sans 3", sans-serif`;
      for (const detection of detectionsByFrame.get(frame.name) || []) {
        if (detection.confidence < threshold) continue;
        const [x1, y1, x2, y2] = detection.bbox.map(value => value * scale);
        const color = classColors[detection.class] || '#fff';
        ctx.lineWidth = Math.max(1, canvas.width / 470);
        ctx.strokeStyle = color;
        ctx.strokeRect(x1, y1, x2 - x1, y2 - y1);
        const title = detectionName(detection) + ' ' + Math.round(detection.confidence * 100) + '%';
        const badgeY = Math.max(0, y1 - font - 6);
        ctx.fillStyle = '#140b19dc';
        ctx.fillRect(x1, badgeY, ctx.measureText(title).width + 8, font + 5);
        ctx.fillStyle = color;
        ctx.fillText(title, x1 + 4, badgeY + font);
      }
    }
  });
  image.src = frame.original_url;
}

function openFrame(frame) {
  if (!frame) return;
  previewFrame = frame;
  $('#preview-caption').textContent = frame.label || frame.name;
  $('#preview-meta').textContent = frame.latitude !== null
    ? frame.latitude.toFixed(5) + ', ' + frame.longitude.toFixed(5) + ' · ' + (coordinateSources[frame.coordinate_source] || 'Координаты кадра')
    : 'Кадр без координат';
  const canvas = $('#preview-canvas');
  canvas.width = canvas.height = 1;
  canvas.setAttribute('aria-label', 'Кадр ' + (frame.label || frame.name));
  drawFrame(canvas, frame, 1920);
  if (!preview.open) preview.showModal();
}
preview.addEventListener('click', event => {
  if (event.target === preview) preview.close();
});

function renderGallery() {
  galleryObserver?.disconnect();
  galleryObserver = new IntersectionObserver(entries => {
    for (const entry of entries) {
      if (!entry.isIntersecting) continue;
      drawFrame(entry.target, framesByName.get(entry.target.dataset.frame), 670);
      galleryObserver.unobserve(entry.target);
    }
  }, { rootMargin: '160px' });

  const cards = current.frames.map(frame => {
    const card = element('button', undefined, 'frame-card');
    card.type = 'button';
    const canvas = element('canvas');
    canvas.dataset.frame = frame.name;
    canvas.setAttribute('aria-hidden', 'true');
    const count = detectionsByFrame.get(frame.name)?.length || 0;
    const info = element('div');
    info.append(element('strong', frame.label || frame.name), element('span', count + ' дет.'));
    card.append(canvas, info);
    card.addEventListener('click', () => openFrame(frame));
    galleryObserver.observe(canvas);
    return card;
  });
  $('#gallery').replaceChildren(...cards);
}

function pointsAtSameLocation(rows) {
  const groups = new Map();
  for (const row of rows) {
    if (row.latitude === null || row.longitude === null) continue;
    const point = [row.latitude, row.longitude];
    const key = point.join(',');
    if (!groups.has(key)) groups.set(key, { point, objects: [] });
    groups.get(key).objects.push(row);
  }
  return Array.from(groups.values());
}

function demoSites(route) {
  if (!demoKindFilter) return route.sites;
  if (demoKindFilter === 'false_positive') return route.sites.filter(site => site.review === 'false_positive');
  return route.sites.filter(site => site.review !== 'false_positive' && (site.kind || 'odn') === demoKindFilter);
}

function renderRouteButtons() {
  const list = $('#route-list');
  const buttons = [];
  if (current) {
    const showResult = element('button', 'Результат анализа');
    showResult.type = 'button';
    showResult.setAttribute('aria-pressed', String(mapMode === 'result'));
    showResult.addEventListener('click', () => { mapMode = 'result'; renderMap(); });
    buttons.push(showResult);
  }
  if (panoramaArchive) {
    const archiveButton = element('button', 'Архив БС');
    archiveButton.type = 'button';
    archiveButton.setAttribute('aria-pressed', String(mapMode === 'archive'));
    archiveButton.addEventListener('click', () => { mapMode = 'archive'; renderMap(); });
    buttons.push(archiveButton);
  }
  if (demoRoutes.length) {
    const showAll = element('button', 'Демо-подборка');
    showAll.type = 'button';
    showAll.setAttribute('aria-pressed', String(mapMode === 'demo' && !routeFilter));
    showAll.addEventListener('click', () => { mapMode = 'demo'; routeFilter = ''; renderMap(); });
    buttons.push(showAll);
  }
  for (const route of mapMode === 'demo' ? demoRoutes : []) {
    const pick = element('button');
    pick.type = 'button';
    pick.setAttribute('aria-pressed', String(mapMode === 'demo' && routeFilter === route.id));
    const dot = element('span', undefined, 'route-chip');
    dot.style.backgroundColor = route.color;
    const count = demoSites(route).length;
    pick.disabled = count === 0;
    pick.append(dot, document.createTextNode(route.name + ' · ' + count));
    pick.addEventListener('click', () => { mapMode = 'demo'; routeFilter = route.id; renderMap(); });
    buttons.push(pick);
  }
  list.replaceChildren(...buttons);
}

const roadAttribution = '&copy; <a href="https://www.openstreetmap.org/copyright">OpenStreetMap</a> · <a href="https://www.openstreetmap.org/fixthemap">Исправить карту</a>';

function initMap() {
  if (!window.L) return;
  map = L.map('map', {
    scrollWheelZoom: true, zoomAnimation: false, zoomControl: false, preferCanvas: true,
  }).setView([55.75, 37.62], 10);
  map.attributionControl.setPrefix(false);
  map.attributionControl.addAttribution(roadAttribution);
  L.control.zoom({ zoomInTitle: 'Приблизить', zoomOutTitle: 'Отдалить' }).addTo(map);
  const RussianScale = L.Control.Scale.extend({
    _updateMetric(maxMeters) {
      const meters = this._getRoundNum(maxMeters);
      const text = meters < 1000 ? `${meters} м` : `${meters / 1000} км`;
      this._updateScale(this._mScale, text, meters / maxMeters);
    },
  });
  new RussianScale({ imperial: false, position: 'bottomleft' }).addTo(map);
  layers = L.layerGroup().addTo(map);
  const eox = 'https://tiles.maps.eox.at/wmts/1.0.0/';
  tiles = L.tileLayer(eox + 's2cloudless_3857/default/g/{z}/{y}/{x}.jpg', {
    className: 'map-satellite', maxNativeZoom: 14, maxZoom: 19,
    attribution: '<a href="https://cloudless.eox.at">EOxCloudless</a> by EOX IT Services GmbH (Copernicus Sentinel 2016 &amp; 2017)',
  });
  roadLabels = L.layerGroup();
  map.on('moveend zoomend', renderRoadLabels);
  let satelliteReceived = false;
  tiles.on('loading', () => { satelliteReceived = false; });
  tiles.on('tileload', () => { satelliteReceived = true; $('#basemap-note').hidden = true; });
  tiles.on('load', () => { $('#basemap-note').hidden = satelliteReceived || !map.hasLayer(tiles); });
  $('#basemap-select').addEventListener('change', setBasemap);
  setBasemap();
}

function setBasemap() {
  const mode = $('#basemap-select').value;
  if (mode === 'scheme') {
    tiles.remove();
    $('#basemap-note').hidden = true;
  } else if (!map.hasLayer(tiles)) {
    tiles.addTo(map);
    tiles.bringToBack();
  }
  if (mode === 'satellite') roadLabels.remove();
  else {
    roadLabels.addTo(map);
    renderRoadLabels();
  }
}

function renderRoadLabels() {
  if (!map.hasLayer(roadLabels)) return;
  roadLabels.clearLayers();
  const overview = map.getZoom() < 12;
  const bounds = map.getBounds();
  const labels = overview ? [{ name: 'Москва', point: [55.7558, 37.6173] }] : [];
  const routes = mapMode === 'archive' ? [] :
    demoRoutes.filter(route => !routeFilter || route.id === routeFilter || mapMode === 'result');
  for (const route of routes) {
    for (const site of mapMode === 'result' ? route.sites : demoSites(route)) {
      if (site.road && /[А-Яа-яЁё]/.test(site.road)) labels.push({ name: site.road, point: [site.latitude, site.longitude] });
    }
  }
  const used = new Set();
  const positions = [];
  for (const { name, point } of labels) {
    if (positions.length >= (overview ? 6 : 18)) break;
    if (used.has(name) || !bounds.contains(point)) continue;
    const px = map.latLngToContainerPoint(point);
    if (positions.some(other => Math.abs(other.x - px.x) < 163 && Math.abs(other.y - px.y) < 28)) continue;
    used.add(name);
    positions.push(px);
    L.tooltip({ permanent: true, direction: 'top', offset: [0, -6], className: 'map-road-label' })
      .setLatLng(point).setContent(element('span', name)).addTo(roadLabels);
  }
}

function siteTitle(site) {
  if (site.review === 'false_positive') return 'Ошибка модели · ' + (site.road || site.site.replace('site_', ''));
  if (site.kind === 'manhole') return site.road ? 'Люк · ' + site.road : 'Люк · ' + site.site.replace('site_', '');
  if (site.kind === 'panorama') return 'Панорама БС ' + site.site.replace('site_', '');
  return site.road || 'ОДН ' + site.site.replace('site_', '');
}

function drawRouteFrame(canvas, site, full) {
  const key = site.frame + (full ? ':full' : ':detail');
  canvas.dataset.view = key;
  canvas.width = canvas.height = 1;
  const image = new Image();
  image.addEventListener('load', () => {
    if (canvas.dataset.view !== key) return;
    const [left, top, right, bottom] = full ? [0, 0, image.naturalWidth, image.naturalHeight] : site.detail_bbox;
    const width = right - left;
    const height = bottom - top;
    const scale = Math.min(1440 / width, 940 / height);
    canvas.width = Math.round(width * scale);
    canvas.height = Math.round(height * scale);
    const ctx = canvas.getContext('2d');
    ctx.drawImage(image, left, top, width, height, 0, 0, canvas.width, canvas.height);
    const font = Math.max(17, Math.round(canvas.width / 48));
    ctx.font = `700 ${font}px "Source Sans 3", sans-serif`;
    for (const item of site.detections) {
      if (!full && item.class === 'pole') continue;
      const [x1, y1, x2, y2] = item.bbox;
      const x = (x1 - left) * scale;
      const y = (y1 - top) * scale;
      const color = classColors[item.class];
      ctx.strokeStyle = '#140b19';
      ctx.lineWidth = Math.max(6, canvas.width / 170);
      ctx.strokeRect(x, y, (x2 - x1) * scale, (y2 - y1) * scale);
      ctx.strokeStyle = color;
      ctx.lineWidth = Math.max(3, canvas.width / 360);
      ctx.strokeRect(x, y, (x2 - x1) * scale, (y2 - y1) * scale);
      const label = detectionName(item) + ' ' + Math.round(item.confidence * 100) + '%';
      const labelWidth = ctx.measureText(label).width + 10;
      const labelX = Math.max(0, Math.min(full ? x : (x2 - left) * scale + 6, canvas.width - labelWidth));
      const labelY = Math.max(0, Math.min(full ? y - font - 7 : y, canvas.height - font - 6));
      ctx.fillStyle = '#140b19ef';
      ctx.fillRect(labelX, labelY, labelWidth, font + 6);
      ctx.fillStyle = color;
      ctx.fillText(label, labelX + 5, labelY + font);
    }
  });
  image.src = site.image_url;
  const detail = site.review === 'false_positive' ? 'Детекция крупно: ' :
    site.kind === 'manhole' ? 'Люк крупно: ' : 'Оборудование крупно: ';
  canvas.setAttribute('aria-label', (full ? 'Полный кадр: ' : detail) + siteTitle(site));
}

function showRouteSite(site, route, routeStops = demoSites(route)) {
  activeSite = site;
  activeRoute = route;
  for (const [name, marker] of demoMarkers) {
    const selected = name === site.frame;
    const outlined = marker.options.infrastructureKind === 'manhole' || marker.options.falsePositive;
    marker.setStyle({
      radius: selected ? 8 : outlined ? 6 : 5,
      weight: selected ? 3 : outlined ? 2 : 1.5,
      fillOpacity: outlined ? .28 : selected ? 1 : .85,
    });
    if (selected) marker.bringToFront();
  }
  $('#route-evidence').hidden = false;
  $('#route-view').hidden = false;
  $('#route-zoom').hidden = false;
  $('#route-evidence-title').textContent = siteTitle(site);
  const manhole = site.kind === 'manhole';
  const falsePositive = site.review === 'false_positive';
  $('#route-view').options[0].textContent = falsePositive ? 'Детекция крупно' : manhole ? 'Люк крупно' : 'Модули крупно';
  const distance = site.distance_to_station_m;
  $('#route-evidence-meta').textContent = [route.name, site.detector || routeModel,
    falsePositive ? site.review_note : manhole ? 'Принадлежность МГТС не подтверждена' :
      distance == null ? '' : distance.toLocaleString('ru-RU', { maximumFractionDigits: 2 }) + ' м до точки БС',
  ].filter(Boolean).join(' · ');
  const stops = routeStops.map((point, index) => {
    const title = siteTitle(point);
    const duplicate = routeStops.some(other => other !== point && siteTitle(other) === title);
    const button = element('button', duplicate ? (index + 1) + '. ' + title : title);
    button.type = 'button';
    button.setAttribute('aria-pressed', String(point.frame === site.frame));
    button.addEventListener('click', () => {
      showRouteSite(point, route, routeStops);
      map.panTo([point.latitude, point.longitude]);
    });
    return button;
  });
  $('#route-stops').replaceChildren(...stops);
  renderRouteDetections(site.detections.filter(detection => detection.class !== 'pole'));
  drawRouteFrame($('#route-canvas'), site, $('#route-view').value === 'frame');
}

function renderRouteDetections(detections) {
  $('#route-modules').replaceChildren(...detections.map(detection => {
    const chip = element('li', detectionName(detection) + ' · ' + Math.round(detection.confidence * 100) + '%', 'pill');
    chip.style.color = classColors[detection.class];
    return chip;
  }));
}

function archiveStations() {
  const stations = panoramaArchive.stations;
  switch (archiveFilter) {
    case 'missing': return stations.filter(station => !station.views_count);
    case 'panorama': return stations.filter(station => station.views_count);
    case 'equipment': return stations.filter(station => station.equipment);
    case 'manhole': return stations.filter(station => station.manhole);
    case 'false_positive': return [];
    default: return stations;
  }
}

function showArchiveStation(station, selectedFrame) {
  archiveSelection = station.id;
  const overview = station.equipment || station.panorama;
  const frame = selectedFrame || (archiveFilter === 'manhole' ? station.manhole : overview);
  activeSite = frame;
  activeRoute = { name: 'Архив панорам' };
  const kind = frame?.kind;
  const view = $('#route-view');
  view.hidden = !frame || kind === 'panorama';
  view.options[0].textContent = kind === 'manhole' ? 'Люк крупно' : 'Модули крупно';
  view.value = kind === 'panorama' ? 'frame' : 'modules';
  $('#route-evidence').hidden = false;
  $('#route-zoom').hidden = !frame;
  $('#route-evidence-title').textContent = 'Точка БС ' + station.id;
  const coordinates = station.latitude.toFixed(5) + ', ' + station.longitude.toFixed(5);
  const status = {
    odn: 'YOLO26s · предварительная детекция оборудования',
    manhole: 'YOLO26s для люков · принадлежность люка МГТС требует сверки',
    panorama: selectedFrame ? 'Исходные детекции на ракурсе' : 'Уверенных детекций оборудования нет',
  }[kind] || 'В архиве нет кадра для анализа';
  $('#route-evidence-meta').textContent = coordinates + ' · ' + status;
  renderRouteDetections(frame?.detections || []);
  const options = [
    ['Оборудование', station.equipment],
    ['Кандидат в люки', station.manhole],
    ...(station.frames || []).map((shot, index) => ['Ракурс ' + (index + 1), shot]),
  ];
  const buttons = options.filter(([, choice]) => choice).map(([label, choice]) => {
    const button = element('button', label);
    button.type = 'button';
    const active = choice.kind === kind && choice.image_url === frame?.image_url;
    button.setAttribute('aria-pressed', String(active));
    button.addEventListener('click', () => showArchiveStation(station, choice));
    return button;
  });
  $('#route-stops').replaceChildren(...buttons);
  if (frame) drawRouteFrame($('#route-canvas'), frame, view.value === 'frame');
  else $('#route-canvas').dataset.view = '';
}

function renderArchiveMap(points) {
  const stations = archiveStations();
  const colors = {
    equipment: '#ff829b', panorama: '#9ecbff', missing: '#9c94a7', manhole: '#ffe18b',
  };
  const errors = !archiveFilter || archiveFilter === 'false_positive'
    ? demoRoutes.flatMap(route => {
      const reviewed = route.sites.filter(site => site.review === 'false_positive');
      return reviewed.map(site => ({ site, route, reviewed }));
    })
    : [];
  const rejectedCovers = new Set(demoRoutes.flatMap(route => route.sites
    .filter(site => site.kind === 'manhole' && site.review === 'false_positive')
    .map(site => site.image_url)));
  const ordered = [[], [], []];
  for (const station of stations) {
    ordered[station.equipment ? 2 : station.views_count ? 1 : 0].push(station);
  }
  const addMarker = (station, cover) => {
    const point = cover ? [cover.observation_latitude, cover.observation_longitude] :
      [station.latitude, station.longitude];
    const category = cover ? 'manhole' : station.equipment ? 'equipment' : station.views_count ? 'panorama' : 'missing';
    const color = colors[category];
    points.push(point);
    const marker = L.circleMarker(point, {
      radius: cover ? 6 : station.equipment ? 4.5 : 3.4,
      color: cover ? color : '#1c1720', weight: cover ? 2 : .65,
      fillColor: color, fillOpacity: cover ? .32 : .9,
    }).addTo(layers);
    marker.bindTooltip(element('span', cover ? 'Кандидат в люки' : 'Точка БС ' + station.id));
    marker.on('click', () => {
      showArchiveStation(station, cover);
      if (window.matchMedia('(max-width: 800px)').matches) $('#route-evidence').scrollIntoView({ block: 'start' });
    });
  };
  const covers = stations.filter(station => station.manhole && !rejectedCovers.has(station.manhole.image_url));
  if (archiveFilter !== 'manhole') {
    for (const station of ordered.flat()) addMarker(station);
  }
  if (!archiveFilter || archiveFilter === 'manhole') {
    for (const station of covers) addMarker(station, station.manhole);
  }
  for (const { site, route, reviewed } of errors) addDemoMarker(site, route, points, reviewed);
  $('#map-summary').textContent = archiveFilter === 'false_positive'
    ? 'Ошибки модели · ' + errors.length
    : archiveFilter === 'manhole' ? 'Кандидаты в люки · ' + covers.length
      : 'Точки БС · ' + stations.length + (errors.length ? ' · ошибки ' + errors.length : '');
  if (archiveFilter === 'false_positive') {
    $('#geo-note').textContent = 'Проверенные примеры ошибок с исходными рамками модели';
  } else if (archiveFilter === 'manhole') {
    $('#geo-note').textContent = 'Точки съёмки кандидатов в люки · принадлежность требует проверки';
  } else {
    $('#geo-note').textContent = 'Исходные координаты БС';
  }
  const selectable = archiveFilter === 'manhole' ? covers : stations;
  const selected = selectable.find(station => station.id === archiveSelection)
    || selectable.find(s => s.equipment) || selectable[0];
  if (selected) showArchiveStation(selected);
  else if (errors.length) showRouteSite(errors[0].site, errors[0].route, errors[0].reviewed);
  else $('#route-evidence').hidden = true;
}

function addDemoMarker(site, route, points, reviewed) {
  const point = [site.latitude, site.longitude];
  const manhole = site.kind === 'manhole';
  const falsePositive = site.review === 'false_positive';
  const outlined = manhole || falsePositive;
  const color = falsePositive ? '#ff7f9a' : manhole ? '#ffe18b' : route.color;
  points.push(point);
  const marker = L.circleMarker(point, {
    infrastructureKind: site.kind, falsePositive,
    radius: outlined ? 6 : 5, color: outlined ? color : '#fff8fc',
    weight: outlined ? 2 : 1.5, fillColor: color, fillOpacity: outlined ? .28 : .85,
    dashArray: falsePositive ? '3 2' : undefined,
  }).addTo(layers);
  marker.bindTooltip(element('span', siteTitle(site)));
  marker.on('click', () => {
    showRouteSite(site, route, reviewed);
    if (window.matchMedia('(max-width: 800px)').matches) {
      $('#route-evidence').scrollIntoView({ block: 'start' });
    }
  });
  demoMarkers.set(site.frame, marker);
}

function initRouteEvidence() {
  $('#route-view').addEventListener('change', () => {
    if (activeSite) drawRouteFrame($('#route-canvas'), activeSite, $('#route-view').value === 'frame');
  });
  $('#route-zoom').addEventListener('click', () => {
    if (!activeSite) return;
    previewFrame = undefined;
    $('#preview-caption').textContent = siteTitle(activeSite);
    $('#preview-meta').textContent = activeRoute.name + ' · исходная точка съёмки ' +
      activeSite.observation_latitude.toFixed(5) + ', ' + activeSite.observation_longitude.toFixed(5);
    drawRouteFrame($('#preview-canvas'), activeSite, $('#route-view').value === 'frame');
    preview.showModal();
  });
}

function renderMap() {
  renderRouteButtons();
  if (!map) return;
  layers.clearLayers();
  demoMarkers.clear();
  const points = [];
  const showingResult = mapMode === 'result' && Boolean(current);
  const showingArchive = mapMode === 'archive' && Boolean(panoramaArchive);
  $('#route-layout').classList.toggle('result-view', showingResult);
  $('#map-filters').hidden = !showingResult;
  $('#archive-map-filters').hidden = !showingArchive;
  $('#demo-map-filters').hidden = showingResult || showingArchive;
  const legend = $('#map-legend');
  legend.hidden = showingResult || (!showingArchive && !demoRoutes.length);
  legend.dataset.mode = mapMode;
  legend.querySelectorAll('[data-archive-key]').forEach(item => { item.hidden = !showingArchive; });
  if (showingResult) {
    $('#route-evidence').hidden = true;
    const cameraPath = [];
    let segment = [];
    for (const frame of current.frames) {
      if (frame.coordinate_source === 'video_gps') {
        if (!segment.length) cameraPath.push(segment);
        segment.push([frame.latitude, frame.longitude]);
      } else {
        segment = [];
      }
    }
    for (const route of cameraPath) {
      points.push(...route);
      if (route.length > 1) {
        L.polyline(route, { color: '#ffd48d', dashArray: '5 7', weight: 3, interactive: false }).addTo(layers);
      }
    }
    const data = pointsAtSameLocation(filterRows());
    for (const { point, objects } of data) {
      points.push(point);
      const popup = element('div');
      for (const item of objects) {
        popup.append(element('strong', typeNames[item.kind]), element('p', statusNames[item.status]));
        if (item.coordinate_source === 'video_gps') popup.append(element('p', 'Позиция камеры'));
        popup.append(evidenceButton(item.frame));
      }
      L.circleMarker(point, { radius: 7, color: '#ffe3ed', weight: 1.5,
        fillColor: objects.some(item => item.kind === 'manhole') ? '#ff517b' : '#9cc2ff', fillOpacity: .95 })
        .addTo(layers).bindPopup(popup);
    }
    const located = data.reduce((sum, row) => sum + row.objects.length, 0);
    $('#map-summary').textContent = located + ' с координатами';
    $('#geo-note').textContent = cameraPath.length ? 'Позиции камеры по GPS · линия по кадрам с координатами' : located
      ? 'Точки по CSV или местоположению кадра'
      : current.frames.some(frame => frame.video_time != null)
        ? 'Для точек видео приложите GPS-трек CSV'
        : 'Для точек добавьте EXIF GPS или CSV';
  } else if (showingArchive) {
    renderArchiveMap(points);
  } else {
    const routes = demoRoutes.filter(route => (!routeFilter || route.id === routeFilter) && demoSites(route).length);
    for (const route of routes) {
      for (const site of demoSites(route)) addDemoMarker(site, route, points);
    }
    const frames = routes.reduce((count, route) => count + demoSites(route).length, 0);
    $('#map-summary').textContent = demoRoutes.length ? 'Точки на карте · ' + frames : 'Маршруты недоступны';
    $('#geo-note').textContent = demoRoutes.length ? 'Точки съёмки, привязанные к дороге' : '';
    const selectedRoute = routes.find(route => route.id === activeRoute?.id) || routes[0];
    if (selectedRoute) {
      const sites = demoSites(selectedRoute);
      showRouteSite(sites.find(site => site.frame === activeSite?.frame) || sites[0], selectedRoute);
    } else {
      $('#route-evidence').hidden = true;
    }
  }
  map.invalidateSize({ pan: false });
  map.options.zoomSnap = showingArchive ? .25 : 1;
  if (points.length) map.fitBounds(L.latLngBounds(points).pad(showingArchive ? .06 : .14), { maxZoom: 15, animate: false });
  else map.setView([55.75, 37.62], 10);
  renderRoadLabels();
}

function indexResultFrames() {
  const { frames, detections, observations } = current;
  detectionsByFrame = byFrame(detections);
  observationsByFrame = byFrame(observations);
  framesByName = new Map(frames.map(frame => [frame.name, frame]));
}

function renderResult() {
  const { stats, downloads, run_id } = current;
  indexResultFrames();
  $('#registry-empty').hidden = true;
  $('#results').hidden = false;
  $('#exports').hidden = false;
  $('#gallery-section').hidden = false;
  $('#stat-frames').textContent = stats.frames;
  $('#stat-objects').textContent = stats.objects;
  $('#stat-geo').textContent = stats.georeferenced_objects;
  $('#stat-actions').textContent = stats.actions;
  $('#run-meta').textContent = [current.model, deviceLabel(current)].filter(Boolean).join(' · ') + ' · ' +
    current.elapsed_seconds.toLocaleString('ru-RU') + ' с' + (current.cpu_fallback ? ' · резервный режим' : '');
  $('#download-actions').href = downloads.inspection;
  $('#download-registry').href = downloads.registry;
  $('#download-detections').href = downloads.detections;
  $('#open-result-map').href = '/routes?run=' + encodeURIComponent(run_id);
  $('#kind-filter').value = '';
  $('#status-filter').value = '';
  renderRegistry();
  renderGallery();
}

for (const selector of ['#kind-filter', '#status-filter']) {
  $(selector).addEventListener('change', () => {
    if (routePage) renderMap();
    else renderRegistry();
  });
}
function updateOverlays() {
  $('#confidence-value').textContent = $('#confidence-filter').value + '%';
  if (current) renderGallery();
  if (preview.open && previewFrame) drawFrame($('#preview-canvas'), previewFrame, 1920);
}
async function initRoutes() {
  initMap();
  initRouteEvidence();
  $('#archive-kind-filter').addEventListener('change', event => {
    archiveFilter = event.target.value;
    renderMap();
  });
  $('#demo-kind-filter').addEventListener('change', event => {
    demoKindFilter = event.target.value;
    if (routeFilter && !demoRoutes.some(route => route.id === routeFilter && demoSites(route).length)) {
      routeFilter = '';
    }
    renderMap();
  });
  const requests = [jsonRequest('/api/routes'), jsonRequest('/api/routes/archive')];
  if (sharedRun) requests.push(jsonRequest('/api/runs/' + encodeURIComponent(sharedRun)));
  const [routes, archive, result] = await Promise.allSettled(requests);
  if (routes.status === 'fulfilled') {
    demoRoutes = routes.value.routes;
    routeModel = routes.value.model || '';
  }
  if (archive.status === 'fulfilled') {
    panoramaArchive = archive.value;
    mapMode = 'archive';
  }
  if (result?.status === 'fulfilled') {
    current = result.value;
    indexResultFrames();
    mapMode = 'result';
    $('#route-registry-link').href = '/?run=' + encodeURIComponent(current.run_id) + '#registry';
  } else if (result?.status === 'rejected') {
    $('#route-error').textContent = 'Результат анализа не найден';
    $('#route-error').hidden = false;
  }
  renderMap();
}

if (routePage) {
  initRoutes();
} else {
  initUpload();
  if (sharedRun) {
    jsonRequest('/api/runs/' + encodeURIComponent(sharedRun)).then(result => {
      if (current || busy) return;
      current = result;
      renderResult();
    }).catch(error => showError(error.message));
  }
}
jsonRequest('/api/health').then(updateHealth).catch(() => {
  modelsReady = false;
  $('#health').className = 'health failed';
  $('#health span').textContent = 'Сервис недоступен';
  const condition = $('#condition-state');
  if (condition) {
    condition.textContent = 'Не удалось проверить модели';
    updateBusy();
  }
});
