# Мониторинг городской инфраструктуры МГТС

Прототип анализирует фото и короткие видео с городских маршрутов. Две обученные YOLO26s находят опоры с оборудованием и крышки люков. Результаты доступны в реестре и на карте, а полный список и очередь на проверку или ремонт выгружаются в CSV.

В репозитории уже есть рабочие веса, шесть кадров для сквозного демо и 118 точек для просмотра на карте. Команда `git lfs pull` для запуска не требуется.

## Скачать проект

```sh
git clone https://github.com/pocketgodru/mgts_hack.git
cd mgts_hack
```

## Запуск без Docker

На Windows нужен Python 3.12 x64. Откройте PowerShell в папке проекта и создайте виртуальное окружение.

```powershell
py -3.12 -m venv .venv
$python = '.\.venv\Scripts\python.exe'
& $python -m pip install --upgrade pip
```

### Windows · CPU

```powershell
& $python -m pip install 'torch==2.12.0+cpu' 'torchvision==0.27.0+cpu' --index-url https://download.pytorch.org/whl/cpu
& $python -m pip install -r requirements.runtime.txt
.\start_prototype.ps1 -Device cpu
```

### Windows · AMD Radeon RX 9070 XT

Установите совместимый драйвер AMD. Для другой видеокарты проверьте архитектуру и команду установки в [документации AMD](https://rocm.docs.amd.com/projects/ai-ecosystem/en/latest/frameworks/pytorch/install.html).

```powershell
& $python -m pip install --index-url https://stable.repo.amd.com/rocm/whl-next/ --extra-index-url https://pypi.org/simple 'torch[device-gfx1201]==2.15.0+rocm10.2.0' 'torchvision[device-gfx1201]==0.29.2+rocm10.2.0'
& $python -m pip install -r requirements.runtime.txt
& $python -c "import torch; assert torch.version.hip and torch.cuda.is_available(); print(torch.cuda.get_device_name(0), torch.__version__)"
.\start_prototype.ps1 -Device auto
```

### Linux · CPU

```sh
python3.12 -m venv .venv
. .venv/bin/activate
python -m pip install 'torch==2.12.0+cpu' 'torchvision==0.27.0+cpu' --index-url https://download.pytorch.org/whl/cpu
python -m pip install -r requirements.runtime.txt
MGTS_DEVICE=cpu python -m uvicorn web_app:app --host 127.0.0.1 --port 8000
```

## Запуск с Docker

Собирайте образ из корня скачанного проекта. На Windows Docker Desktop должен работать в режиме Linux-контейнеров.

### CPU

```sh
docker build --build-arg MGTS_TORCH_BACKEND=cpu -t mgts-hack:cpu .
docker run --rm -p 127.0.0.1:8000:8000 -v mgts-hack-data:/data mgts-hack:cpu
```

### ROCm · AMD Radeon RX 9070 XT на Linux

Сборка устанавливает PyTorch ROCm для `gfx1201`. Контейнер перед стартом проверяет доступность GPU, выполняет свёртку и NMS. На Linux-хосте должны быть доступны `/dev/kfd` и каталог `/dev/dri`.

```sh
docker build --build-arg MGTS_TORCH_BACKEND=rocm -t mgts-hack:rocm .
docker run --rm -p 127.0.0.1:8000:8000 -v mgts-hack-data:/data \
  --device=/dev/kfd --device=/dev/dri \
  --group-add "$(stat -c '%g' /dev/kfd)" \
  --group-add "$(stat -c '%g' /dev/dri/renderD128)" \
  mgts-hack:rocm
```

Если узел вашей карты отличается от `renderD128`, подставьте его имя. Для одной проверки GPU без веб-сервера добавьте `--check` после имени образа.

## Проверить работу

1. Откройте [интерфейс](http://127.0.0.1:8000/).
2. Нажмите **Запустить демо**. После обработки откройте реестр, затем карту или выгрузку CSV.
3. Для своих данных загрузите до 100 изображений JPEG или PNG либо одно видео MP4, MOV или WebM до 60 секунд и 150 МБ. Формы загрузки содержат шаблоны CSV для координат фото и GPS-трека видео.

В [проверке готовности](http://127.0.0.1:8000/api/health) отображаются обе модели и выбранное устройство. Спутниковая подложка загружается через интернет. Демо-кадры и данные точек входят в проект.

Координаты в реестре обозначают место съёмки. Принадлежность находки МГТС и координаты самого объекта можно подтвердить по инвентарю.

## Веса

| Файл | Задача |
|---|---|
| `models/odn-yolo26s.pt` | Детекция опор, антенн, радиомодулей и РРЛ |
| `models/manhole-yolo26s.pt` | Детекция крышек люков |

Для оценки состояния можно подключить обученный классификатор и сегментатор через `models/condition.json`. Шаблон лежит в `models/condition.example.json`. Пока веса состояния отсутствуют, объект получает статус «Не оценивалось».

Для подготовки и обучения моделей состояния есть `final_train.py`. Для подготовки классификации и импорта Excel установите дополнительные зависимости.

```sh
python -m pip install scikit-learn openpyxl
python final_train.py --help
```

Передайте собственные стартовые веса задачи через `--weights`. Флаг `--install` подключает результат к локальному приложению. Датасеты и результаты пользовательских прогонов хранятся локально.
