# Bat Jackson Optimization App

Aplikacja w języku Python służąca do optymalizacji liczby kanałów obsługi ($M/M/c$) w otwartych sieciach kolejkowych Jacksona za pomocą **Algorytmu Nietoperza (Bat Algorithm - BA)**.

Projekt został zrealizowany w ramach przedmiotu *Algorytmy i metody optymalizacji* (Akademia Tarnowska).

---

## 🚀 Główne Cechy Projektu

1. **Wydajny Algorytm Nietoperza (Bat Algorithm)**:
   - Zoptymalizowana wektorowo implementacja w **NumPy** z kompilacją JIT (**Numba**),
   - Obsługa całkowitoliczbowych zmiennych decyzyjnych ($c_i \in [c_{min}, c_{max}]$),
   - Sprawdzanie warunku stabilności sieci ($\rho_i < 1.0$),
   - Funkcja celu minimalizująca łączny czas przebywania w sieci oraz koszt kanałów:
     $$f(c) = \alpha \cdot W_{sieci} + \beta \cdot \sum_{i=1}^n k_i c_i$$

2. **Czysta Architektura i Separacja Zadań**:
   - Moduł `src.core` zawiera niezależny od zagadnienia optymalizator BA z protokołem `ObjectiveFunctionProtocol`,
   - Moduł `src.evaluation` zawiera adapter `JacksonNetworkEvaluator` realizujący model matematyczny sieci Jacksona i węzłów $M/M/c$, co umożliwia łatwą integrację z modułem rozwijanym przez drugiego programistę,
   - Dostępne ewaluatory funkcji benchmarkowych (`SphereEvaluator`, `RastriginEvaluator`).

3. **Przejrzysty Interfejs Użytkownika (GUI)**:
   - Okienkowy interfejs oparty na bibliotece `Tkinter` / `ttk` (działa bez dodatkowych serwerów graficznych na Linuxie oraz Windowsie),
   - Wykres zbieżności funkcji celu w czasie rzeczywistym dzięki integracji z biblioteką `Matplotlib`,
   - Możliwość uruchomienia zarówno w trybie okienkowym (GUI), jak i konsolowym (CLI).

4. **Kompilacja Standalone (Wieloplatformowość)**:
   - Gotowy skompilowany plik binarny w katalogu `dist/` (niewymagający lokalnej instalacji Pythona ani bibliotek zewnętrznych),
   - Kompatybilność z systemami Linux oraz Windows.

---

## 📁 Struktura Projektu

```
bat-jackson-app/
├── dist/                       # Skompilowane pliki binarne (PyInstaller)
│   └── bat_opt_app             # Samodzielny plik wykonywalny Linux
├── src/                        # Kod źródłowy aplikacji
│   ├── core/                   # Rdzeń algorytmu nietoperza
│   │   ├── parameters.py       # Hiperparametry algorytmu (BatParameters)
│   │   ├── types.py            # Typy danych i protokoły (OptimizationResult)
│   │   ├── bat_kernels.py      # Akceleracja numeryczna (Numba/NumPy)
│   │   └── bat_algorithm.py    # Klasa BatAlgorithm
│   ├── evaluation/             # Ewaluacja funkcji celu i sieci Jacksona
│   │   ├── base.py             # Abstrakcyjna klasa bazowa BaseEvaluator
│   │   ├── benchmark_funcs.py  # Funkcje benchmarkowe (Sphere, Rastrigin)
│   │   └── jackson_adapter.py  # Model matematyczny otwartej sieci Jacksona M/M/c
│   ├── gui/                    # Interfejs graficzny
│   │   ├── app.py              # Główne okno aplikacji (Tkinter)
│   │   └── components.py       # Panele parametrów i tabel wyników
│   └── main.py                 # Punkt wejściowy programu (GUI/CLI)
├── tests/                      # Zestaw testów automatycznych (pytest)
│   ├── test_bat_algorithm.py   # Testy zbieżności i hiperparametrów
│   ├── test_evaluator.py       # Testy równań ruchu, stabilności i M/M/c
│   └── test_performance.py     # Benchmarki wydajnościowe
├── pyproject.toml              # Zależności i konfiguracja projektu
├── .gitignore                  # Pliki ignorowane przez Git
└── README.md                   # Niniejsza dokumentacja
```

---

## 🛠️ Wymagania i Uruchomienie

### Wymagania:
- Python 3.11+ (zalecany Python 3.12)
- Narzędzie `uv` lub tradycyjny `pip`

### 1. Uruchomienie ze skompilowanego pliku binarnego (bez Pythona)
W katalogu głównym:
```bash
./dist/bat_opt_app
```
Aby uruchomić binarkę w trybie konsolowym (CLI):
```bash
./dist/bat_opt_app --cli --pop 30 --iters 100
```

### 2. Uruchomienie ze źródeł za pomocą `uv`
Zainstaluj środowisko:
```bash
uv venv --python 3.12
source .venv/bin/activate
uv pip install -e ".[dev]"
```
Uruchom aplikację GUI:
```bash
python src/main.py
```
Lub w trybie CLI:
```bash
python src/main.py --cli
```

### 3. Uruchomienie testów jednostkowych
```bash
pytest tests/ -v
```

---

## 📦 Budowanie Pliku Wykonywalnego (`.exe` / binarka Linux)

Aby zbudować samodzielny plik za pomocą PyInstaller:
```bash
pyinstaller --onefile --name bat_opt_app --clean --collect-all matplotlib --collect-all numba src/main.py
```
Skompilowany plik zostanie wygenerowany w katalogu `dist/` (na platformie Windows powstanie `dist/bat_opt_app.exe`, na Linuxie `dist/bat_opt_app`).

---

## 🌿 Przepływ Pracy Git (Git Workflow)
Projekt był rozwijany zgodnie z przyjętą metodyką gałęzi:
- `master` – stabilna gałąź produkcyjna ze zbudowanym programem,
- `dev` – główna gałąź integracyjna,
- `feature/*` – dedykowane gałęzie funkcjonalne:
  - `feature/core-bat-algorithm`
  - `feature/evaluation-adapter`
  - `feature/unit-tests-benchmark`
  - `feature/user-interface`
  - `feature/build-and-docs`
Wszystkie commity sformatowano w języku angielskim z opisem 2–5 słów.
