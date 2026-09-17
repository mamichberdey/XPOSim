# XPOSim

> **X-ray Paraxial Optical Simulation**

**XPOSim** — численная библиотека для моделирования **параксиальной волновой оптики рентгеновских схем** в **1D и 2D**.

Главная идея библиотеки — предельно простой интерфейс:

```text
        complex E(x) / E(x,y)
                 │
                 ▼
          ┌──────────────┐
          │ Optical      │
          │ element      │
          └──────────────┘
                 │
                 ▼
        transformed E-field
                 │
                 ▼
          Fresnel propagation
                 │
                 ▼
                E(z)
```

Каждый оптический элемент получает **готовый комплексный массив поля** и преобразует его. Единственное исключение — `PointSource`, который поле создаёт.

---

## ✨ Основные возможности

- 📐 **1D и 2D** расчёты через независимые `Grid1D` / `Grid2D`
- ⚡ **CuPy/GPU** и **NumPy/CPU**
- 🔄 автоматическое сохранение backend и формы поля
- 🌊 параксиальное **Fresnel propagation**
- 🎯 точечный источник
- ◯ симметричная двухдырочная апертура
- ◣ blade-маска
- 🔬 **CRL** в 1D и 2D
- 🔄 поворот цилиндрической CRL через `rotation=(rx, ry, rz)`
- 🧩 произвольные **дефекты толщины CRL** без создания новых классов линз
- 📏 теоретический и эффективный фокус CRL
- ⚠️ автоматическая проверка максимально допустимого числа линз
- 🔎 численный `focus_scan()` без повторного пересчёта самой CRL
- 📊 FWHM, центр пучка, максимум интенсивности и divergence
- 💾 ROI вместо хранения огромного полного 3D-стека изображений
- 🧱 табличные X-ray material constants с кэшированием

Архитектура и семантика `E(z)` зафиксированы в текущем API: `E(0)` — поле сразу после элемента, `E(z)` — поле после дополнительного распространения. fileciteturn7file1L1020-L1039

---

## ⚠️ Физическая область применимости

XPOSim — **параксиальный** wave-optics solver.

Это означает, что библиотеку не следует воспринимать как полноценный non-paraxial или Maxwell solver. Особенно важно учитывать это при работе с большими углами, сильными наклонами элементов и конфигурациями, где формальный фокус CRL оказывается внутри самой линзы.

Текущая реализация поддерживает только in-plane rotation `rz` для поперечной 2D CRL. Ненулевые `rx`/`ry` намеренно отклоняются, поскольку полноценная модель наклонённого элемента пока не реализована. fileciteturn7file0L959-L1004

---

# 🚀 Quick start

### 1. Создать 2D сетку

```python
import numpy as np
import XPOSim as xpos

ENERGY = 12.0

grid = xpos.Grid2D(
    dx=5e-8,
    dy=5e-8,
    nx=1024,
    ny=1024,
    backend="cupy",
)
```

### 2. Создать источник

```python
source = xpos.PointSource(
    grid=grid,
    position=(0.0, 0.0),
    energy_kev=ENERGY,
)

E0 = source.E(z=15.0)
```

`PointSource.E()` требует `z > 0`, поскольку точечное поле в самой точке источника сингулярно. fileciteturn7file0L775-L831

### 3. Добавить CRL

```python
material = xpos.Material(
    formula="Si",
    density=2.33,
)

crl = xpos.CRL(
    input_field=E0,
    grid=grid,
    radius=6.25e-6,
    aperture=50e-6,
    bridge_thickness=2e-6,
    n_lenses=20,
    material=material,
    energy_kev=ENERGY,
)
```

### 4. Проверить физическую границу

```python
print("maximum valid lens count:", crl.max_lens_count())
print("theoretical focus:", crl.theoretical_focus())
```

Если при создании CRL `n_lenses` превышает допустимое число, библиотека выдаёт `RuntimeWarning`. fileciteturn7file1L1060-L1068

### 5. Найти численный фокус

```python
import numpy as np

z = np.linspace(1e-3, 50e-3, 100)

scan = crl.focus_scan(
    z,
    roi=((-10e-6, 10e-6), (-10e-6, 10e-6)),
)

print("numerical focus:", scan.peak_z)
print("FWHM at focus:", scan.fwhm_at_peak)
```

`focus_scan()` вычисляет выходное поле CRL один раз и далее только распространяет его на разные `z`. fileciteturn8file0L357-L413

---

# 🧠 Core concept

XPOSim сознательно не вводит отдельный `Wavefront` object: поле остаётся обычным `numpy.ndarray` или `cupy.ndarray`.

Схема строится напрямую:

```python
E0 = source.E(z=100e-3)

aperture = xpos.Hole(
    input_field=E0,
    grid=grid,
    center_offset=(20e-6, 0.0),
    radius=5e-6,
    energy_kev=ENERGY,
)

E1 = aperture.E(z=50e-6)

crl = xpos.CRL(
    input_field=E1,
    grid=grid,
    radius=6.25e-6,
    aperture=50e-6,
    bridge_thickness=2e-6,
    n_lenses=20,
    material=xpos.Material("Si", density=2.33),
    energy_kev=ENERGY,
)

E2 = crl.E()
```

То есть **каждый объект трансформирует массив поля**, а не создаёт сложный граф зависимостей между объектами. Это центральный принцип библиотеки. fileciteturn7file0L734-L767

---

# 📐 1D и 2D

## 1D

```python
grid = xpos.Grid1D(
    dx=5e-8,
    nx=4096,
    backend="cupy",
)
```

Поле имеет форму:

```text
(nx,)
```

## 2D

```python
grid = xpos.Grid2D(
    dx=5e-8,
    dy=5e-8,
    nx=2048,
    ny=2048,
    backend="cupy",
)
```

Поле имеет форму:

```text
(ny, nx)
```

1D и 2D — разные grid modes. Элементы не должны смешивать массивы разных размерностей, а backend и shape поля должны совпадать с grid. fileciteturn7file0L499-L530

---

# 🔬 CRL

`CRL` — один класс для всех основных случаев:

### Настоящая 1D схема

```python
crl = xpos.CRL(
    input_field=E0,
    grid=grid1d,
    radius=R,
    aperture=A,
    bridge_thickness=d,
    n_lenses=N,
    material=material,
    energy_kev=ENERGY,
)
```

### Обычная 2D CRL

```python
crl = xpos.CRL(
    input_field=E0,
    grid=grid2d,
    radius=R,
    aperture=A,
    bridge_thickness=d,
    n_lenses=N,
    material=material,
    energy_kev=ENERGY,
)
```

### Одноосевая CRL в 2D

```python
crl = xpos.CRL(
    input_field=E0,
    grid=grid2d,
    radius=R,
    aperture=A,
    bridge_thickness=d,
    n_lenses=N,
    material=material,
    transverse_length=33e-6,
    rotation=(0.0, 0.0, np.pi / 2),
    energy_kev=ENERGY,
)
```

`transverse_length` включает цилиндрическую геометрию: локальная `x` является фокусирующим направлением, локальная `y` задаёт конечную поперечную ширину. `rz` поворачивает эту локальную систему в мировой `x/y` плоскости. fileciteturn7file0L971-L1004

---

# 📏 Physical CRL limit

Для CRL:

```text
p  = d + A²/(4R)
Lc = sqrt(pR/(2δ))
Ncritical = (π/2) * Lc/p
```

В API:

```python
crl.lens_pitch
crl.characteristic_length
crl.characteristic_lens_count
crl.critical_lens_count()
crl.max_lens_count()
```

`max_lens_count()` возвращает последний допустимый integer count до критической границы. При слишком большом `n_lenses` CRL предупреждает пользователя непосредственно при инициализации. fileciteturn7file1L1041-L1068

---

# 🧩 Defects

Старые многочисленные классы CRL с дефектами заменены единым механизмом:

```python
def defect(thickness, grid, lens_index):
    return thickness + ...
```

и:

```python
crl = xpos.CRL(
    ...,
    thickness_defect=defect,
)
```

Дефект меняет **готовую карту толщины**, а затем библиотека сама строит transmission. Можно использовать один дефект для всех линз или передать отдельный дефект для каждого элемента. fileciteturn7file1L1070-L1078

---

# 📊 Beam analysis

```python
metrics = xpos.beam_metrics(
    field,
    grid,
)
```

Результат содержит:

```python
metrics.max_intensity
metrics.center
metrics.fwhm
metrics.divergence
```

Также доступны:

```python
xpos.full_width_half_max(...)
xpos.divergence(...)
xpos.intensity(...)
```

В 2D FWHM возвращается как `(fwhm_x, fwhm_y)`, а центр как `(x, y)`. fileciteturn8file0L421-L469 fileciteturn8file0L559-L651

---

# 💻 NumPy / CuPy

Для CPU:

```python
grid = xpos.Grid2D(..., backend="numpy")
```

Для GPU:

```python
grid = xpos.Grid2D(..., backend="cupy")
```

При этом `E` остаётся тем же типом массива:

```text
CuPy → element → CuPy
NumPy → element → NumPy
```

Grid проверяет backend и форму поля при передаче в оптический элемент. fileciteturn7file0L504-L513

---

# 📚 Material data

Структура данных:

```text
xray_material_data/
├── f1f2_Windt.dat
├── CrossSec-Compton_McMaster.dat
└── AtomicConstants.dat
```

`Material` использует их для вычисления `delta` и `beta` и кэширует разобранные элементные таблицы. fileciteturn7file2L1124-L1134

---

# 📁 API at a glance

### Core

```python
xpos.Grid1D
xpos.Grid2D
xpos.PointSource
xpos.OpticalObject
xpos.OpticalElement
```

### Optical elements

```python
xpos.Hole
xpos.Blade
xpos.CRL
```

### Materials

```python
xpos.Material
xpos.wavelength_energy
```

### CRL defects

```python
xpos.ThicknessDefect
xpos.FunctionThicknessDefect
```

### Analysis

```python
xpos.intensity
xpos.full_width_half_max
xpos.beam_metrics
xpos.divergence
xpos.gaussian_filter_1d
xpos.gaussian_filter_2d
```

### Results

```python
xpos.BeamMetrics
xpos.FocusScanResult
```

Все эти публичные объекты экспортируются через `XPOSim.__all__`. fileciteturn7file0L51-L74

---

# 🎯 Design philosophy

XPOSim намеренно придерживается небольшой модели:

```text
Grid
 ↓
complex E array
 ↓
Optical element
 ↓
complex E array
```

Вместо множества специализированных классов CRL используются композиция и дефекты толщины. Вместо глобальной сетки у каждого расчёта есть собственный `Grid`. Вместо хранения положения `z` внутри объекта расстояние задаётся непосредственно вызовом `E(z)`. Это делает 1D и 2D модели единообразными на уровне API, не смешивая их вычислительные сетки. fileciteturn7file0L13-L26

---

# 🧪 Recommended workflow

```text
Create Grid
    ↓
Create PointSource
    ↓
source.E(z)
    ↓
Create optical element
    ↓
element.E(z)
    ↓
Create next element
    ↓
...
    ↓
Beam analysis / focus scan
```

Для CRL:

```text
CRL
 ├─ max_lens_count()
 ├─ theoretical_focus()
 ├─ effective_focus()
 ├─ thickness()
 ├─ transmission()
 └─ focus_scan()
```

---

# 📖 Documentation

Полное руководство пользователя:

**[`USER_GUIDE.md`](USER_GUIDE.md)**

Там подробно описаны:

- 1D/2D grids;
- источники;
- `E(z)`;
- CRL geometry;
- rotation;
- material database;
- defects;
- `max_lens_count()`;
- focus scan;
- FWHM/divergence;
- NumPy/CuPy;
- типовые ошибки;
- полные примеры 1D и 2D схем.

---

# 📦 Project structure

Минимальная структура проекта:

```text
XPOSim/
├── XPOSim.py
├── xray_material_data/
│   ├── f1f2_Windt.dat
│   ├── CrossSec-Compton_McMaster.dat
│   └── AtomicConstants.dat
├── README.md
└── USER_GUIDE.md
```

---

# 🔖 Version

Current API:

```python
import XPOSim as xpos
print(xpos.__version__)
```

```text
0.3.0
```

---

## License

Add your project license here.
