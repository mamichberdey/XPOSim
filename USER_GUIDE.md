# XPOSim 0.3.0 — User Guide

**XPOSim** — численная библиотека для моделирования параксиальной волновой оптики в рентгеновских схемах в 1D и 2D.

Центральная идея библиотеки очень простая:

```text
complex E-field
      │
      ▼
 optical element
      │
      ▼
complex E-field
      │
      └── Fresnel propagation → E(z)
```

Обычный оптический элемент получает готовый массив комплексной амплитуды `E`, преобразует его и предоставляет `E(z)`. Точечный источник является исключением: он не получает входное поле, а сам создаёт его. В текущей реализации `E(z=0)` означает поле непосредственно после элемента, а `E(z)` — это это же поле после Fresnel-распространения на расстояние `z`. fileciteturn7file0L11-L26 fileciteturn7file1L1020-L1039

> **Важно:** XPOSim является параксиальным решателем. Он предназначен для схем, в которых используется параксиальное приближение и Fresnel propagation. Это не полноценный non-paraxial Maxwell solver.

---

## 1. Установка структуры

XPOSim поставляется как один основной модуль `XPOSim.py` и каталог с таблицами параметров материалов:

```text
project/
├── XPOSim.py
└── xray_material_data/
    ├── f1f2_Windt.dat
    ├── CrossSec-Compton_McMaster.dat
    └── AtomicConstants.dat
```

Каталог `xray_material_data/` является новым названием старого `chi0_chih/`. `Material` ожидает именно эти три файла и кэширует прочитанные таблицы в памяти. fileciteturn7file2L1124-L1134 fileciteturn7file1L1080-L1094

В notebook:

```python
import XPOSim as xpos
```

Версия текущего API:

```python
print(xpos.__version__)
# 0.3.0
```

---

# 2. Главная концепция: Grid + E-field + objects

В XPOSim есть два независимых режима сетки:

- `Grid1D` — поле имеет форму `(nx,)`;
- `Grid2D` — поле имеет форму `(ny, nx)`.

Оптический объект привязан к одной сетке. Входной массив и выходной массив должны иметь ту же размерность, форму и backend. NumPy и CuPy поддерживаются, причём backend поля сохраняется при преобразованиях. fileciteturn7file0L499-L530 fileciteturn7file1L1033-L1039

Типичная схема выглядит так:

```python
import XPOSim as xpos

# 1. Создаём grid
# 2. Создаём PointSource
# 3. Получаем E(z) в плоскости первого элемента
# 4. Создаём элемент, передав ему этот E
# 5. Получаем E(z) после элемента
# 6. Передаём результат следующему элементу
```

Важный принцип: **объект не хранит положение по `z` как часть своего состояния**. Положение плоскости наблюдения задаётся непосредственно вызовом `E(z=...)`. fileciteturn7file0L734-L767

---

# 3. Единицы измерения

Внутренние физические единицы XPOSim:

| Величина | Единица |
|---|---|
| координаты | м |
| `dx`, `dy` | м |
| расстояния `z` | м |
| радиусы и апертура | м |
| толщина | м |
| энергия фотона | кэВ |
| углы `rotation` | радианы |
| `delta`, `beta` | безразмерные |

Для отображения удобно переводить значения в `µm`, `mm`, `mrad` уже в notebook.

---

# 4. Создание 1D-сетки

```python
grid = xpos.Grid1D(
    dx=5e-8,
    nx=4096,
    backend="cupy",
)
```

Основные атрибуты:

```python
grid.x          # пространственная координата

grid.qx         # поперечный волновой вектор

grid.dx         # шаг по x
grid.dqx        # шаг по qx

grid.nx         # число отсчётов
grid.shape      # (nx,)
grid.dimension  # 1
```

`grid.x` и `grid.qx` находятся на том же backend, который выбран для сетки. fileciteturn7file0L532-L566

### NumPy

```python
grid = xpos.Grid1D(
    dx=5e-8,
    nx=4096,
    backend="numpy",
)
```

### CuPy

```python
grid = xpos.Grid1D(
    dx=5e-8,
    nx=4096,
    backend="cupy",
)
```

CuPy является предпочтительным backend для больших расчётов, если доступна совместимая GPU-система. Сам XPOSim не требует CuPy для NumPy-режима. fileciteturn7file0L40-L48

---

# 5. Создание 2D-сетки

```python
grid = xpos.Grid2D(
    dx=5e-8,
    dy=5e-8,
    nx=2048,
    ny=2048,
    backend="cupy",
)
```

Основные атрибуты:

```python
grid.x            # 2D meshgrid координаты x
grid.y            # 2D meshgrid координаты y

grid.x_axis       # 1D ось x
grid.y_axis       # 1D ось y

grid.qx           # 2D meshgrid qx
grid.qy           # 2D meshgrid qy

grid.qx_axis      # 1D ось qx
grid.qy_axis      # 1D ось qy

grid.dx
grid.dy
grid.dqx
grid.dqy
grid.nx
grid.ny
grid.shape       # (ny, nx)
grid.dimension    # 2
```

Расположение индексов согласовано с NumPy-массивом: первый индекс — `y`, второй — `x`. fileciteturn7file0L598-L646

---

# 6. Точечный источник

`PointSource` создаёт параксиальное сферическое поле.

## 1D

```python
source = xpos.PointSource(
    grid=grid,
    position=0.0,
    energy_kev=12.0,
)
```

Получение поля:

```python
E = source.E(z=0.1)
```

Для источника `z` обязан быть положительным: поле непосредственно в точке источника (`z=0`) в виде обычного массива не определяется из-за сингулярности точечного источника. fileciteturn7file0L775-L831

## 2D

```python
source = xpos.PointSource(
    grid=grid,
    position=(0.0, 0.0),
    energy_kev=12.0,
)

E = source.E(z=0.1)
```

Нецентральный источник:

```python
source = xpos.PointSource(
    grid=grid,
    position=(10e-6, -5e-6),
    energy_kev=12.0,
)
```

---

# 7. Энергия и длина волны

Для оптического объекта можно задавать либо:

```python
energy_kev=12.0
```

либо:

```python
wavelength_m=1.0332e-10
```

Но не оба сразу.

Также доступна функция:

```python
wavelength, energy = xpos.wavelength_energy(energy_kev=12.0)
```

или:

```python
wavelength, energy = xpos.wavelength_energy(
    wavelength_m=1.0332e-10
)
```

Функция возвращает `(wavelength_m, energy_kev)`. fileciteturn7file0L139-L173

### Практический совет

При построении схемы удобнее определить энергию один раз:

```python
ENERGY = 12.0
```

и передавать её всем элементам:

```python
hole = xpos.Hole(
    input_field=E0,
    grid=grid,
    center_offset=(20e-6, 0.0),
    radius=5e-6,
    energy_kev=ENERGY,
)
```

В текущем API оптический объект не пытается угадывать энергию из произвольного массива `E`; `wavelength_m` или `energy_kev` должны быть явно заданы при создании объекта. fileciteturn7file0L690-L715

---

# 8. Общий интерфейс оптического элемента

Все элементы, которые преобразуют существующее поле, наследуются от `OpticalElement`.

Их базовый контракт:

```python
E(z=0)
```

возвращает поле непосредственно после элемента.

```python
E(z=distance)
```

возвращает поле после Fresnel propagation на `distance` метров от элемента.

```python
I(z=distance)
```

возвращает:

```text
|E(z)|²
```

Также есть описательный alias:

```python
intensity(z=distance)
```

Эта семантика является основным API всей библиотеки. fileciteturn7file0L734-L772

---

# 9. Передача поля между элементами

Самый важный паттерн XPOSim:

```python
E0 = source.E(z=100e-3)

hole = xpos.Hole(
    input_field=E0,
    grid=grid,
    center_offset=(20e-6, 0.0),
    radius=5e-6,
    energy_kev=12.0,
)

E1 = hole.E()

crl = xpos.CRL(
    input_field=E1,
    grid=grid,
    radius=6.25e-6,
    aperture=50e-6,
    bridge_thickness=2e-6,
    n_lenses=20,
    material=xpos.Material("Si", density=2.33),
    energy_kev=12.0,
)

E2 = crl.E()
```

То есть объект получает **массив**, а не ссылку на другой объект. Это делает схемы линейно компонуемыми и оставляет `E` обычным NumPy/CuPy массивом. Поля не оборачиваются в дополнительный объект-обёртку. fileciteturn7file0L734-L767

---

# 10. Hole

`Hole` создаёт симметричную пару отверстий.

## 2D

```python
hole = xpos.Hole(
    input_field=E0,
    grid=grid,
    center_offset=(20e-6, 0.0),
    radius=5e-6,
    energy_kev=12.0,
)
```

Центры отверстий:

```text
(+x0, +y0)
(-x0, -y0)
```

## 1D

```python
hole = xpos.Hole(
    input_field=E0,
    grid=grid,
    center_offset=20e-6,
    radius=5e-6,
    energy_kev=12.0,
)
```

В 1D создаются два симметричных интервала вокруг `+x0` и `-x0`.

Маску можно получить отдельно:

```python
mask = hole.transmission_mask()
```

При пересечении двух отверстий маска остаётся бинарной: перекрытие не создаёт значение `2`. fileciteturn7file0L834-L897

---

# 11. Blade

`Blade` сохраняет только положительную часть пространства.

В 1D:

```text
x >= 0
```

В 2D:

```text
x >= 0 AND y >= 0
```

```python
blade = xpos.Blade(
    input_field=E0,
    grid=grid,
    energy_kev=12.0,
)

E1 = blade.E()
```

Получить маску:

```python
mask = blade.transmission_mask()
```

Такое определение сохраняет конвенцию старой библиотеки. fileciteturn7file0L900-L917

---

# 12. Material

Материал задаётся химической формулой и плотностью:

```python
si = xpos.Material(
    formula="Si",
    density=2.33,
)
```

Для соединения:

```python
material = xpos.Material(
    formula="AlSe8O15",
    density=2.5,
)
```

Оптические константы:

```python
delta, beta = material.optical_constants(
    energy_kev=12.0
)
```

`Material` поддерживает лёгкий разбор формул вида `Si` или `AlSe8O15`. Скобочные формулы намеренно не поддерживаются текущей реализацией. Табличные `f1`, `f2`, Compton corrections и atomic masses загружаются из `xray_material_data/`. Значения энергии вне диапазона таблицы отклоняются, а не экстраполируются молча. fileciteturn7file0L297-L412

### Прямое задание `delta` и `beta`

Для тестов или пользовательских материалов можно не использовать таблицы:

```python
material = xpos.Material(
    delta=3.0e-6,
    beta=2.0e-9,
)
```

---

# 13. CRL

`CRL` — единый класс для:

- настоящей 1D-схемы;
- обычной rotationally symmetric 2D CRL;
- одномерно фокусирующей CRL внутри 2D схемы.

В параксиальном приближении один элемент CRL состоит из двух противоположных параболических поверхностей и перемычки `d`.

Полная толщина одного элемента:

```text
T = 2 * sag + d
```

где `d` — расстояние между двумя противоположными параболическими поверхностями. fileciteturn7file0L959-L1008

## Базовая CRL

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
    n_lenses=26,
    material=material,
    energy_kev=12.0,
)
```

Параметры:

| Параметр | Смысл |
|---|---|
| `radius` | радиус кривизны параболических поверхностей `R` |
| `aperture` | полная оптическая апертура `A` |
| `bridge_thickness` | `d`, расстояние между поверхностями |
| `n_lenses` | число элементов CRL |
| `material` | материал линзы |
| `transverse_length` | поперечная физическая длина цилиндрической CRL в 2D |
| `rotation` | `(rx, ry, rz)` |
| `thickness_defect` | модель дефекта толщины |
| `reuse_lens_geometry` | переиспользовать ли одну геометрию для всех одинаковых линз |

---

# 14. Настоящая 1D CRL

В `Grid1D` единственная координата является фокусирующей координатой.

```python
grid = xpos.Grid1D(
    dx=5e-8,
    nx=4096,
    backend="cupy",
)

source = xpos.PointSource(
    grid=grid,
    position=0.0,
    energy_kev=12.0,
)

E0 = source.E(z=15.0)

crl = xpos.CRL(
    input_field=E0,
    grid=grid,
    radius=6.25e-6,
    aperture=50e-6,
    bridge_thickness=2e-6,
    n_lenses=26,
    material=xpos.Material("Si", density=2.33),
    energy_kev=12.0,
)

E_image = crl.E(z=0.01)
```

`transverse_length` для `Grid1D` не задаётся: он имеет смысл только для цилиндрической CRL в 2D. fileciteturn8file0L45-L70

---

# 15. Одноосевая CRL внутри 2D схемы

Если в 2D задаётся `transverse_length`, CRL становится цилиндрически фокусирующей:

```python
crl = xpos.CRL(
    input_field=E0,
    grid=grid2d,
    radius=6.25e-6,
    aperture=50e-6,
    bridge_thickness=2e-6,
    n_lenses=26,
    material=xpos.Material("Si", density=2.33),
    transverse_length=33e-6,
    rotation=(0.0, 0.0, 0.0),
    energy_kev=12.0,
)
```

Локальная система такой CRL:

```text
local x → фокусирующее направление
local y → конечная поперечная ширина
```

То есть направление цилиндрической CRL в мировой `x/y` плоскости задаётся `rz`.

### Фокусировка вдоль мировой `x`

```python
rotation=(0.0, 0.0, 0.0)
```

### Поворот на 90°

```python
rotation=(0.0, 0.0, np.pi / 2)
```

Текущая параксиальная поперечная модель реализует только `rz`. Ненулевые `rx` или `ry` специально отклоняются через `NotImplementedError`, чтобы не имитировать полноценный наклон элемента некорректной моделью. fileciteturn7file0L971-L1004 fileciteturn8file0L62-L70

---

# 16. Толщина и transmission CRL

Толщину можно посмотреть напрямую:

```python
T = crl.thickness()
```

Для конкретной линзы при индивидуальных дефектах:

```python
T0 = crl.thickness(lens_index=0)
T5 = crl.thickness(lens_index=5)
```

Комплексная transmission-функция:

```python
transmission = crl.transmission()
```

Внутренне используется:

```text
exp[-i k (delta - i beta) T]
```

а propagation между центрами соседних линз использует CRL pitch.

Pitch между центрами соседних элементов:

```text
p = d + A² / (4R)
```

В API:

```python
crl.lens_pitch
```

fileciteturn8file0L77-L80 fileciteturn7file1L1050-L1058

---

# 17. Кэширование геометрии CRL

По умолчанию:

```python
reuse_lens_geometry=True
```

Это означает, что если все элементы CRL одинаковые, вычисленная карта толщины и transmission может переиспользоваться.

Для явного отключения:

```python
crl = xpos.CRL(
    ...,
    reuse_lens_geometry=False,
)
```

Сбросить кэш:

```python
crl.clear_cache()
```

Это особенно важно для больших 2D карт: построение геометрии может быть существенно дороже, чем повторное применение уже рассчитанной transmission map. Логика кэша реализована отдельно для thickness и transmission. fileciteturn8file0L199-L255

---

# 18. Максимальное число линз

У CRL есть физическая граница применимости параксиального внешнего фокуса.

Доступны:

```python
crl.characteristic_length
```

```python
crl.characteristic_lens_count
```

```python
crl.critical_lens_count()
```

```python
crl.max_lens_count()
```

В текущей модели:

```text
Lc = sqrt(p R / (2 delta))

N_critical = (pi/2) * Lc / p
```

`critical_lens_count()` возвращает вещественное критическое значение, а `max_lens_count()` — последний целый допустимый счётчик, при котором внешний фокус ещё не находится внутри CRL. fileciteturn7file1L1060-L1068

### Важное поведение при создании CRL

Если:

```python
n_lenses > crl.max_lens_count()
```

XPOSim выдаёт `RuntimeWarning` уже при инициализации.

Например:

```python
crl = xpos.CRL(
    input_field=E0,
    grid=grid,
    radius=R,
    aperture=A,
    bridge_thickness=d,
    n_lenses=1000,
    material=material,
    energy_kev=12.0,
)
```

создание объекта не скрывает физически сомнительный режим: предупреждение сообщает, что формальный фокус лежит внутри CRL и внешний focal-plane analysis уже не имеет физического смысла. fileciteturn8file0L85-L95

---

# 19. Фокусные величины CRL

## Теоретический фокус

```python
focus = crl.theoretical_focus()
```

Для другого числа линз:

```python
focus = crl.theoretical_focus(n_lenses=20)
```

Метод предупреждает, если указанное число линз находится на критической границе или за ней. fileciteturn8file0L267-L282

## Эффективный фокус конечного источника

Для конечного расстояния до объекта:

```python
image_distance = crl.effective_focus(
    source_distance=0.5,
)
```

Для аналитической проверки другого числа линз:

```python
image_distance = crl.effective_focus(
    source_distance=0.5,
    n_lenses=10,
)
```

Метод возвращает расстояние от выходной reference plane CRL до изображения. fileciteturn8file0L284-L311

## Масштаб изображения

```python
scale = crl.image_scale(
    source_distance=0.5,
)
```

## Аналитическое увеличение

```python
m = crl.magnification(
    source_distance=0.5,
    image_distance=0.2,
)
```

## Проверка положения фокуса

```python
is_valid = crl.is_focus_outside_crl()
```

или для другого числа линз:

```python
is_valid = crl.is_focus_outside_crl(n_lenses=20)
```

---

# 20. Дефекты CRL

В новой архитектуре дефекты не являются отдельными наследниками `CRL`.

Вместо этого дефект получает:

```python
ideal_thickness, grid, lens_index
```

и возвращает изменённую карту толщины той же формы и backend. fileciteturn7file0L920-L951 fileciteturn7file1L1070-L1078

## Простейший пользовательский дефект

```python
def extra_center_thickness(thickness, grid, lens_index):
    xp = grid.xp
    defect = 0.1e-6 * xp.exp(-(grid.x / 2e-6) ** 2 / 2)
    return thickness + defect
```

Передача в CRL:

```python
crl = xpos.CRL(
    input_field=E0,
    grid=grid,
    radius=6.25e-6,
    aperture=50e-6,
    bridge_thickness=2e-6,
    n_lenses=20,
    material=xpos.Material("Si", density=2.33),
    thickness_defect=extra_center_thickness,
    energy_kev=12.0,
)
```

Функция получает номер линзы, поэтому можно использовать его для детерминированного отличия линз:

```python
def lens_dependent_defect(thickness, grid, lens_index):
    factor = 1.0 + 0.01 * lens_index
    return thickness * factor
```

---

# 21. Разные дефекты для каждой линзы

Можно передать последовательность дефектов длиной `n_lenses`:

```python
defect_a = lambda thickness, grid, i: thickness

def defect_b(thickness, grid, i):
    return thickness + 5e-9

defects = [
    defect_a,
    defect_b,
    defect_a,
    defect_b,
]

crl = xpos.CRL(
    input_field=E0,
    grid=grid,
    radius=6.25e-6,
    aperture=50e-6,
    bridge_thickness=2e-6,
    n_lenses=4,
    material=xpos.Material("Si", density=2.33),
    thickness_defect=defects,
    energy_kev=12.0,
)
```

Если передана последовательность индивидуальных дефектов, переиспользование одной геометрии автоматически отключается. fileciteturn8file0L97-L124

Для сложных моделей можно создать собственный класс-наследник `ThicknessDefect`.

---

# 22. Анализ пучка

Анализ отделён от CRL и работает с любым подходящим полем.

## Интенсивность

```python
I = xpos.intensity(E)
```

или:

```python
I = obj.I(z=...) 
```

Функция `intensity()` не меняет backend массива. fileciteturn8file0L472-L476

---

# 23. FWHM

Для 1D профиля:

```python
width = xpos.full_width_half_max(
    intensity_profile,
    grid.x,
)
```

Алгоритм не предполагает строго гауссовский профиль. Он ищет пересечения с половиной максимума и интерполирует их линейно. Если валидная пара пересечений отсутствует, возвращается `None`. fileciteturn8file0L487-L528

---

# 24. BeamMetrics

Для общего анализа:

```python
metrics = xpos.beam_metrics(
    field,
    grid,
)
```

Результат:

```python
metrics.max_intensity
metrics.center
metrics.fwhm
metrics.divergence
```

Для 1D:

```text
center   → float
fwhm     → float | None
divergence → float | None
```

Для 2D:

```text
center   → (x, y)
fwhm     → (fwhm_x, fwhm_y)
divergence → (div_x, div_y) | None
```

Можно ограничить анализ ROI:

```python
metrics = xpos.beam_metrics(
    field,
    grid,
    roi=(( -5e-6, 5e-6), (-5e-6, 5e-6)),
)
```

Для 1D:

```python
metrics = xpos.beam_metrics(
    field,
    grid,
    roi=(-5e-6, 5e-6),
)
```

ROI задаётся в физических координатах, а не в индексах массива. fileciteturn8file0L559-L616

---

# 25. Дивергенция

Можно запросить дальнеполевую угловую FWHM:

```python
div = xpos.divergence(
    field,
    grid,
    k=crl.k,
)
```

или сразу через:

```python
metrics = xpos.beam_metrics(
    field,
    grid,
    calculate_divergence=True,
    k=crl.k,
)
```

Без `k` расчёт divergence не выполняется.

Если корректная область углового спектра недостаточна для определения FWHM, функция возвращает `None`, а не пытается выдать недостоверное число. fileciteturn8file0L619-L651

---

# 26. Focus scan

`CRL.focus_scan()` — это основной инструмент для численного поиска фокуса.

Главная идея:

```text
             ┌─ propagate(z1)
E_exit ──────┼─ propagate(z2)
             ├─ propagate(z3)
             └─ ...
```

Сама CRL вычисляется один раз. Затем выходное поле распространяется к нескольким плоскостям `z`. Если `return_fields=False`, библиотека не накапливает весь стек изображений. Можно задать ROI, если не нужна полная сетка. fileciteturn8file0L357-L413

Пример:

```python
z_values = np.linspace(
    9e-3,
    11e-3,
    101,
)

scan = crl.focus_scan(
    z_values,
    roi=(-5e-6, 5e-6),
    return_fields=False,
)
```

Для 2D:

```python
scan = crl.focus_scan(
    z_values,
    roi=(
        (-5e-6, 5e-6),
        (-5e-6, 5e-6),
    ),
)
```

Получить найденную плоскость максимума:

```python
print(scan.peak_z)
```

FWHM в найденной плоскости:

```python
print(scan.fwhm_at_peak)
```

Divergence:

```python
print(scan.divergence_at_peak)
```

Если формальный фокус CRL расположен внутри самой CRL, `focus_scan()` не запускается и возвращает invalid-result с предупреждением. fileciteturn8file0L372-L378

---

# 27. Сохранение ROI изображений

Если нужно сохранить сами карты:

```python
scan = crl.focus_scan(
    z_values,
    roi=(
        (-5e-6, 5e-6),
        (-5e-6, 5e-6),
    ),
    return_fields=True,
)
```

Тогда:

```python
scan.fields
```

содержит ROI-поля. Для уменьшения зависимости результатов от backend эти сохранённые поля переводятся в NumPy. Метрики при этом вычисляются в исходном backend. fileciteturn8file0L389-L413

---

# 28. Полная простая 2D схема

Ниже минимальный законченный пример:

```python
import numpy as np
import cupy as cp
import matplotlib.pyplot as plt
import XPOSim as xpos

ENERGY = 12.0

# 2D grid
grid = xpos.Grid2D(
    dx=5e-8,
    dy=5e-8,
    nx=1024,
    ny=1024,
    backend="cupy",
)

# point source
source = xpos.PointSource(
    grid=grid,
    position=(0.0, 0.0),
    energy_kev=ENERGY,
)

# field in the first optical plane
E0 = source.E(z=15.0)

# aperture
hole = xpos.Hole(
    input_field=E0,
    grid=grid,
    center_offset=(10e-6, 0.0),
    radius=3e-6,
    energy_kev=ENERGY,
)

E1 = hole.E(z=50e-6)

# CRL
material = xpos.Material(
    formula="Si",
    density=2.33,
)

crl = xpos.CRL(
    input_field=E1,
    grid=grid,
    radius=6.25e-6,
    aperture=50e-6,
    bridge_thickness=2e-6,
    n_lenses=20,
    material=material,
    energy_kev=ENERGY,
)

print("maximum valid lens count:", crl.max_lens_count())
print("theoretical focus:", crl.theoretical_focus())

# numerical field after CRL
E2 = crl.E()

# focus scan
z_values = np.linspace(
    1e-3,
    50e-3,
    100,
)

scan = crl.focus_scan(
    z_values,
    roi=(
        (-10e-6, 10e-6),
        (-10e-6, 10e-6),
    ),
)

print("numerical peak z:", scan.peak_z)
print("FWHM at peak:", scan.fwhm_at_peak)

# inspect one selected plane
E_focus = crl.E(z=scan.peak_z)
I_focus = xpos.intensity(E_focus)

plt.figure(figsize=(7, 6))
plt.pcolormesh(
    cp.asnumpy(grid.x_axis) * 1e6,
    cp.asnumpy(grid.y_axis) * 1e6,
    cp.asnumpy(I_focus),
    shading="auto",
)
plt.xlabel("x, µm")
plt.ylabel("y, µm")
plt.colorbar(label="Intensity")
plt.tight_layout()
plt.show()
```

---

# 29. Полная простая 1D схема

```python
import numpy as np
import matplotlib.pyplot as plt
import XPOSim as xpos

ENERGY = 12.0

grid = xpos.Grid1D(
    dx=5e-8,
    nx=4096,
    backend="numpy",
)

source = xpos.PointSource(
    grid=grid,
    position=0.0,
    energy_kev=ENERGY,
)

E0 = source.E(z=0.1)

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

print("Nmax =", crl.max_lens_count())
print("focus =", crl.theoretical_focus())

z_values = np.linspace(1e-3, 50e-3, 200)
scan = crl.focus_scan(
    z_values,
    roi=(-10e-6, 10e-6),
)

print("numerical focus =", scan.peak_z)
print("FWHM =", scan.fwhm_at_peak)

E_focus = crl.E(z=scan.peak_z)
I_focus = xpos.intensity(E_focus)

plt.plot(
    grid.x * 1e6,
    I_focus,
)
plt.xlabel("x, µm")
plt.ylabel("Intensity")
plt.show()
```

---

# 30. Двухкратная CRL-схема с поворотом на 90°

Это соответствует случаю, когда две одноосевые CRL последовательно фокусируют в ортогональных направлениях.

Первая CRL:

```python
crl_y = xpos.CRL(
    input_field=E0,
    grid=grid,
    radius=R,
    aperture=A,
    bridge_thickness=d,
    n_lenses=N1,
    material=material,
    transverse_length=LT,
    rotation=(0.0, 0.0, np.pi / 2),
    energy_kev=ENERGY,
)
```

После распространения:

```python
E_before_x = crl_y.E(z=Z1)
```

Вторая CRL:

```python
crl_x = xpos.CRL(
    input_field=E_before_x,
    grid=grid,
    radius=R,
    aperture=A,
    bridge_thickness=d,
    n_lenses=N2,
    material=material,
    transverse_length=LT,
    rotation=(0.0, 0.0, 0.0),
    energy_kev=ENERGY,
)
```

Здесь обе CRL принадлежат одной 2D сетке; отличие только в локальной ориентации цилиндрической геометрии. Формально `rz` поворачивает локальную transverse-систему внутри мировой `x/y` плоскости. fileciteturn8file0L162-L197

---

# 31. NumPy ↔ CuPy

Одно из важных требований архитектуры XPOSim: backend поля не меняется автоматически.

Если:

```text
E0 — CuPy array
```

то:

```python
E1 = element.E()
```

тоже будет CuPy array.

Аналогично для NumPy.

Проверка:

```python
print(type(E0))
print(type(E1))
```

Grid также не позволит случайно подать поле другого backend или неправильной формы: это приводит к `TypeError` или `ValueError`. fileciteturn7file0L499-L529

---

# 32. Частые ошибки

## Неправильная размерность

```python
Grid1D(...)
```

нельзя использовать с 2D полем и наоборот.

## Неправильный backend

```python
grid = xpos.Grid2D(..., backend="cupy")
E = np.zeros(grid.shape, dtype=complex)
```

не будет принят элементом этой сетки.

Нужно либо создать NumPy grid, либо перенести поле на CuPy.

## Нет `energy_kev` / `wavelength_m`

Каждый оптический объект должен знать длину волны. Поэтому:

```python
xpos.CRL(...)
```

без `energy_kev` или `wavelength_m` вызовет ошибку, если прямые `delta/beta` не заданы через `Material` и не обеспечивают этот контекст. Общий `OpticalObject` требует ровно одну из этих величин. fileciteturn7file0L690-L715

## Точечный источник на `z=0`

Не допускается:

```python
source.E(z=0)
```

Источник требует `z > 0`. fileciteturn7file0L815-L820

## Слишком много линз

Проверяйте:

```python
crl.max_lens_count()
```

Если число линз больше допустимого, при создании CRL выдаётся `RuntimeWarning`. Для аналитического `effective_focus()` превышение приводит к `ValueError`. fileciteturn8file0L145-L160 fileciteturn8file0L284-L311

---

# 33. Рекомендованный стиль notebook

Хорошо разделять notebook на четыре секции:

```text
1. Parameters
2. Grid + source
3. Optical scheme
4. Analysis + plotting
```

Например:

```python
# ==========================
# PARAMETERS
# ==========================
ENERGY = 12.0
DX = 5e-8
DY = 5e-8
NX = 2048
NY = 2048

# ==========================
# GRID / SOURCE
# ==========================
...

# ==========================
# OPTICAL SCHEME
# ==========================
...

# ==========================
# ANALYSIS
# ==========================
...
```

Такой стиль хорошо соответствует архитектуре XPOSim: геометрия схемы строится последовательно через массивы `E`, а диагностика выполняется отдельно.

---

# 34. Рекомендуемый workflow для реального расчёта

1. Выберите физическую размерность: `Grid1D` или `Grid2D`.
2. Выберите пространственный шаг так, чтобы поле и все существенные геометрические структуры были разрешены сеткой.
3. Выберите `numpy` для отладки маленьких задач и `cupy` для тяжёлых расчётов.
4. Создайте `PointSource` и получите поле в плоскости первого элемента.
5. Стройте схему последовательно, передавая `E` от одного объекта к следующему.
6. Для CRL сначала проверьте `max_lens_count()`.
7. Для сомнительных случаев проверьте `is_focus_outside_crl()`.
8. Для численного поиска фокуса используйте `focus_scan()` вместо повторного полного пересчёта CRL в каждой плоскости.
9. Для больших 2D расчётов используйте ROI и не сохраняйте полный стек фокусировки без необходимости.
10. Для сравнения аналитики и численного решения сначала сравните `theoretical_focus()`/`effective_focus()` с `focus_scan().peak_z`.

---

# 35. Что не моделирует текущая версия

Текущая архитектура намеренно остаётся параксиальной.

Не следует рассматривать XPOSim 0.3.0 как универсальный электромагнитный solver или как полноценную non-paraxial трассировку толстых оптических элементов.

В текущей реализации также нет полноценного моделирования наклонённого элемента через `rx`/`ry`; ненулевые значения отклоняются явно. Это лучше, чем молча использовать некорректную геометрию. fileciteturn7file0L975-L1004

---

# 36. Краткая справка API

## Grid

```python
xpos.Grid1D(dx, nx, backend="cupy")
xpos.Grid2D(dx, dy, nx, ny, backend="cupy")
```

## Source

```python
xpos.PointSource(
    grid,
    position=...,
    wavelength_m=...,
    energy_kev=...,
)
```

## Elements

```python
xpos.Hole(
    input_field,
    grid,
    center_offset=...,
    radius=...,
    wavelength_m=...,
    energy_kev=...,
)

xpos.Blade(
    input_field,
    grid,
    wavelength_m=...,
    energy_kev=...,
)

xpos.CRL(
    input_field,
    grid,
    radius=...,
    aperture=...,
    bridge_thickness=...,
    n_lenses=...,
    material=...,
    transverse_length=...,
    rotation=(rx, ry, rz),
    thickness_defect=...,
    reuse_lens_geometry=True,
    wavelength_m=...,
    energy_kev=...,
)
```

## CRL methods

```python
crl.thickness()
crl.transmission()
crl.lens_pitch
crl.characteristic_length
crl.characteristic_lens_count
crl.critical_lens_count()
crl.max_lens_count()
crl.theoretical_focus()
crl.effective_focus(source_distance=...)
crl.magnification(source_distance=..., image_distance=...)
crl.image_scale(source_distance=...)
crl.is_focus_outside_crl()
crl.focus_scan(z_values, roi=..., return_fields=False)
crl.clear_cache()
crl.E(z=...)
crl.I(z=...)
```

## Analysis

```python
xpos.intensity(field)
xpos.full_width_half_max(intensity_array, coordinate)
xpos.beam_metrics(field, grid, roi=..., calculate_divergence=False, k=...)
xpos.divergence(field, grid, k)
xpos.gaussian_filter_1d(field, coordinate, sigma)
xpos.gaussian_filter_2d(field, x, y, sigma_x, sigma_y)
```

---

# 37. Minimal mental model

Если нужно запомнить только пять вещей, достаточно помнить:

```text
Grid1D / Grid2D
        ↓
PointSource.E(z)
        ↓
E array
        ↓
Element(input_field=E)
        ↓
Element.E(z)
```

Для CRL дополнительно:

```text
CRL
 ├── thickness()
 ├── transmission()
 ├── theoretical_focus()
 ├── effective_focus()
 ├── max_lens_count()
 └── focus_scan()
```

Это и есть основной пользовательский интерфейс XPOSim 0.3.0.
