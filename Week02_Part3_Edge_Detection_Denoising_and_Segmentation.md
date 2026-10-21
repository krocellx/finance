# Week 2 (Part 3): Edge Detection, Image Denoising & Image Segmentation

**Sources Covered (100% No-Skip)**:
- `02-week02/01-MLDS_UDA_2026-27_ImageNotes.pdf` (Sections 4, 5, & 6, pp. 22–55)
- `02-week02/05-MLDS_UDA_Slides_EdgeDetectionImages.pdf` (Slides 1–26)
- `02-week02/06-MLDS_UDA_Slides_DenoisingImages.pdf` (Slides 1–17)
- `02-week02/07-MLDS_UDA_Slides_SegmentationImages.pdf` (Slides 1–13)

---

## PART A: EDGE DETECTION (Section 4 & Edge Detection Slides)

### 1. What is an Edge & How Derivatives Detect It (Sections 4.1 & 4.2)
An **edge** is a local discontinuity in gray-level intensity. There are **three principal physical edge types**:
1. **Step Edges (Most Common)**: Occur where two regions of different gray levels meet.
2. **Line Edges (Roof/Ridge Edges)**: Thin bright/dark objects against a background (e.g., roads and rivers in satellite images).
3. **Junctions (Corners)**: Where at least two physical edges meet.

#### Concrete 1D Numerical Example: How 1st and 2nd Derivatives Pinpoint a Step Edge!
Suppose we scan across a 1D row of 8 pixels where a dark object (`10`) transitions to a bright background (`110`):

| Quantity | Formula | Pixel Values Across the Row | What Happens at the Edge? |
| :--- | :--- | :--- | :--- |
| **Image Intensity $I(x)$** | Original row | `10,  10,  10,  60, 110, 110, 110` | Ramps up from `10` to `110` across `60`. |
| **1st Derivative $I'(x)$** | $I(x+1) - I(x)$ | `  0,   0, +50, +50,   0,   0` | **Local Maximum (`+50`)** right on the edge! |
| **2nd Derivative $-I''(x)$** | $-[I'(x) - I'(x-1)]$ | `     0, -50,   0, +50,   0` | Negative (`-50`) on dark side, positive (`+50`) on bright side $\implies$ **Zero-Crossing (`0`)** right at the inflection point `60`! |

---

### 2. First-Order Gradient & Compass Operators (Section 4.2) — Concrete Worked Examples

To see how **every single edge detector in Section 4.2** works, let's test them all on the **exact same $3 \times 3$ image patch $I$** containing a vertical step edge (dark column `10` on the left, bright columns `110` in the middle and right), centered at pixel $(2, 2)$:
$$I = \begin{bmatrix} 10 & 110 & 110 \\ 10 & \mathbf{110} & 110 \\ 10 & 110 & 110 \end{bmatrix}$$

#### 2.1 Roberts Cross Operator on Patch $I$ at Center $(2, 2)$
- **Formulas**:
  $$S_x(2, 2) = I(2, 2) - I(1, 1) = 110 - 10 = \mathbf{100}$$
  $$S_y(2, 2) = I(2, 1) - I(1, 2) = 10 - 110 = \mathbf{-100}$$
- **Form 1 ($L_2$ Magnitude)**: $\sqrt{100^2 + (-100)^2} = \sqrt{20000} \approx \mathbf{141.42}$
- **Form 2 ($L_1$ Magnitude — faster)**: $|100| + |-100| = \mathbf{200}$

#### 2.2 Prewitt vs. Sobel Operators on Patch $I$ at Center $(2, 2)$ (Definitions 4.1–4.3)

| Operator | Vertical Edge Mask $s_1$ (Horizontal Gradient) | Horizontal Edge Mask $s_2$ (Vertical Gradient) | Calculation on $I = \begin{bmatrix} 10 & 110 & 110 \\ 10 & 110 & 110 \\ 10 & 110 & 110 \end{bmatrix}$ | Magnitude $\sqrt{s_1^2+s_2^2}$ & Direction $\tan^{-1}(s_1/s_2)$ |
| :--- | :---: | :---: | :--- | :--- |
| **Prewitt** | $\begin{bmatrix} -1 & 0 & 1 \\ -1 & 0 & 1 \\ -1 & 0 & 1 \end{bmatrix}$ | $\begin{bmatrix} -1 & -1 & -1 \\ 0 & 0 & 0 \\ 1 & 1 & 1 \end{bmatrix}$ | • $p_2 = (110-10) + (110-10) + (110-10) = \mathbf{300}$<br/>• $p_1 = (10+110+110) - (10+110+110) = \mathbf{0}$ | • Magnitude = $\sqrt{300^2 + 0^2} = \mathbf{300}$<br/>• Direction = $\tan^{-1}(0/300) = \mathbf{0^\circ}$ *(gradient points horizontally across the vertical edge!)* |
| **Sobel** | $\begin{bmatrix} -1 & 0 & 1 \\ -2 & 0 & 2 \\ -1 & 0 & 1 \end{bmatrix}$ | $\begin{bmatrix} -1 & -2 & -1 \\ 0 & 0 & 0 \\ 1 & 2 & 1 \end{bmatrix}$ | • $s_2 = 1(110-10) + 2(110-10) + 1(110-10) = \mathbf{400}$<br/>• $s_1 = (10+220+110) - (10+220+110) = \mathbf{0}$ | • Magnitude = $\sqrt{400^2 + 0^2} = \mathbf{400}$<br/>• Direction = $\tan^{-1}(0/400) = \mathbf{0^\circ}$ |

#### 2.3 Kirsch & Robinson Compass Masks on Patch $I$ (Definitions 4.4–4.7, pp. 25–26)
- **Kirsch Masks ($k_0, \dots, k_7$)**: Convolve $I$ with $k_0 = \begin{bmatrix} -3 & -3 & 5 \\ -3 & 0 & 5 \\ -3 & -3 & 5 \end{bmatrix}$ vs. $k_2 = \begin{bmatrix} 5 & 5 & 5 \\ -3 & 0 & -3 \\ -3 & -3 & -3 \end{bmatrix}$:
  - Response of $k_0$: $(-3)(10+10+10) + (-3)(110+110) + 5(110+110+110) = -90 - 660 + 1650 = \mathbf{+900}$.
  - Response of $k_2$ (horizontal edge mask): $5(10+110+110) - 3(10+110) - 3(10+110+110) = 1150 - 360 - 690 = \mathbf{+100}$.
  - Taking the maximum across all 8 masks gives **Edge Magnitude $= 900$** and identifies the edge direction via the winning mask!
- **Robinson Masks ($r_0, \dots, r_7$)**: Since $r_0 = \begin{bmatrix} -1 & 0 & 1 \\ -2 & 0 & 2 \\ -1 & 0 & 1 \end{bmatrix}$ is identical to the Sobel mask and $r_4 = -r_0$, convolving $r_0$ with $I$ gives $+400$ (and $r_4$ gives $-400$), which is why we only need to compute $r_0, r_1, r_2, r_3$ and negate them!

---

### 3. Second-Order Laplacian & LoG Operators (Section 4.3) — Concrete Worked Example

#### 3.1 How the Discrete 4-Neighbor Laplacian Mask Creates a Zero-Crossing Across a Step Edge!
Consider a $3 \times 4$ image strip where columns 1–2 are dark (`10`) and columns 3–4 are bright (`110`):
$$\text{Image Strip } F = \begin{bmatrix} 10 & \mathbf{10} & \mathbf{110} & 110 \\ 10 & \mathbf{10} & \mathbf{110} & 110 \\ 10 & \mathbf{10} & \mathbf{110} & 110 \end{bmatrix}, \qquad \text{4-Neighbor Laplacian Mask } \mathbf{H} = \begin{bmatrix} 0 & -1 & 0 \\ -1 & 4 & -1 \\ 0 & -1 & 0 \end{bmatrix}$$
Let's convolve $\mathbf{H}$ at the **dark-side edge pixel $(2, 2)$** (value `10`) and the **bright-side edge pixel $(2, 3)$** (value `110`):
1. **At Dark-Side Pixel $(2, 2)$** ($3 \times 3$ patch is $\begin{bmatrix} 10 & 10 & 110 \\ 10 & \mathbf{10} & 110 \\ 10 & 10 & 110 \end{bmatrix}$):
   $$G(2, 2) = 4(10) - 1(10) - 1(10) - 1(10) - 1(110) = 40 - 30 - 110 = \mathbf{-100 \text{ (Negative!)}}$$
2. **At Bright-Side Pixel $(2, 3)$** ($3 \times 3$ patch is $\begin{bmatrix} 10 & 110 & 110 \\ 10 & \mathbf{110} & 110 \\ 10 & 110 & 110 \end{bmatrix}$):
   $$G(2, 3) = 4(110) - 1(110) - 1(110) - 1(10) - 1(110) = 440 - 340 = \mathbf{+100 \text{ (Positive!)}}$$
3. **Result**: The Laplacian output jumps from **$-100$ to $+100$** right between column 2 and column 3! That **sign change (zero-crossing)** localizes the exact boundary between the dark and bright regions!

#### 3.2 Marr–Hildreth LoG Mask Size Rule ($n \ge 6\sigma$) — Concrete Example
- **Formula**: $H(x, y) = \frac{1}{\pi \sigma^4}\left(1 - \frac{x^2 + y^2}{2\sigma^2}\right)\exp\left\{-\frac{x^2 + y^2}{2\sigma^2}\right\}$.
- **Concrete Example of Choosing Mask Size $n$**:
  Suppose you use a Gaussian smoothing standard deviation of $\sigma = 1.4$ pixels.
  - Compute $6\sigma = 6 \times 1.4 = 8.4$.
  - Choose $n$ to be the **smallest odd integer $\ge 8.4$**, which is **$n = 9$** (a $9 \times 9$ LoG mask)!

---

### 4. Advanced Edge Detectors & Pratt's Figure of Merit (Sections 4.4 & 4.5)

#### 4.1 Concrete Example of Canny's Step 3 (Nonmaxima Suppression) & Step 4 (Hysteresis)
Suppose after Step 2 of Canny, we have a vertical edge (so the gradient direction is **horizontal**, left-to-right $\leftrightarrow$) with the following $3 \times 3$ gradient magnitudes:
$$\text{Raw Gradient Magnitudes: } \begin{bmatrix} 20 & \mathbf{85} & 30 \\ 25 & \mathbf{90} & 40 \\ 15 & \mathbf{55} & 10 \end{bmatrix}$$
1. **Step 3 (Nonmaxima Suppression along the horizontal gradient direction $\leftrightarrow$)**:
   - In Row 1: Center `85` is compared to left `20` and right `30`. Since $85 > 20$ and $85 > 30$, keep **`85`**, and set `20` and `30` to **`0`**!
   - In Row 2: Keep **`90`**, set `25` and `40` to **`0`**.
   - In Row 3: Keep **`55`**, set `15` and `10` to **`0`**.
   - **Thinned 1-Pixel Edge Ridge**: $\begin{bmatrix} 0 & \mathbf{85} & 0 \\ 0 & \mathbf{90} & 0 \\ 0 & \mathbf{55} & 0 \end{bmatrix}$.
2. **Step 4 (Hysteresis Thresholding with $T_{\text{high}} = 80$ and $T_{\text{low}} = 50$)**:
   - Top two pixels (`85` and `90`) are $> T_{\text{high}} = 80 \implies$ marked immediately as **Strong Edges**.
   - Bottom pixel (`55`) is between $T_{\text{low}} = 50$ and $T_{\text{high}} = 80$ (**Weak Edge**). If we used a single threshold of $80$, this edge pixel would be lost! Because `55` touches the strong edge pixel `90` at $(2,2)$, hysteresis thresholding **rescues `55`** and keeps the entire 3-pixel vertical line intact!

#### 4.2 Concrete Numerical Calculation of Pratt's Figure of Merit ($\text{FOM}_P$, Definition 4.9)
$$\text{FOM}_P = \frac{1}{\max(I_I, I_F)} \sum_{i=1}^{I_F} \frac{1}{1 + \alpha d_i^2}$$
- **Concrete Example**: Suppose an ideal edge has $I_I = 4$ pixels, and our detector finds $I_F = 4$ pixels (so $I_N = \max(4, 4) = 4$) with $\alpha = \frac{1}{9}$:
  - 2 detected pixels land on the exact ideal edge ($d_1 = 0, \, d_2 = 0$).
  - 2 detected pixels are offset by 1 pixel ($d_3 = 1, \, d_4 = 1$).
  - **Compute $\text{FOM}_P$**:
    $$\text{FOM}_P = \frac{1}{4}\left( \frac{1}{1 + \frac{1}{9}(0)^2} + \frac{1}{1 + \frac{1}{9}(0)^2} + \frac{1}{1 + \frac{1}{9}(1)^2} + \frac{1}{1 + \frac{1}{9}(1)^2} \right) = \frac{1}{4}\left(1 + 1 + 0.9 + 0.9\right) = \mathbf{0.95}$$

---

## PART B: IMAGE DENOISING (Section 5 & Denoising Slides)

### 1. Concrete Numerical Example of Signal-to-Noise Ratio (SNR, Section 5, p. 35)
$$\text{SNR} = 10 \log_{10}\left(\frac{V_s^2}{V_n^2}\right) \text{ dB}$$
- **Example**: If the clean signal voltage is $V_s = 100$ and the noise voltage is $V_n = 10$, then $\frac{V_s^2}{V_n^2} = \frac{10000}{100} = 100$, so:
  $$\text{SNR} = 10 \log_{10}(100) = 10 \times 2 = \mathbf{20\text{ dB}}$$
  If a denoising filter cuts the noise voltage in half to $V_n = 5$, then $\frac{V_s^2}{V_n^2} = \frac{10000}{25} = 400 \implies \text{SNR} = 10\log_{10}(400) \approx \mathbf{26.02\text{ dB}}$ (a $+6\text{ dB}$ improvement!).

---

### 2. Concrete Comparison of ALL Spatial Denoising Filters on the Exact Same $3 \times 3$ Patch! (Section 5.2.1)

Suppose a flat image region has a true gray value of **`100`**, and the center pixel is hit by a bright **Salt Noise spike (`250`)** while two bottom pixels have tiny Gaussian fluctuations (`98` and `102`):
$$W = \begin{bmatrix} 100 & 100 & 100 \\ 100 & \mathbf{250} & 100 \\ 100 & 102 & 98 \end{bmatrix} \quad (\text{True clean value at center } = \mathbf{100})$$

Let's compute the exact output $f_e(x, y)$ of **every single filter in Section 5.2.1** on this $3 \times 3$ window ($mn = 9$ pixels):

| Filter Name | Formula | Step-by-Step Arithmetic on Window $W$ | Output $f_e$ | How Close to True `100`? |
| :--- | :--- | :--- | :---: | :--- |
| **1. Arithmetic Mean** | $\frac{1}{9}\sum g(p,q)$ | $\frac{100+100+100+100+250+100+100+102+98}{9} = \frac{1050}{9}$ | **$116.67$** | Pulled up $+16.67$ by the `250` spike! |
| **2. Geometric Mean** | $\left(\prod g(p,q)\right)^{1/9}$ | $(100^6 \times 250 \times 102 \times 98)^{1/9}$ | **$110.67$** | Less distorted than arithmetic mean. |
| **3. Harmonic Mean** | $\frac{9}{\sum \frac{1}{g(p,q)}}$ | $\frac{9}{6(0.01) + \frac{1}{250} + \frac{1}{102} + \frac{1}{98}} = \frac{9}{0.084008}$ | **$107.13$** | Suppresses the bright `250` salt spike much better! |
| **4. Inverse Harmonic ($k = -1$)** | $\frac{\sum g^0}{\sum g^{-1}}$ | Reduces to Harmonic Mean ($k < 0$ filters **salt** noise!) | **$107.13$** | Good salt suppression. |
| **5. Inverse Harmonic ($k = +1$)** | $\frac{\sum g^2}{\sum g^1}$ | $\frac{6(100^2) + 250^2 + 102^2 + 98^2}{1050} = \frac{142508}{1050}$ | **$135.72$** | Bad! Because $k > 0$ is for **pepper** noise, it amplifies salt! |
| **6. Median Filter** | Middle (5th) sorted value | Sorted: `[98, 100, 100, 100, `**`100`**`, 100, 100, 102, 250]` | **$100.00$** | **PERFECT!** Completely eliminates the `250` spike! |
| **7. Minimum Filter** | Smallest value in $W$ | $\min(98, 100, \dots, 250)$ | **$98.00$** | Removes salt spike (`250`). |
| **8. Maximum Filter** | Largest value in $W$ | $\max(98, 100, \dots, 250)$ | **$250.00$** | Spreads salt spike (only use Max for pepper `0`!). |
| **9. Mid-point Filter** | $\frac{1}{2}(\max_W g + \min_W g)$ | $\frac{1}{2}(250 + 98)$ | **$174.00$** | Ruined by impulse outlier (only use for Gaussian/Uniform!). |
| **10. Mode Filter** | Most frequent value in $W$ | Value `100` appears **6 times** out of 9! | **$100.00$** | **PERFECT!** Restores `100` and sharpens edges! |

---

### 3. Concrete Walkthrough of the Truncated-Median Mode Filter (Section 5.2.1, pp. 41–42)
Suppose a $3 \times 3$ window near an edge has 9 sorted pixel values:
$$\text{Initial values: } \texttt{[10,  18,  20,  20,  22,  25,  30,  80,  90]}$$
- Notice the true mode (cluster peak) is at **`20`**, while two pixels (`80, 90`) lie on the bright side of the edge (creating a long right tail).
1. **Step 1**: Find the median (5th element): $\text{med} = \mathbf{22}$.
   - Distance to minimum (`10`): $22 - 10 = \mathbf{12}$ (shorter left tail).
   - Distance to maximum (`90`): $90 - 22 = \mathbf{68}$ (longer right tail).
2. **Step 2 (Truncate the longer right tail to match length `12`)**:
   - Keep only values $\le \text{med} + 12 = 22 + 12 = 34$:
     $$\text{Remaining values: } \texttt{[10,  18,  20,  20,  22,  25,  30]}$$
3. **Step 3 (Recompute the median of the remaining 7 values — 4th element)**:
   - New median = **`20`** (which is the exact **mode** of the distribution!).

---

### 4. Concrete Walkthrough of the Selective Filter Criteria (Section 5.2.2, pp. 43–44)
Let's test if the center pixel $f(x, y) = \mathbf{250}$ in our $3 \times 3$ window $W = \begin{bmatrix} 100 & 100 & 100 \\ 100 & \mathbf{250} & 100 \\ 100 & 102 & 98 \end{bmatrix}$ is detected as Salt-and-Pepper noise using:
- Image range $[L_{\min}, L_{\max}] = [0, 255]$, grayscale threshold $T_g = 15$.
- Neighbor difference threshold $T_v = 50$, neighbor fraction threshold $T_n = 0.5$ ($50\%$).

1. **Check Criterion 1 (Grayscale Range Criterion)**:
   - Safe range is $[L_{\min} + T_g, \, L_{\max} - T_g] = [15, \, 240]$.
   - Since $f(x, y) = 250 \notin [15, 240]$, **Criterion 1 is SATISFIED!**
2. **Check Criterion 2 (Local Difference Criterion)**:
   - Look at all $\# [N(x, y)] = 8$ neighbors around `250`: `[100, 100, 100, 100, 100, 100, 102, 98]`.
   - How many of those 8 neighbors satisfy $|250 - f(s, t)| > T_v = 50$? **All 8 of them!** (Since $|250 - 100| = 150 > 50$.)
   - Compute the fraction:
     $$\frac{\#\big[\,|f(x, y) - f(s, t)| > 50\,\big]}{\#[N(x, y)]} = \frac{8}{8} = 1.0 > T_n = 0.5 \implies \text{Criterion 2 is SATISFIED!}$$
3. **Conclusion**: Because `250` meets **both** criteria, the Selective Filter flags it as Salt-and-Pepper noise and replaces it using its clean neighbors!

---

## PART C: IMAGE SEGMENTATION (Section 6 & Segmentation Slides)

To see how **every segmentation algorithm in Section 6** works in practice, let's trace them all step by step on the **exact same $4 \times 4$ grayscale image $I$** (where the left two columns are a dark object around `10–15` and the right two columns are a bright background around `80–90`):

$$I = \begin{bmatrix} 10 & 12 & 80 & 85 \\ 11 & 14 & 82 & 88 \\ 12 & 15 & 84 & 86 \\ 10 & 13 & 81 & 90 \end{bmatrix}$$

---

### 1. Seeded Region Growing (SRG — Section 6.1.1, p. 48) on Grid $I$
- **Step 1 (Initial Seeds)**: User selects 2 seeds: $p_1 = (1, 1)$ with value `10` (Cluster $C_1$) and $p_2 = (4, 4)$ with value `90` (Cluster $C_2$), with similarity threshold **$t = 15$**.
- **Step 2 (First Growth Iteration)**:
  - Neighbors of $p_1(1,1)$ are $(1,2)=12$, $(2,1)=11$, $(2,2)=14$. Their differences from $\mu(C_1) = 10$ are $|12-10|=2 \le 15$, $|11-10|=1 \le 15$, $|14-10|=4 \le 15 \implies$ **all three join $C_1$!**
  - Neighbors of $p_2(4,4)$ are $(3,3)=84, (3,4)=86, (4,3)=81$. Their differences from $\mu(C_2) = 90$ are $6, 4, 9 \le 15 \implies$ **all three join $C_2$!**
- **Step 3 (Update Boundaries & Cluster Means)**:
  - New mean of $C_1$: $\mu(C_1) = \frac{10 + 12 + 11 + 14}{4} = \mathbf{11.75}$.
  - New mean of $C_2$: $\mu(C_2) = \frac{90 + 84 + 86 + 81}{4} = \mathbf{85.25}$.
- **Step 4 (Continue Until All Pixels Allocated)**:
  - On the next steps, $C_1$ absorbs $(3,1)=12, (3,2)=15, (4,1)=10, (4,2)=13$, but **stops at column 3** because $|80 - 11.75| = 68.25 > 15$!
  - Meanwhile, $C_2$ absorbs $(1,3)=80, (1,4)=85, (2,3)=82, (2,4)=88$.
  - **Final SRG Partition**: Left $4 \times 2$ block = Cluster $C_1$ (mean $12.125$), Right $4 \times 2$ block = Cluster $C_2$ (mean $84.5$).

---

### 2. Unseeded Region Growing (URG — Section 6.1.1, pp. 48–49) on Grid $I$
What if we don't know how many objects exist and provide **NO seeds** (threshold $t = 15$)?
1. **Step 1**: Automatically initialize Cluster $C_1$ with the top-left pixel $(1, 1) = 10$ ($\mu(C_1) = 10$).
2. **Steps 2–4 (Growing $C_1$)**: In the first 7 iterations, the bordering pixels in columns 1 and 2 (`11, 12, 14, 12, 15, 10, 13`) all have $\delta(z, C_1) < 15$, so they all get added to $C_1$. Now $C_1$ covers the entire left $4 \times 2$ block with mean $\mu(C_1) = \mathbf{12.125}$.
3. **The Automatic Cluster Spawning Moment (Step 3 on p. 49)**:
   - Now the bordering set $S$ consists of column 3: `80, 82, 84, 81`.
   - The closest pixel in $S$ to $C_1$ is $z = (1, 3) = \mathbf{80}$, with distance:
     $$\delta(z, C_1) = |80 - 12.125| = \mathbf{67.875} \ge t = 15$$
   - Because $\delta(z, C_1) \ge 15$, pixel `80` is substantially different from $C_1$ $\implies$ **URG automatically creates a brand-new Cluster $C_2 = \{(1, 3)\}$ with mean $\mu(C_2) = 80$!**
   - In the remaining iterations, all the `81–90` pixels in columns 3 and 4 have $\delta(x, C_2) \le 10 < 15$, so they all join $C_2$!

---

### 3. Region Splitting and Merging (Quadtree — Section 6.1.2, p. 49) on Grid $I$
Define the homogeneity predicate $Q(R_k) = \text{TRUE} \iff \max(R_k) - \min(R_k) \le 10$:
1. **Step 1 (Splitting)**:
   - Test whole $4 \times 4$ image $R$: $\max(R) - \min(R) = 90 - 10 = 80 > 10 \implies Q(R) = \text{FALSE}$!
   - **Split $R$ into 4 disjoint $2 \times 2$ quadrants**:
     $$R_{\text{TL}} = \begin{bmatrix} 10 & 12 \\ 11 & 14 \end{bmatrix}, \quad R_{\text{BL}} = \begin{bmatrix} 12 & 15 \\ 10 & 13 \end{bmatrix}, \quad R_{\text{TR}} = \begin{bmatrix} 80 & 85 \\ 82 & 88 \end{bmatrix}, \quad R_{\text{BR}} = \begin{bmatrix} 84 & 86 \\ 81 & 90 \end{bmatrix}$$
   - Test each $2 \times 2$ quadrant:
     - $R_{\text{TL}}$: $14 - 10 = 4 \le 10 \implies Q(R_{\text{TL}}) = \text{TRUE}$ (Stop splitting!).
     - $R_{\text{BL}}$: $15 - 10 = 5 \le 10 \implies Q(R_{\text{BL}}) = \text{TRUE}$ (Stop splitting!).
     - $R_{\text{TR}}$: $88 - 80 = 8 \le 10 \implies Q(R_{\text{TR}}) = \text{TRUE}$ (Stop splitting!).
     - $R_{\text{BR}}$: $90 - 81 = 9 \le 10 \implies Q(R_{\text{BR}}) = \text{TRUE}$ (Stop splitting!).
2. **Step 2 (Merging)**:
   - Check adjacent quadrants $R_{\text{TL}} \cup R_{\text{BL}}$: $\max - \min = 15 - 10 = 5 \le 10 \implies Q(R_{\text{TL}} \cup R_{\text{BL}}) = \text{TRUE} \implies$ **Merge into Left Region $R_1$!**
   - Check adjacent quadrants $R_{\text{TR}} \cup R_{\text{BR}}$: $\max - \min = 90 - 80 = 10 \le 10 \implies Q(R_{\text{TR}} \cup R_{\text{BR}}) = \text{TRUE} \implies$ **Merge into Right Region $R_2$!**
   - Check $R_1 \cup R_2$: $90 - 10 = 80 > 10 \implies \text{FALSE}$ (Cannot merge!).
3. **Step 3**: Stop! We have cleanly recovered the two regions $R_1$ and $R_2$.

---

### 4. Iterative Global Thresholding (Section 6.2, p. 50) on Grid $I$
Let's trace the 5-step iterative threshold estimation algorithm on $I$ with stopping tolerance $\Delta t = 0.1$:
1. **Step 1 (Initial Estimate $t^{(0)}$)**: Start with the average intensity of all 16 pixels in $I$:
   $$t^{(0)} = \frac{(10+12+11+14+12+15+10+13) + (80+85+82+88+84+86+81+90)}{16} = \frac{97 + 676}{16} = \mathbf{48.3125}$$
2. **Step 2 (Segment into $G_1$ and $G_2$ using $t^{(0)} = 48.3125$)**:
   - $G_1$ (pixels $> 48.3125$): All 8 right-half pixels `{80, 85, 82, 88, 84, 86, 81, 90}`.
   - $G_2$ (pixels $\le 48.3125$): All 8 left-half pixels `{10, 12, 11, 14, 12, 15, 10, 13}`.
3. **Step 3 (Compute Means $m_1$ and $m_2$)**:
   $$m_1 = \frac{676}{8} = \mathbf{84.5}, \qquad m_2 = \frac{97}{8} = \mathbf{12.125}$$
4. **Step 4 (New Threshold Midpoint $t^{(1)}$)**:
   $$t^{(1)} = \frac{1}{2}(m_1 + m_2) = \frac{1}{2}(84.5 + 12.125) = \mathbf{48.3125}$$
5. **Step 5 (Convergence Check)**:
   $$|t^{(1)} - t^{(0)}| = |48.3125 - 48.3125| = 0 < \Delta t \implies \text{Converged in 1 iteration to } \mathbf{T = 48.3125}\text{!}$$
