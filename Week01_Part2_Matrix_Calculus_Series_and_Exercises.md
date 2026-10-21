# Week 1 (Part 2): Multivariate & Matrix Calculus, Matrix Series & Practice Exercises

**Sources Covered (100% No-Skip)**:
- `01-week01/01-MLDS_UDA_2026-27_MathRecap.pdf` (Sections 3.1–3.5, pp. 11–22)
- `01-week01/04-MLDS_UDA_26_27_Matrices.pdf` (Slides 7–16)
- `01-week01/02-MLDS_UDA_2026-27_MathRecap_Exercises.pdf` & `03-Solutions.pdf` (Problems 2 & 3)

---

## 1. Calculus of Functions, Gradients & Jacobians (Section 3.1)

### 1.1 Univariate Calculus & Taylor Series (Definitions 3.1–3.4) + Concrete Examples
1. **Difference Quotient vs. Derivative (Definitions 3.1–3.2)**:
   $$\frac{\delta y}{\delta x} := \frac{f(x + \delta x) - f(x)}{\delta x} \xrightarrow{\delta x \to 0} \frac{\mathrm{d}f}{\mathrm{d}x} = \lim_{h \to 0} \frac{f(x + h) - f(x)}{h}$$
   - **Concrete Example**: Let $f(x) = x^2$ at $x = 3$ (where exact tangent slope is $f'(3) = 2(3) = \mathbf{6}$):
     - For discrete pixel step $\delta x = 1$: $\frac{\delta y}{\delta x} = \frac{(3+1)^2 - 3^2}{1} = \frac{16 - 9}{1} = \mathbf{7}$ (secant slope).
     - For smaller step $\delta x = 0.01$: $\frac{\delta y}{\delta x} = \frac{3.01^2 - 3^2}{0.01} = \frac{9.0601 - 9}{0.01} = \mathbf{6.01} \approx 6$.
2. **Taylor Polynomial & Taylor Series (Definitions 3.3–3.4)**:
   $$T_n(x) := \sum_{k=0}^n \frac{f^{(k)}(x_0)}{k!}(x - x_0)^k, \qquad T_\infty(x) = \sum_{k=0}^\infty \frac{f^{(k)}(x_0)}{k!}(x - x_0)^k$$
   - **Concrete Example**: Approximate $f(x) = e^x$ near $x_0 = 0$ using its degree-2 Taylor (Maclaurin) polynomial $T_2(x)$.
     Since $f(0) = f'(0) = f''(0) = e^0 = 1$:
     $$T_2(x) = 1 + \frac{1}{1!}x + \frac{1}{2!}x^2 = 1 + x + 0.5x^2$$
     At $x = 0.2$: $T_2(0.2) = 1 + 0.2 + 0.5(0.04) = \mathbf{1.22}$, which is within $0.0014$ of the true value $e^{0.2} \approx 1.2214$!

---

### 1.2 Partial Derivatives, Gradients & The Jacobian Matrix (Definitions 3.5–3.6)

#### Case 1: Scalar-Valued Function $f : \mathbb{R}^n \to \mathbb{R}$ (Gradient Row Vector)
$$\nabla_{\mathbf{x}} f = \operatorname{grad} f = \frac{\mathrm{d}f}{\mathrm{d}\mathbf{x}} = \begin{bmatrix} \frac{\partial f}{\partial x_1} & \frac{\partial f}{\partial x_2} & \dots & \frac{\partial f}{\partial x_n} \end{bmatrix} \in \mathbb{R}^{1 \times n}$$
- **Concrete Example**: Let $f(x_1, x_2) = x_1^2 x_2 + 5x_2^3$.
  - $\frac{\partial f}{\partial x_1}$ (treat $x_2$ as a constant number): $2x_1 x_2 + 0 = 2x_1 x_2$.
  - $\frac{\partial f}{\partial x_2}$ (treat $x_1$ as a constant number): $x_1^2(1) + 15x_2^2 = x_1^2 + 15x_2^2$.
  - **Gradient row vector**: $\nabla_{\mathbf{x}} f = \begin{bmatrix} 2x_1 x_2 & x_1^2 + 15x_2^2 \end{bmatrix}_{1\times 2}$. At $(x_1, x_2) = (3, 1)$, $\nabla_{\mathbf{x}} f = \begin{bmatrix} 6 & 24 \end{bmatrix}$.

#### Case 2: Vector-Valued Function $\mathbf{f} : \mathbb{R}^n \to \mathbb{R}^m$ (Jacobian Matrix)
$$\mathbf{J} = \nabla_{\mathbf{x}} \mathbf{f} = \frac{\mathrm{d}\mathbf{f}(\mathbf{x})}{\mathrm{d}\mathbf{x}} = \begin{bmatrix} \frac{\partial \mathbf{f}}{\partial x_1} & \dots & \frac{\partial \mathbf{f}}{\partial x_n} \end{bmatrix} = \begin{bmatrix} \frac{\partial f_1}{\partial x_1} & \dots & \frac{\partial f_1}{\partial x_n} \\ \vdots & \ddots & \vdots \\ \frac{\partial f_m}{\partial x_1} & \dots & \frac{\partial f_m}{\partial x_n} \end{bmatrix} \in \mathbb{R}^{m \times n}, \qquad J(i, j) = \frac{\partial f_i}{\partial x_j}$$

- **Concrete Step-by-Step Example ($n = 2$ inputs $\to m = 3$ outputs)**:
  Let $\mathbf{f}\begin{pmatrix} x_1 \\ x_2 \end{pmatrix} = \begin{bmatrix} f_1(x_1, x_2) \\ f_2(x_1, x_2) \\ f_3(x_1, x_2) \end{bmatrix} = \begin{bmatrix} x_1^2 + x_2 \\ 3x_1 x_2 \\ \sin(x_1) \end{bmatrix}$.
  Since there are $m=3$ outputs and $n=2$ inputs, the Jacobian $\mathbf{J}$ has shape **$3 \times 2$**:
  $$\mathbf{J} = \begin{bmatrix} \frac{\partial (x_1^2 + x_2)}{\partial x_1} & \frac{\partial (x_1^2 + x_2)}{\partial x_2} \\ \frac{\partial (3x_1 x_2)}{\partial x_1} & \frac{\partial (3x_1 x_2)}{\partial x_2} \\ \frac{\partial (\sin x_1)}{\partial x_1} & \frac{\partial (\sin x_1)}{\partial x_2} \end{bmatrix} = \begin{bmatrix} 2x_1 & 1 \\ 3x_2 & 3x_1 \\ \cos(x_1) & 0 \end{bmatrix}_{3 \times 2}$$
  Evaluating at $\mathbf{x} = \begin{pmatrix} 0 \\ 2 \end{pmatrix}$ gives $\mathbf{J}\begin{pmatrix} 0 \\ 2 \end{pmatrix} = \begin{bmatrix} 0 & 1 \\ 6 & 0 \\ 1 & 0 \end{bmatrix}$.

- **Worked Example 3.7 (Notes p. 14)**: For $\mathbf{f}(\mathbf{x}) = \mathbf{A}\mathbf{x}$ ($\mathbf{A} \in \mathbb{R}^{M \times N}$), $f_i(\mathbf{x}) = \sum_{j=1}^N A_{ij} x_j \implies \frac{\partial f_i}{\partial x_j} = A_{ij} \implies \frac{\mathrm{d}(\mathbf{A}\mathbf{x})}{\mathrm{d}\mathbf{x}} = \mathbf{A}$.
- **Worked Example 3.8 (Notes pp. 14–15)**: For $h(t) = f(g(t))$ with $f(\mathbf{x}) = \exp(x_1 x_2^2)$ and $g(t) = [t\cos t, \, t\sin t]^\top$:
  $$\frac{\mathrm{d}h}{\mathrm{d}t} = \underbrace{\begin{bmatrix} \exp(x_1 x_2^2)x_2^2 & 2\exp(x_1 x_2^2)x_1 x_2 \end{bmatrix}}_{1\times 2} \underbrace{\begin{bmatrix} \cos t - t\sin t \\ \sin t + t\cos t \end{bmatrix}}_{2\times 1} = \exp(x_1 x_2^2)\big(x_2^2(\cos t - t\sin t) + 2x_1 x_2(\sin t + t\cos t)\big)$$

---

### 1.3 The "Expand $f(x+h)$ and Read Off the Linear Term" Trick (Slides 21–35 & 43–47)
Using $f(x + h) = f(x) + \mathrm{D}f(x) h + o(\|h\|)$, whatever matrix multiplies $h$ on the left in the **linear-in-$h$ term** is $\mathrm{D}f(x)$!

| Function $f(x)$ | Expansion of $f(x + h)$ | Linear-in-$h$ Term | Jacobian $\mathrm{D}f(x)$ |
| :--- | :--- | :--- | :--- |
| **$f(x) = Ax$** | $A(x + h) = Ax + Ah$ | $Ah$ | $\mathrm{D}f(x) = A \in \mathbb{R}^{m\times n}$ |
| **$f(x) = a^\top x$** | $a^\top(x + h) = a^\top x + a^\top h$ | $a^\top h$ | $\mathrm{D}f(x) = a^\top \in \mathbb{R}^{1\times n}$ |
| **Quadratic Form: $f(x) = x^\top A x$** | $x^\top Ax + x^\top Ah + h^\top Ax + h^\top Ah$ | $x^\top(A + A^\top)h$ | $\mathrm{D}f(x) = x^\top(A + A^\top) \in \mathbb{R}^{1\times n}$ |
| **Least Squares: $f(w) = \|y - Xw\|^2$** | $\|r\|^2 - 2r^\top Xh + \|Xh\|^2$ ($r = y - Xw$) | $-2(y - Xw)^\top X h$ | $\mathrm{D}f(w) = -2(y - Xw)^\top X \in \mathbb{R}^{1\times d}$ |
| **1-Layer Neural Net: $z(x) = \frac{1}{2}\|Ax + b\|^2$** | Chain rule on $g(x)=Ax+b, f(y)=\frac{1}{2}y^\top y$ | $(Ax+b)^\top A h$ | $\mathrm{D}z(x) = (Ax + b)^\top A \in \mathbb{R}^{1\times n}$ |

#### Concrete Numerical Example: Why $\mathrm{D}(x^\top A x) = x^\top(A + A^\top)$ Instead of $2x^\top A$ When $A$ is Not Symmetric!
Let $A = \begin{pmatrix} 1 & 3 \\ 0 & 2 \end{pmatrix}$ (not symmetric!) and $x = \begin{pmatrix} x_1 \\ x_2 \end{pmatrix}$.
1. **Multiply out $f(x) = x^\top A x$ into a scalar polynomial**:
   $$f(x) = \begin{pmatrix} x_1 & x_2 \end{pmatrix}\begin{pmatrix} 1 & 3 \\ 0 & 2 \end{pmatrix}\begin{pmatrix} x_1 \\ x_2 \end{pmatrix} = \begin{pmatrix} x_1 & x_2 \end{pmatrix}\begin{pmatrix} x_1 + 3x_2 \\ 2x_2 \end{pmatrix} = x_1^2 + 3x_1 x_2 + 2x_2^2$$
2. **Take ordinary partial derivatives w.r.t. $x_1$ and $x_2$**:
   $$\frac{\partial f}{\partial x_1} = 2x_1 + 3x_2, \qquad \frac{\partial f}{\partial x_2} = 3x_1 + 4x_2 \implies \mathrm{D}f(x) = \begin{bmatrix} 2x_1 + 3x_2 & 3x_1 + 4x_2 \end{bmatrix}$$
3. **Now check the formula $x^\top(A + A^\top)$**:
   $$A + A^\top = \begin{pmatrix} 1 & 3 \\ 0 & 2 \end{pmatrix} + \begin{pmatrix} 1 & 0 \\ 3 & 2 \end{pmatrix} = \begin{pmatrix} 2 & 3 \\ 3 & 4 \end{pmatrix} \implies x^\top(A + A^\top) = \begin{pmatrix} x_1 & x_2 \end{pmatrix}\begin{pmatrix} 2 & 3 \\ 3 & 4 \end{pmatrix} = \begin{bmatrix} 2x_1 + 3x_2 & 3x_1 + 4x_2 \end{bmatrix} \text{ (Exact Match!)}$$
   *(Notice that if you carelessly wrote $2x^\top A = \begin{bmatrix} 2x_1 & 6x_1 + 4x_2 \end{bmatrix}$, it would be completely wrong because $A \neq A^\top$!)*

---

## 2. Matrix Calculus & Layout Conventions (Section 3.2 & Slides 48–62)

### 2.1 The Course's Block-Wise Matrix Derivative Convention
When differentiating $\mathbf{F}(\mathbf{X}) \in \mathbb{R}^{p \times q}$ with respect to $\mathbf{X} \in \mathbb{R}^{m \times n}$:
1. **Outer Layout is the TRANSPOSE of $\mathbf{X}$'s shape ($n \times m$ blocks)**:
   $$\frac{\partial \mathbf{F}}{\partial \mathbf{X}} = \begin{bmatrix} \frac{\partial \mathbf{F}}{\partial X_{1,1}} & \dots & \frac{\partial \mathbf{F}}{\partial X_{m,1}} \\ \vdots & \ddots & \vdots \\ \frac{\partial \mathbf{F}}{\partial X_{1,n}} & \dots & \frac{\partial \mathbf{F}}{\partial X_{m,n}} \end{bmatrix}$$
2. **Scalar Function of a Matrix $f : \mathbb{R}^{m \times n} \to \mathbb{R}$**:
   $\frac{\partial f}{\partial \mathbf{X}} \in \mathbb{R}^{n \times m}$ has $(j, i)$th entry $\left(\frac{\partial f}{\partial \mathbf{X}}\right)_{ji} := \frac{\partial f}{\partial X_{ij}}$.

#### Concrete $2 \times 2$ Verification of the Scalar-by-Matrix Table (Slides 48–52)
Let $X = \begin{pmatrix} x_{11} & x_{12} \\ x_{21} & x_{22} \end{pmatrix}$, $a = \begin{pmatrix} 2 \\ 3 \end{pmatrix}$, and $b = \begin{pmatrix} 4 \\ 5 \end{pmatrix}$.
1. **Verify $\frac{\partial (a^\top X b)}{\partial X} = b a^\top$**:
   - Write out the scalar $f(X) = a^\top X b$:
     $$f(X) = \begin{pmatrix} 2 & 3 \end{pmatrix}\begin{pmatrix} x_{11} & x_{12} \\ x_{21} & x_{22} \end{pmatrix}\begin{pmatrix} 4 \\ 5 \end{pmatrix} = 8x_{11} + 10x_{12} + 12x_{21} + 15x_{22}$$
   - Differentiate w.r.t. each $x_{ij}$ and place in the **transposed** position $(j, i)$:
     $$\frac{\partial f}{\partial X} = \begin{pmatrix} \frac{\partial f}{\partial x_{11}} & \frac{\partial f}{\partial x_{21}} \\ \frac{\partial f}{\partial x_{12}} & \frac{\partial f}{\partial x_{22}} \end{pmatrix} = \begin{pmatrix} 8 & 12 \\ 10 & 15 \end{pmatrix} = \begin{pmatrix} 4 \\ 5 \end{pmatrix}\begin{pmatrix} 2 & 3 \end{pmatrix} = b a^\top \text{ ✓}$$
2. **Verify $\frac{\partial \operatorname{tr}(AX)}{\partial X} = A$**:
   - Let $A = \begin{pmatrix} 1 & 2 \\ 3 & 4 \end{pmatrix}$. Then $AX = \begin{pmatrix} x_{11} + 2x_{21} & x_{12} + 2x_{22} \\ 3x_{11} + 4x_{21} & 3x_{12} + 4x_{22} \end{pmatrix}$.
   - Its trace is $\operatorname{tr}(AX) = (x_{11} + 2x_{21}) + (3x_{12} + 4x_{22})$.
   - Assemble in the transposed layout:
     $$\frac{\partial \operatorname{tr}(AX)}{\partial X} = \begin{pmatrix} \frac{\partial}{\partial x_{11}} & \frac{\partial}{\partial x_{21}} \\ \frac{\partial}{\partial x_{12}} & \frac{\partial}{\partial x_{22}} \end{pmatrix} \operatorname{tr}(AX) = \begin{pmatrix} 1 & 2 \\ 3 & 4 \end{pmatrix} = A \text{ ✓}$$

---

### 2.2 Concrete Walkthrough of the Two Classic Matrix Calculus Traps (Examples 3.9 & 3.10)

#### Trap 1: Commutativity & Dimension Mismatch (Example 3.9, Notes p. 16)
Let $\mathbf{f}(\mathbf{x}) = (\mathbf{a}^\top \mathbf{x})\mathbf{b}$ with concrete vectors $\mathbf{a} = \begin{bmatrix} 1 \\ 2 \end{bmatrix} \in \mathbb{R}^2$, $\mathbf{x} = \begin{bmatrix} x_1 \\ x_2 \end{bmatrix} \in \mathbb{R}^2$, and $\mathbf{b} = \begin{bmatrix} 3 \\ 4 \\ 5 \end{bmatrix} \in \mathbb{R}^3$.
- Let's write out $\mathbf{f}(\mathbf{x})$ explicitly! Since $\mathbf{a}^\top \mathbf{x} = x_1 + 2x_2$ is a scalar:
  $$\mathbf{f}(\mathbf{x}) = (x_1 + 2x_2)\begin{bmatrix} 3 \\ 4 \\ 5 \end{bmatrix} = \begin{bmatrix} 3x_1 + 6x_2 \\ 4x_1 + 8x_2 \\ 5x_1 + 10x_2 \end{bmatrix} \in \mathbb{R}^3$$
- Taking the $3 \times 2$ Jacobian $\frac{\partial \mathbf{f}}{\partial \mathbf{x}}$ directly from those 3 rows gives:
  $$\frac{\partial \mathbf{f}}{\partial \mathbf{x}} = \begin{bmatrix} 3 & 6 \\ 4 & 8 \\ 5 & 10 \end{bmatrix}_{3 \times 2} = \begin{bmatrix} 3 \\ 4 \\ 5 \end{bmatrix}\begin{bmatrix} 1 & 2 \end{bmatrix} = \mathbf{b}\mathbf{a}^\top$$
  *(Notice why $\mathbf{a}^\top \mathbf{b}$ doesn't even exist here: $\mathbf{a}^\top$ is $1 \times 2$ and $\mathbf{b}$ is $3 \times 1$! Moving the scalar $(\mathbf{a}^\top \mathbf{x})$ to the right of $\mathbf{b}$ as $\mathbf{f}(\mathbf{x}) = \mathbf{b}(\mathbf{a}^\top \mathbf{x})$ gives $\frac{\partial \mathbf{f}}{\partial \mathbf{x}} = \mathbf{b}\mathbf{a}^\top I_2 = \mathbf{b}\mathbf{a}^\top$.)*

#### Trap 2: Differentiating a Transposed Vector (Example 3.10, Notes pp. 16–17)
For $f(\mathbf{w}) = (y - \mathbf{w}^\top \mathbf{x})^2$, let's check with $\mathbf{w} = \begin{bmatrix} w_1 \\ w_2 \end{bmatrix}, \mathbf{x} = \begin{bmatrix} 2 \\ 3 \end{bmatrix}, y = 10$:
- Scalar expansion: $f(w_1, w_2) = (10 - 2w_1 - 3w_2)^2$.
- By ordinary calculus:
  $$\frac{\partial f}{\partial w_1} = 2(10 - 2w_1 - 3w_2)(-2), \qquad \frac{\partial f}{\partial w_2} = 2(10 - 2w_1 - 3w_2)(-3)$$
- Collected as a $1 \times 2$ row vector:
  $$\frac{\partial f}{\partial \mathbf{w}} = -2(10 - 2w_1 - 3w_2)\begin{bmatrix} 2 & 3 \end{bmatrix} = -2(y - \mathbf{w}^\top \mathbf{x})\mathbf{x}^\top = -2y\mathbf{x}^\top + 2(\mathbf{w}^\top \mathbf{x})\mathbf{x}^\top$$

---

## 3. Multivariate & Matrix Chain Rule (Section 3.3 & Slides 36–42)

### Concrete Example Showing Why Matrix Order Matters in the Chain Rule (Slides 39–42)
Let $g(\mathbf{x}) = G\mathbf{x}$ and $f(\mathbf{y}) = F\mathbf{y}$ with:
$$G = \begin{pmatrix} 1 & 1 \\ 0 & 1 \end{pmatrix}, \qquad F = \begin{pmatrix} 0 & 1 \\ 1 & 0 \end{pmatrix}$$
- Since $f(g(\mathbf{x})) = F(G\mathbf{x}) = (FG)\mathbf{x}$, the true Jacobian of $f \circ g$ is:
  $$\mathrm{D}(f \circ g)(\mathbf{x}) = \mathrm{D}f(g(\mathbf{x})) \cdot \mathrm{D}g(\mathbf{x}) = FG = \begin{pmatrix} 0 & 1 \\ 1 & 0 \end{pmatrix}\begin{pmatrix} 1 & 1 \\ 0 & 1 \end{pmatrix} = \begin{pmatrix} 0 & 1 \\ 1 & 1 \end{pmatrix}$$
- If you multiplied the Jacobians in the wrong order ($GF$), both are $2 \times 2$ so the shapes wouldn't warn you, but you would get the wrong answer $\begin{pmatrix} 1 & 1 \\ 1 & 0 \end{pmatrix} \neq \begin{pmatrix} 0 & 1 \\ 1 & 1 \end{pmatrix}$!

---

## 4. Matrix Series & Functions of Matrices (Sections 3.4 & 3.5)

### 4.1 Matrix Series Convergence (Example 3.13, Notes pp. 20–21)
For $A_k = \begin{bmatrix} y^k & 2^{-k} \\ \frac{x^k}{k!} & (-1)^k \frac{\theta^{2k}}{(2k)!} \end{bmatrix}$, summing from $k = 0$ to $\infty$ entry-by-entry gives:
$$\sum_{k=0}^\infty A_k = \begin{bmatrix} \sum_{k=0}^\infty y^k & \sum_{k=0}^\infty 2^{-k} \\ \sum_{k=0}^\infty \frac{x^k}{k!} & \sum_{k=0}^\infty \frac{(-1)^k \theta^{2k}}{(2k)!} \end{bmatrix} = \begin{bmatrix} \frac{1}{1-y} & 2 \\ e^x & \cos\theta \end{bmatrix} \quad (\text{converges } \iff |y| < 1)$$

### 4.2 Concrete Numerical Example of the Neumann Series (Example 3.14, Notes p. 22)
**Theorem**: $(I - A)^{-1} = I + A + A^2 + A^3 + \dots = \sum_{k=0}^\infty A^k$ whenever all eigenvalues of $A$ satisfy $|\lambda_i| < 1$.
- **Concrete Numerical Test**: Let $A = \begin{pmatrix} 0.5 & 0 \\ 0 & 0.2 \end{pmatrix}$. Since $A$ is diagonal, its eigenvalues are $\lambda_1 = 0.5$ and $\lambda_2 = 0.2$ (both $< 1$).
  - **Exact Inverse**:
    $$(I - A)^{-1} = \begin{pmatrix} 1 - 0.5 & 0 \\ 0 & 1 - 0.2 \end{pmatrix}^{-1} = \begin{pmatrix} 0.5 & 0 \\ 0 & 0.8 \end{pmatrix}^{-1} = \begin{pmatrix} \mathbf{2} & 0 \\ 0 & \mathbf{1.25} \end{pmatrix}$$
  - **Sum of the first 4 terms of the Neumann Series ($I + A + A^2 + A^3$)**:
    $$I + A + A^2 + A^3 = \begin{pmatrix} 1 + 0.5 + 0.25 + 0.125 & 0 \\ 0 & 1 + 0.2 + 0.04 + 0.008 \end{pmatrix} = \begin{pmatrix} \mathbf{1.875} & 0 \\ 0 & \mathbf{1.248} \end{pmatrix} \xrightarrow{k \to \infty} \begin{pmatrix} \mathbf{2} & 0 \\ 0 & \mathbf{1.25} \end{pmatrix}$$

---

## 5. Complete Step-by-Step Solutions to Practice Exercises (Problems 2 & 3)

### Problem 2.1: Compute $e^A$ for $A = \begin{bmatrix} 3 & 5 \\ 1 & -1 \end{bmatrix}$
1. **Eigenvalues**: $\det(A - \lambda I) = (3-\lambda)(-1-\lambda) - 5 = \lambda^2 - 2\lambda - 8 = (\lambda - 4)(\lambda + 2) = 0 \implies \lambda_1 = 4, \lambda_2 = -2$.
2. **Eigenvectors**:
   - $\lambda_1 = 4 \implies \begin{bmatrix} -1 & 5 \\ 1 & -5 \end{bmatrix}\begin{bmatrix} x \\ y \end{bmatrix} = \begin{bmatrix} 0 \\ 0 \end{bmatrix} \implies p_1 = \begin{bmatrix} 5 \\ 1 \end{bmatrix}$.
   - $\lambda_2 = -2 \implies \begin{bmatrix} 5 & 5 \\ 1 & 1 \end{bmatrix}\begin{bmatrix} x \\ y \end{bmatrix} = \begin{bmatrix} 0 \\ 0 \end{bmatrix} \implies p_2 = \begin{bmatrix} -1 \\ 1 \end{bmatrix}$.
3. **Compute $e^A = P e^D P^{-1}$**:
   $$e^A = \begin{bmatrix} 5 & -1 \\ 1 & 1 \end{bmatrix} \begin{bmatrix} e^4 & 0 \\ 0 & e^{-2} \end{bmatrix} \cdot \frac{1}{6}\begin{bmatrix} 1 & 1 \\ -1 & 5 \end{bmatrix} = \frac{1}{6}\begin{bmatrix} 5e^4 + e^{-2} & 5e^4 - 5e^{-2} \\ e^4 - e^{-2} & e^4 + 5e^{-2} \end{bmatrix}$$

### Problem 2.2: Properties of $e^A$ (with $A = PDP^{-1}$)
- **(a)** $e^{0A} = P e^{0D} P^{-1} = P I P^{-1} = I$.
- **(b)** $(e^A)^\top = (P e^D P^{-1})^\top = (P^\top)^{-1} e^D P^\top = e^{(P^\top)^{-1} D P^\top} = e^{A^\top}$.
- **(c)** $e^A e^{-A} = (P e^D P^{-1})(P e^{-D} P^{-1}) = P e^D e^{-D} P^{-1} = P I P^{-1} = I \implies (e^A)^{-1} = e^{-A}$.
- **(d)** $e^{aA} e^{bA} = (P e^{aD} P^{-1})(P e^{bD} P^{-1}) = P e^{(a+b)D} P^{-1} = e^{(a+b)A}$.
- **(e) Non-commuting counterexample ($e^{A+B} \neq e^A e^B$)**:
  For $A = \begin{bmatrix} 0 & 0 \\ 1 & 0 \end{bmatrix}$ and $B = \begin{bmatrix} 0 & 0 \\ 0 & 1 \end{bmatrix}$, $e^A e^B = \begin{bmatrix} 1 & 0 \\ 1 & 1 \end{bmatrix}\begin{bmatrix} 1 & 0 \\ 0 & e \end{bmatrix} = \begin{bmatrix} 1 & 0 \\ 1 & e \end{bmatrix}$, whereas $e^{A+B} = \begin{bmatrix} 1 & 0 \\ e-1 & e \end{bmatrix}$.

### Problem 3: Matrix Calculus & Chain Rule Exercises
1. **Problem 3.1**: For $C = AB = \begin{bmatrix} a_1b_1 + a_2b_3 & a_1b_2 + a_2b_4 \\ a_3b_1 + a_4b_3 & a_3b_2 + a_4b_4 \end{bmatrix}$:
   $$\frac{\partial C}{\partial A} = \begin{bmatrix} \frac{\partial C}{\partial a_1} & \frac{\partial C}{\partial a_3} \\ \frac{\partial C}{\partial a_2} & \frac{\partial C}{\partial a_4} \end{bmatrix} = \begin{bmatrix} b_1 & b_2 & 0 & 0 \\ 0 & 0 & b_1 & b_2 \\ b_3 & b_4 & 0 & 0 \\ 0 & 0 & b_3 & b_4 \end{bmatrix}$$
2. **Problem 3.2**: For $A = \begin{bmatrix} 2xy & y^2 & y \\ x^2 & 2xy & x \end{bmatrix}$ and $X = \begin{bmatrix} x \\ y \end{bmatrix}$:
   $$\frac{\partial A}{\partial X} = \begin{bmatrix} \frac{\partial A}{\partial x} & \frac{\partial A}{\partial y} \end{bmatrix} = \begin{bmatrix} 2y & 0 & 0 & 2x & 2y & 1 \\ 2x & 2y & 1 & 0 & 2x & 0 \end{bmatrix}, \qquad \frac{\partial^2 A}{\partial X^2} = \begin{bmatrix} 0 & 0 & 0 & 2 & 0 & 0 & 2 & 0 & 0 & 0 & 2 & 0 \\ 2 & 0 & 0 & 0 & 2 & 0 & 0 & 2 & 0 & 0 & 0 & 0 \end{bmatrix}$$
3. **Problem 3.3**: For $Y = \begin{bmatrix} a & b & c \\ d & e & f \end{bmatrix}_{2\times 3}$ and $X = \begin{bmatrix} u & x \\ v & y \\ w & z \end{bmatrix}_{3\times 2}$, $\frac{\partial Y}{\partial X} = \begin{bmatrix} \frac{\partial Y}{\partial u} & \frac{\partial Y}{\partial v} & \frac{\partial Y}{\partial w} \\ \frac{\partial Y}{\partial x} & \frac{\partial Y}{\partial y} & \frac{\partial Y}{\partial z} \end{bmatrix}_{4\times 9}$.
4. **Problem 3.4**: For $z = f(Y)$ with $Y = AX + B$:
   - **Concrete $2 \times 2$ Example First**: Suppose $f(Y) = \operatorname{tr}(Y) = Y_{11} + Y_{22}$ (so $\frac{\partial f}{\partial Y} = I_2$), and $Y = AX + B$. Then $z = \operatorname{tr}(AX + B) = \operatorname{tr}(AX) + \operatorname{tr}(B)$. By our scalar-by-matrix table, $\frac{\partial z}{\partial X} = A = I_2 A = \frac{\partial f}{\partial Y} A$!
   - **General Index Proof**:
     $$Y_{k\ell} = \sum_m A_{km} X_{m\ell} + B_{k\ell} \implies \frac{\partial Y_{k\ell}}{\partial X_{ij}} = \sum_m A_{km} \delta_{mi} \delta_{\ell j} = A_{ki} \delta_{\ell j}$$
     $$\left(\frac{\partial z}{\partial X}\right)_{ji} = \frac{\partial z}{\partial X_{ij}} = \sum_{k,\ell} \frac{\partial f}{\partial Y_{k\ell}} A_{ki} \delta_{\ell j} = \sum_k \frac{\partial f}{\partial Y_{kj}} A_{ki} = \sum_k \left(\frac{\partial f}{\partial Y}\right)_{jk} A_{ki} = \left(\frac{\partial f}{\partial Y} A\right)_{ji} \implies \frac{\partial z}{\partial X} = \frac{\partial f}{\partial Y} A$$
