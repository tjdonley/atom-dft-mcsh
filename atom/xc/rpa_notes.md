## Random Phase Approximation and OEP Summary



---

### 1. Core Definitions

- **Orbitals**: occupied indices $p,q,\dots$ and virtual indices $a,b,\dots$, each with quantum numbers $(n,l)$ and eigenvalues $\epsilon$.
- **Occupation**: $f_p \in \{0,1,2\}$; the angular degeneracy is $2l_p+1$.
- **Energy difference**: $\Delta\epsilon_{pq} = \epsilon_p - \epsilon_q$.
- **Orbital pair product**: $\Phi_{pq}(r) = \phi_p(r)\phi_q(r)$, the product of two orbitals at radial point $r$. This appears in RPA response kernels and OEP driving terms.
- **Radial quadrature**: nodes $(r_i, w_i)$ after mapping finite elements to physical space.
- **Imaginary frequency**: all kernels are evaluated at $i\omega$ with $\omega>0$.

---

### 2. Frequency Integration

1. **Semi-infinite grid**  
   Map Gauss–Legendre nodes $\xi \in [-1,1]$ to $\omega \in [0,\infty)$:
   $$
     \omega(\xi) = \alpha\frac{1+\xi}{1-\xi},\qquad
     \frac{d\omega}{d\xi} = \frac{2\alpha}{(1-\xi)^2},
   $$
   where $\alpha$ sets the frequency compression.

2. **Low-frequency smoothing**  
   A secondary grid on $[0,\omega_c]$ refines the $\omega \to 0$ region. Both grids use the driver parameter `frequency_quadrature_point_number`; their weights incorporate the Jacobian factors above.

---

### 3. Independent-Particle Response $\chi_0(i\omega)$

For each angular channel $L$, the reduced susceptibility includes both occ-virt and occ-occ contributions:

**Occ-virt contribution**:
$$
  \chi_{0,L}^{\text{occ-virt}}(i\omega) =
  \sum_{p \in \text{occ}} \sum_{q \in \text{virt}}
  \frac{(2l_p+1)(2l_q+1)}{2L+1}
  \frac{\Delta\epsilon_{pq}}{(\Delta\epsilon_{pq})^2 + \omega^2}
  \left(
    \begin{matrix}
      l_p & l_q & L \\
      0   & 0   & 0
    \end{matrix}
  \right)^2
  R^{(L)}_{pq}.
$$

**Occ-occ contribution** (for fractional occupations):
$$
  \chi_{0,L}^{\text{occ-occ}}(i\omega) =
  \sum_{p,p' \in \text{occ}}
  C_{pp'}^{\text{occ-occ}}
  \frac{\Delta\epsilon_{pp'}}{(\Delta\epsilon_{pp'})^2 + \omega^2}
  \left(
    \begin{matrix}
      l_p & l_{p'} & L \\
      0   & 0     & 0
    \end{matrix}
  \right)^2
  R^{(L)}_{pp'},
$$
where $C_{pp'}^{\text{occ-occ}} = f_p(2l_{p'}+1) - f_{p'}(2l_p+1)$ is the occ-occ coupling constant.

**Total response**:
$$
  \chi_{0,L}(i\omega) = \chi_{0,L}^{\text{occ-virt}}(i\omega) + \chi_{0,L}^{\text{occ-occ}}(i\omega).
$$

Important points:
- The Lorentzian factor $\Delta\epsilon_{pq}/[(\Delta\epsilon_{pq})^2 + \omega^2]$ belongs **only** to $\chi_0(i\omega)$.
- $(p,q)$ index orbitals. The quantities $R^{(L)}_{pq}$ are radial matrix elements defined in §5.
- The occ-occ terms are essential for fractional occupations and contribute to both the RPA response kernel and the OEP driving terms.

---

### 4. Angular Expansion and Wigner Factors

- **Triangle rule**: $|l_p - l_q| \le L \le l_p + l_q$.
- **Parity rule**: $l_p + l_q + L$ is even.
- The weight
  $$
    \left(
      \begin{matrix}
        l_p & l_q & L \\
        0   & 0   & 0
      \end{matrix}
    \right)^2
  $$
  enforces these constraints automatically. Angular indices govern coupling; radial indices $(i,j)$ appear only after projecting onto $R^{(L)}$.

---

### 5. Radial Coulomb Kernels

For each channel $L$ the projector on radial grids is
$$
  R^{(L)}(r_i, r_j) =
  \frac{1}{2L+1}
  \frac{r_<^{L}}{r_>^{L+1}}
  w_i w_j,
$$
with $r_< = \min(r_i, r_j)$ and $r_> = \max(r_i, r_j)$. Once orbitals are interpolated to $(r_i)$, $R^{(L)}$ acts purely in radial space. This separation is what allows $(p,q)$ and $(i,j)$ to be treated distinctly.

---

### 6. RPA Correlation Energy

- Define the radial matrix entering the log-determinant:
  $$
    M_L(\omega) = v_L\,\chi_{0,L}(i\omega) = v_L\left[\chi_{0,L}^{\text{occ-virt}}(i\omega) + \chi_{0,L}^{\text{occ-occ}}(i\omega)\right],
  $$
  where $v_L$ is the Coulomb kernel in channel $L$ (numerically identical to $R^{(L)}$ up to normalization).
- The orbital indices are fully contracted inside $\chi_{0,L}$; only the radial matrix $M_L(\omega)$ appears in the determinant/traces.
- The occ-occ contribution modifies the response kernel, which affects the screened response and correlation energy.
- Discrete RPA correlation energy:
  $$
    E_c^{\text{RPA}} =
    \frac{1}{2\pi} \sum_{L}(2L+1)
    \sum_{\omega}
    \left[
      \log\det(I - M_L(\omega))
      + \operatorname{Tr}(M_L(\omega))
    \right] w_\omega.
  $$
  The trace term subtracts the linear contribution and improves numerical stability.

---

### 7. OEP Driving Terms ($Q_{1c}$, $Q_{2c}$, $\widehat{\Sigma}_c$)

**Orbital pair product**:
$$
  \Phi_{pq}(r) = \phi_p(r)\phi_q(r),
$$
where $\phi_p(r)$ and $\phi_q(r)$ are the radial parts of orbitals $p$ and $q$ evaluated at radial point $r$.

**Wigner 3-j symbol squared**:
$$
  W_{pq}^{(L)} = \left(\begin{smallmatrix} l_p & l_q & L \\ 0 & 0 & 0 \end{smallmatrix}\right)^2.
$$

Using the channel kernel $R^{(L)}$, define the channel-projected products
$$
  \mathcal{R}^{(L)}_{pq}(r) =
  \int dr'\, R^{(L)}(r,r')\, \Phi_{pq}(r').
$$

#### 7.1. $Q_{1c}$ Term (First Correlation Driving Term)

The $Q_{1c}$ term involves the correlation self-energy $\Sigma_c$ computed via Dyson's equation. The self-energy includes contributions from both occ-virt and occ-occ pairs (the latter through the modified response kernel).

At quadrature point $r_i$:
$$
  Q_{1c}(r_i) =
  \frac{1}{\pi}
  \sum_{L}(2L+1)\sum_{\omega} w_\omega
  \sum_{p\in\text{occ}}\sum_{q\in\text{virt}}
  \frac{\Delta\epsilon_{pq}}{(\Delta\epsilon_{pq})^2 + \omega^2}
  W_{pq}^{(L)}
  \Phi_{pq}(r_i)\,
  \Sigma^{(L)}_{pq}(\omega),
$$

**Implementation details** (from `_compute_rpa_correlation_driving_term`):
- The self-energy term $\Sigma^{(L)}_{pq}(\omega)$ is computed by solving Dyson's equation:
  $$
    \chi_L(i\omega) = \chi_{0,L}(i\omega) + \chi_{0,L}(i\omega) v_L \chi_L(i\omega),
  $$
  which gives the screened response $\chi_L(i\omega) = (I - \chi_{0,L} v_L)^{-1} \chi_{0,L}$.
- **Important**: The response kernel $\chi_{0,L}$ includes occ-occ contributions, which modify the screened response $\chi_L$ and thus affect the self-energy.
- The self-energy contribution is:
  $$
    \Sigma^{(L)}_{pq}(\omega) = \int dr'\, \left[\chi_L(r,r';i\omega) - \chi_{0,L}(r,r';i\omega)\right] \Phi_{pq}(r'),
  $$
  where the difference $\chi_L - \chi_{0,L}$ is computed via `dyson_solved_response = solve(I - rpa_response_kernel, radial_kernel) - radial_kernel`.
- The $Q_{1c}$ term is then constructed by summing over all $l_{\text{OEP}}$ channels (angular momentum channels for the OEP potential):
  $$
    Q_{1c}(r_i) = 4 \sum_{l_{\text{OEP}}} \sum_{i,j} \phi_i(r_i) \frac{1}{\epsilon_i - \epsilon_j} \phi_j(r_i) \left[\Sigma_c\right]_{ij}(r_i),
  $$
  where the factor 4 comes from spin and angular degeneracy, and $\left[\Sigma_c\right]_{ij}$ is the self-energy matrix element in the orbital basis. The occ-occ contributions are implicitly included through the modified self-energy.

#### 7.2. $Q_{2c}$ Term (Second Correlation Driving Term)

The $Q_{2c}$ term arises from the frequency derivative of $\chi_{0,L}$ and includes both occ-virt and occ-occ contributions:

**Occ-virt contribution**:
$$
  Q_{2c}^{\text{occ-virt}}(r_i) =
  \frac{1}{\pi}
  \sum_{L}(2L+1)\sum_{\omega} w_\omega
  \sum_{p \in \text{occ}}\sum_{q \in \text{virt}}
  f_p (2l_q + 1)
  \frac{(\Delta\epsilon_{pq})^2 - \omega^2}{\left[(\Delta\epsilon_{pq})^2 + \omega^2\right]^2}
  W_{pq}^{(L)}
  \left[\phi_p^2(r_i) - \phi_q^2(r_i)\right]\,
  \tilde{\Sigma}^{(L)}_{pq}(\omega),
$$

**Occ-occ contribution** (for fractional occupations):
$$
  Q_{2c}^{\text{occ-occ}}(r_i) =
  \frac{1}{2\pi}
  \sum_{L}(2L+1)\sum_{\omega} w_\omega
  \sum_{p,p' \in \text{occ}}
  C_{pp'}^{\text{occ-occ}}
  \frac{(\Delta\epsilon_{pp'})^2 - \omega^2}{\left[(\Delta\epsilon_{pp'})^2 + \omega^2\right]^2}
  W_{pp'}^{(L)}
  \phi_p^2(r_i)\,
  \tilde{\Sigma}^{(L)}_{pp'}(\omega),
$$
where $C_{pp'}^{\text{occ-occ}} = f_p(2l_{p'}+1) - f_{p'}(2l_p+1)$.

**Total $Q_{2c}$ term**:
$$
  Q_{2c}(r_i) = Q_{2c}^{\text{occ-virt}}(r_i) + Q_{2c}^{\text{occ-occ}}(r_i).
$$

The factor $[(\Delta\epsilon_{pq})^2 - \omega^2]/[(\Delta\epsilon_{pq})^2 + \omega^2]^2$ arises directly from the derivative $\partial \chi_{0,L}/\partial(i\omega)$.

**Implementation details**:
- For occ-virt pairs: The orbital squared difference $\phi_p^2(r) - \phi_q^2(r)$ appears instead of $\Phi_{pq}(r)$ due to the frequency derivative structure.
- For occ-occ pairs: Only $\phi_p^2(r)$ appears (not the difference), as both orbitals are occupied.
- The prefactor for occ-virt is: `prefactors_q2c = f_p (2l_q + 1) * [(\Delta\epsilon_{pq})^2 - \omega^2] / [(\Delta\epsilon_{pq})^2 + \omega^2]^2`.
- The prefactor for occ-occ is: `select_common_constants_term2 = C_{pp'}^{\text{occ-occ}} * [(\Delta\epsilon_{pp'})^2 - \omega^2] / [(\Delta\epsilon_{pp'})^2 + \omega^2]^2`.
- The final $Q_{2c}$ term includes a factor of 2 for occ-virt: `q2c_term_all += 2 * q2c_term` (from the implementation).

#### 7.3. $\widehat{\Sigma}_c$ and Grid Interpolation

- At the OEP-basis level, $\widehat{\Sigma}_c$ is formed by mapping the pairwise $\Sigma^{(L)}_{pq}$ contributions to the auxiliary basis via the global interpolation matrix $K$, producing the matrix that multiplies the OEP coefficients during the solve.
- The implementation uses two grids:
  - **Standard grid**: dense quadrature grid for computing $\chi_0$ and driving terms
  - **OEP grid**: sparser grid for solving the OEP linear system (reduces computational cost)
- Grid conversion is performed via:
  $$
    \chi_0^{\text{OEP}} = K^\top (\chi_0 \odot w) K, \qquad
    Q^{\text{OEP}} = K^\top (Q \odot w),
  $$
  where $K$ is the `global_interpolation_matrix` and $w$ are the quadrature weights.

#### 7.4. OEP Linear System

After summing over all $(L,\omega)$ and interpolating to the OEP grid via $K$,
$$
  \chi^{\text{OEP}} c_c = K^\top \!\left[(Q_{1c} + Q_{2c}) \odot w\right],
  \qquad
  \chi^{\text{OEP}} = K^\top (\chi_0 \odot w) K.
$$
The resulting potential is $V_c^{\text{OEP}} = K c_c$, scaled by $\lambda^3$ for double-hybrid mixing.

**Numerical considerations**:
- The linear system is solved using `scipy.linalg.solve`, with condition number monitoring.
- If the system is ill-conditioned, warnings are caught and recorded (via `rcond_list_for_correlation`).
- For correlation potential, the HOMO coefficient is shifted to zero: `rpa_correlation_coefficient -= rpa_correlation_coefficient[-1]`.

---

### 8. Exchange-Only OEP (OEPx / EXX)

- Uses the same projected susceptibility $\chi^{\text{OEP}}$ but replaces the right-hand side with $Q_x$, derived from exact-exchange matrix elements.
- The solution $c_x$ produces $V_x^{\text{OEP}} = K c_x$. A constant shift ensures the HOMO expectation value of $V_x^{\text{OEP}}$ matches that of the underlying HF exchange operator.

#### 8.1. Exchange Kernel and Driving Term

The OEP exchange kernel $\chi_0$ and driving term $Q_x$ are computed in `_compute_oep_kernel_and_exchange_driving_term`:

**Exchange kernel**:
$$
  \chi_0(r_i, r_j) = 4 \sum_{p \in \text{occ}} (2l_p + 1) \phi_p(r_i) G_p(r_i, r_j) \phi_p(r_j),
$$
where $G_p$ is the Green's function block for orbital $p$:
$$
  G_p(r_i, r_j) = \sum_{q \in \text{unocc}, l_q = l_p} \frac{\phi_q(r_i) \phi_q(r_j)}{\epsilon_p - \epsilon_q}.
$$
The factor 4 comes from spin degeneracy, and $(2l_p + 1)$ is the angular degeneracy.

**Exchange driving term**:
$$
  Q_x(r_i) = 4 \sum_{p \in \text{occ}} (2l_p + 1) \phi_p(r_i) \int G_p(r_i, r_j) V_x^{\text{HF}}[\phi_p](r_j) \, dr_j,
$$
where $V_x^{\text{HF}}[\phi_p]$ is the exact exchange potential acting on orbital $\phi_p$, computed via `compute_exchange_potentials`.

#### 8.2. Zero-Point Shift Correction

The OEP exchange potential is corrected to match the HOMO expectation value:
$$
  V_x^{\text{OEP}} \leftarrow V_x^{\text{OEP}} + \Delta,
$$
where
$$
  \Delta = \langle \phi_{\text{HOMO}} | V_x^{\text{HF}} | \phi_{\text{HOMO}} \rangle - \langle \phi_{\text{HOMO}} | V_x^{\text{OEP}} | \phi_{\text{HOMO}} \rangle.
$$
This ensures that the HOMO energy from OEP matches the exact exchange HOMO energy, which is important for ionization potential consistency.

---

### 9. Self-Consistent Workflow

1. **Inner loop**
   - Solve the KS eigenproblem with current potentials.
   - Update density through Pulay mixing (history 7, mixing frequency 2).
   - Cache eigenvalues, eigenvectors, and interpolation matrices as needed for HF or RPA steps.

2. **Outer loop**
   - Needed whenever Hartree–Fock exchange or OEP (exchange/correlation) is enabled.
   - After each inner-loop convergence, recompute HF matrices, RPA energies, and OEP potentials, then restart the inner loop.

3. **Convergence**
   - Inner: $\|\rho^{n+1} - \rho^n\|/\|\rho^n\| < 10^{-6}$ (typical).
   - Outer: similar metric with tolerance $\sim 10^{-5}$.

---

### 10. Energy Accounting and Mixing

- Kinetic energy is split into radial ($E_{Ts1}$) and angular ($E_{Ts2}$) contributions.
- Potential components:
  - external ($V_{\text{ext}}$),
  - Hartree ($E_H$),
  - local exchange–correlation ($E_x^{\text{loc}}, E_c^{\text{loc}}$),
  - exact exchange $E_x^{\text{HF}}$,
  - RPA correlation $E_c^{\text{RPA}}$.
- Hybrid mixing rules:
  $$
    E_x = \lambda E_x^{\text{HF}} + (1-\lambda) E_x^{\text{GGA}},
  $$
  $$
    E_c = \lambda^3 E_c^{\text{RPA}} + (1-\lambda^3) E_c^{\text{GGA}}.
  $$
  Here $\lambda=1$ for pure HF/RPA, while $\lambda=0.25$ for PBE0-like hybrids (handled automatically via `SwitchesFlags`).

---

### 11. Practical Guidance

- Maintain explicit spin resolution so that the factors $f_p/(2l_p+1)$ remain meaningful.
- Ensure Wigner $3j$ evaluations fulfill triangle and parity rules.
- If $\chi^{\text{OEP}}$ becomes ill-conditioned, switch to least-squares or pseudo-inverse solvers.
- Convergence with respect to radial mesh, frequency nodes, and smoothing cutoff $\omega_c$ should be tested; the log-det integrals are sensitive to these parameters.

#### 11.1. Implementation-Specific Notes

**Dual Grid System**:
- The implementation uses two `RadialOperatorsBuilder` instances:
  - `ops_builder`: standard dense grid for computing $\chi_0$ and driving terms
  - `ops_builder_oep`: sparser OEP grid for solving the linear system
- Both grids must have the same `quadrature_nodes` and `quadrature_weights` (checked in `_check_ops_builder_consistency_at_quadrature_nodes`).
- The `global_interpolation_matrix` $K$ maps from the standard grid to the OEP grid.

**Frequency Grid**:
- The frequency scale parameter $\alpha = 2.5$ (hardcoded in `_initialize_frequency_grid_and_weights`).
- Only the semi-infinite grid $[0, \infty)$ is implemented; the low-frequency smoothing grid mentioned in §2 is not currently used.

**Angular Momentum Cutoff**:
- The `angular_momentum_cutoff` parameter limits the maximum $L$ value in the angular expansion.
- This affects both RPA correlation energy and OEP correlation potential calculations.
- For exchange-only OEP, the angular momentum is determined by the occupied orbitals' $l$ values.

**Numerical Stability**:
- The RPA correlation energy includes a trace term: $\log\det(I - M_L) + \operatorname{Tr}(M_L)$ to improve numerical stability.
- The OEP linear system solver monitors condition numbers and records warnings for ill-conditioned systems.
- **Note**: The correlation potential coefficient is NOT shifted by the HOMO value in the reference implementation (this was removed to match the reference code).
- The boundary condition $-1/r$ is applied for $r \geq 9$ Bohr to enforce correct long-range behavior.

**Orbital Product Computations**:
- The implementation uses efficient einsum operations for orbital products:
  - `orbital_pair_product`: $\Phi_{pq}(r) = \phi_p(r)\phi_q(r)$ for valid $(p,q)$ pairs (occ-virt)
  - `orbital_product_outer`: 3D array $\phi_p(r) \otimes \phi_q(r)$ for all pairs, used for self-energy computation
  - `orbital_squared_diff`: $\phi_p^2(r) - \phi_q^2(r)$ for $Q_{2c}$ occ-virt term
  - `occ_eigenvectors_prod`: $\phi_p(r)\phi_{p'}(r)$ for occ-occ pairs, used in occ-occ coupling terms
  - `occ_eigenvectors_square`: $\phi_p^2(r)$ for occ-occ pairs in $Q_{2c}$ term

**Occ-Occ Coupling**:
- The occ-occ terms are essential for fractional occupations and are computed separately from occ-virt terms.
- Occ-occ coupling constant: $C_{pp'}^{\text{occ-occ}} = f_p(2l_{p'}+1) - f_{p'}(2l_p+1)$.
- Occ-occ terms contribute to:
  1. The RPA response kernel $M_L(\omega)$ (via `select_common_constants_Kl_double_dash_term1`)
  2. The $Q_{2c}$ driving term (via `select_common_constants_term2`)
  3. The self-energy $\Sigma_c$ (indirectly through the modified response kernel)

---

This structured summary preserves the original formulas and intent while clarifying notation, separating orbital and radial indices, and explicitly documenting the roles of $Q_{1c}$, $Q_{2c}$, and $\widehat{\Sigma}_c$ for the OEP implementation.

