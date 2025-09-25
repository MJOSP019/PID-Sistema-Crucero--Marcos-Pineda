# app.py — Dashboard de Control Crucero (PID) con comparación A/B
import math
from dataclasses import dataclass
from typing import Tuple
import pandas as pd
import numpy as np
import streamlit as st
import plotly.graph_objects as go
import matplotlib.pyplot as plt
import control as ct

from modelo_crucero import CruiseParams, linearize_at, equilibrium_for

# ======================
# Simulación + métricas
# ======================

@dataclass
class SimConfig:
    v_star_kmh: float = 90.0
    theta_deg: float = 0.0
    dt: float = 0.01
    t_end: float = 60.0
    ref_step: float = 1.0
    u_min: float = 0.0
    u_max: float = 1.0
    tau_aw: float = 0.5

@dataclass
class PID:
    Kp: float
    Ki: float
    Kd: float = 0.0

def simulate_with_aw(pid: PID, cfg: SimConfig, params: CruiseParams = CruiseParams()) -> Tuple[np.ndarray, np.ndarray, np.ndarray]:
    """Simula lazo cerrado (linealizado) con saturación y anti-windup. Devuelve (t, y, u_real)."""
    v_star = cfg.v_star_kmh / 3.6
    theta = np.deg2rad(cfg.theta_deg)

    lin = linearize_at(v_star, params)
    A, B = lin["A"], lin["B"]

    # u* del equilibrio (depende de pendiente)
    eq = equilibrium_for(v_star, theta, params)
    u_star = eq["u_star"]

    t = np.arange(0.0, cfg.t_end + cfg.dt, cfg.dt)
    x = 0.0    # x = v - v*
    I = 0.0

    y_hist, u_hist = [], []
    ref = cfg.ref_step

    for _ in t:
        y = x
        e = ref - y

        # PID ideal sobre u_tilde (omitimos D discreto en este demo)
        u_tilde_unsat = pid.Kp * e + I + pid.Kd * 0.0

        # Entrada real y saturación
        u_real_unsat = u_star + u_tilde_unsat
        u_real_sat = min(cfg.u_max, max(cfg.u_min, u_real_unsat))
        u_tilde_sat = u_real_sat - u_star

        # Anti-windup (back-calculation)
        I += (pid.Ki * e + (u_tilde_sat - u_tilde_unsat) / cfg.tau_aw) * cfg.dt

        # Planta linealizada
        x += (A * x + B * u_tilde_sat) * cfg.dt

        y_hist.append(y)
        u_hist.append(u_real_sat)

    return t, np.array(y_hist), np.array(u_hist)

def trapezoid(y: np.ndarray, x: np.ndarray) -> float:
    """Integral numérica (similar a scipy.trapezoid, usando NumPy)."""
    return float(np.trapezoid(y, x))

def compute_metrics(t: np.ndarray, y: np.ndarray, u: np.ndarray, ref: float = 1.0):
    e = ref - y
    overshoot = max(0.0, (np.max(y) - ref) / max(ref, 1e-12) * 100.0)

    band = 0.02 * max(abs(ref), 1e-12)
    idx_outside = np.where(np.abs(y - ref) > band)[0]
    if len(idx_outside) == 0:
        ts = 0.0
    else:
        last_out = idx_outside[-1]
        ts = t[min(last_out + 1, len(t) - 1)]

    ess = float(ref - y[-1])
    IAE = trapezoid(np.abs(e), t)
    ITAE = trapezoid(t * np.abs(e), t)
    energy = trapezoid(u**2, t)
    return {
        "overshoot_pct": float(overshoot),
        "settling_time_s": float(ts),
        "ess": float(ess),
        "IAE": float(IAE),
        "ITAE": float(ITAE),
        "control_energy": float(energy),
        "u_peak": float(np.max(u)),
        "u_min": float(np.min(u)),
        "u_max": float(np.max(u)),
    }

# ======================
# Estado para comparación A/B
# ======================

st.set_page_config(page_title="Control Crucero - Dashboard", layout="wide")

if "ctrlA" not in st.session_state:
    st.session_state.ctrlA = None  # dict con {"name","pid","t","y","u","metrics"}
if "ctrlB" not in st.session_state:
    st.session_state.ctrlB = None

def make_result_dict(name: str, pid: PID, t, y, u, metrics):
    return {"name": name, "pid": pid, "t": t, "y": y, "u": u, "metrics": metrics}

# ======================
# Sidebar (parámetros)
# ======================

st.title("🚗 Dashboard de Control Crucero (PID)")

with st.expander("¿Qué estoy viendo? (Explicación corta)", expanded=True):
    st.markdown(
        """
**Objetivo**: mantener la velocidad (alrededor de un punto de operación) usando un **PID**.  
Con los **sliders** podrás cambiar la pendiente, los parámetros del auto y las ganancias del PID y ver **en vivo**:
- La **respuesta** de la velocidad (desvío respecto a la velocidad objetivo): paso de 1 m/s.
- La **señal de control** (acelerador real) saturada en `[0, 1]`.
- **Métricas**: *overshoot*, *tiempo de establecimiento*, *IAE*, *ITAE*, y **energía de control**.
        """
    )

st.sidebar.header("Parámetros del vehículo")
m = st.sidebar.slider("Masa m [kg]", 800.0, 2000.0, 1500.0, 50.0)
Fmax = st.sidebar.slider("Fuerza máxima Fmax [N]", 1000.0, 8000.0, 4000.0, 100.0)
rho = st.sidebar.slider("Densidad aire ρ [kg/m³]", 1.0, 1.4, 1.2, 0.01)
Cd = st.sidebar.slider("Coef. arrastre Cd [-]", 0.20, 0.45, 0.30, 0.01)
Area = st.sidebar.slider("Área frontal A [m²]", 1.5, 3.5, 2.4, 0.1)
Cr = st.sidebar.slider("Coef. rodadura Cr [-]", 0.005, 0.03, 0.012, 0.001)

st.sidebar.header("Punto operativo")
v_star_kmh = st.sidebar.slider("Velocidad objetivo v* [km/h]", 40.0, 130.0, 90.0, 1.0)
theta_deg = st.sidebar.slider("Pendiente θ [°]", -10.0, 10.0, 0.0, 0.1)

st.sidebar.header("Controlador PID")
Kp = st.sidebar.slider("Kp", 0.0, 2.0, 0.36, 0.01)
Ki = st.sidebar.slider("Ki", 0.0, 1.0, 0.184, 0.001)
Kd = st.sidebar.slider("Kd", 0.0, 0.1, 0.02, 0.001)

st.sidebar.header("Simulación")
t_end = st.sidebar.slider("Duración [s]", 5.0, 120.0, 60.0, 1.0)
dt = st.sidebar.select_slider("dt [s]", options=[0.001, 0.005, 0.01, 0.02, 0.05], value=0.01)
ref_step = st.sidebar.selectbox("Tipo de prueba", ["Escalón 1 m/s"], index=0)
tau_aw = st.sidebar.slider("Anti-windup τaw", 0.1, 2.0, 0.5, 0.1)

# Construir objetos
params = CruiseParams(m=m, Fmax=Fmax, rho=rho, Cd=Cd, A=Area, Cr=Cr, g=9.81)
cfg = SimConfig(v_star_kmh=v_star_kmh, theta_deg=theta_deg, dt=dt, t_end=t_end, ref_step=1.0, tau_aw=tau_aw)
pid = PID(Kp=Kp, Ki=Ki, Kd=Kd)

# ======================
# TABS
# ======================
tab1, tab2, tab3, tab4, tab5 = st.tabs(
    ["🎛️ Simulación (PID)", "📈 Análisis de planta", "📚 Modelo matemático", "⚖️ Comparación", "🛡️ Robustez"])


# ---------------------------------------------------------------------------------
# TAB 1: Simulación (PID)
# ---------------------------------------------------------------------------------
with tab1:
    # Simular
    t, y, u = simulate_with_aw(pid, cfg, params)
    metrics = compute_metrics(t, y, u, ref=cfg.ref_step)

    col1, col2 = st.columns([2, 1])

    with col1:
        # Respuesta (v - v*)
        fig_resp = go.Figure()
        fig_resp.add_trace(go.Scatter(x=t, y=y, mode="lines", name="Salida (v - v*)"))
        fig_resp.add_hline(y=1.0, line=dict(dash="dash", width=1), annotation_text="Referencia = 1 m/s")
        fig_resp.update_layout(
            title="Respuesta al escalón (salida = v - v*)",
            xaxis_title="Tiempo [s]",
            yaxis_title="m/s (desvío)",
            height=380,
        )
        st.plotly_chart(fig_resp, use_container_width=True)

        # Señal de control u(t)
        fig_u = go.Figure()
        fig_u.add_trace(go.Scatter(x=t, y=u, mode="lines", name="u(t) real"))
        fig_u.add_hline(y=0.0, line=dict(dash="dot", width=1))
        fig_u.add_hline(y=1.0, line=dict(dash="dot", width=1), annotation_text="Saturación [0,1]")
        fig_u.update_layout(
            title="Señal de control u(t) con saturación [0,1]",
            xaxis_title="Tiempo [s]",
            yaxis_title="u (real)",
            height=320,
        )
        st.plotly_chart(fig_u, use_container_width=True)

    with col2:
        st.subheader("Métricas")
        c1, c2 = st.columns(2)
        c1.metric("Overshoot", f"{metrics['overshoot_pct']:.1f} %")
        c2.metric("t_establecimiento", f"{metrics['settling_time_s']:.2f} s")
        c1.metric("IAE", f"{metrics['IAE']:.3f}")
        c2.metric("ITAE", f"{metrics['ITAE']:.3f}")
        c1.metric("Energía control", f"{metrics['control_energy']:.3f}")
        c2.metric("u_peak", f"{metrics['u_peak']:.3f}")

        st.divider()
        st.subheader("Equilibrio en el punto operativo")
        v_star = v_star_kmh / 3.6
        eq = equilibrium_for(v_star, math.radians(theta_deg), params)
        st.write(f"**u\\*** (acelerador para sostener v*): `{eq['u_star']:.3f}` (0..1)")
        st.caption("En equilibrio, fuerza del motor = arrastre + rodadura + componente de peso en pendiente.")

    st.divider()
    cA, cB = st.columns(2)
    with cA:
        if st.button("💾 Guardar como Controlador A (manual)", use_container_width=True):
            st.session_state.ctrlA = make_result_dict("A (manual)", pid, t, y, u, metrics)
            st.success("Guardado A ✅")
    with cB:
        if st.button("💾 Guardar como Controlador B (actual)", use_container_width=True):
            st.session_state.ctrlB = make_result_dict("B (actual)", pid, t, y, u, metrics)
            st.success("Guardado B ✅")

# ---------------------------------------------------------------------------------
# TAB 2: Análisis de planta
# ---------------------------------------------------------------------------------
with tab2:
    st.markdown("La planta linealizada alrededor de v* es de **1er orden**:  \n"
                r"$G(s)=\dfrac{B}{s+a}$ con $a=-A$ y $B = \frac{F_{\max}}{m}$ evaluados en el punto operativo.")
    lin = linearize_at(v_star_kmh/3.6, params)
    A, B, C, D = lin["A"], lin["B"], 1.0, 0.0
    sys = ct.ss(A, B, C, D)
    G = ct.ss2tf(sys)

    c1, c2 = st.columns(2)

    # Paso
    with c1:
        fig = plt.figure()
        t_s, y_s = ct.step_response(sys)
        plt.plot(t_s, y_s)
        plt.grid(True); plt.title("Paso - Planta linealizada")
        plt.xlabel("Tiempo [s]"); plt.ylabel("m/s (desvío)")
        st.pyplot(fig, clear_figure=True)

    # Impulso
    with c2:
        fig = plt.figure()
        t_i, y_i = ct.impulse_response(sys)
        plt.plot(t_i, y_i)
        plt.grid(True); plt.title("Impulso - Planta linealizada")
        plt.xlabel("Tiempo [s]"); plt.ylabel("m/s (desvío)")
        st.pyplot(fig, clear_figure=True)

    # Bode
    c3, c4 = st.columns(2)
    with c3:
        fig = plt.figure()
        ct.bode_plot(sys, dB=False)  # magnitud |G(jw)|
        plt.suptitle("Bode - Planta", y=1.02)
        st.pyplot(fig, clear_figure=True)

    # Root Locus
    with c4:
        fig = plt.figure()
        ct.root_locus(G, plot=True, grid=True)
        plt.title("Root Locus - Planta")
        st.pyplot(fig, clear_figure=True)

    st.info(f"G(s) = {G}")

# ---------------------------------------------------------------------------------
# TAB 3: Modelo matemático
# ---------------------------------------------------------------------------------
with tab3:
    st.markdown("**Ecuación dinámica no lineal (balance de fuerzas):**")
    st.latex(r"""
    m\,\dot v(t) \;=\; F_{\text{trac}}(u) \;-\; 
    \tfrac{1}{2}\rho C_d A\,v(t)^2 \;-\; C_r\,m\,g \;-\; m\,g\,\sin(\theta)
    """)
    st.markdown("Donde asumimos $F_{\\text{trac}}(u) = u\\,F_{\\max}$, con $0\\le u\\le 1$.")
    st.markdown("**Forma de estado (no lineal):**")
    st.latex(r"""
    \dot v(t) \;=\; \frac{1}{m}\left[\,u(t)\,F_{\max}
    - \tfrac{1}{2}\rho C_d A\,v(t)^2 - C_r\,m\,g - m\,g\,\sin(\theta)\right]
    """)
    st.markdown("**Punto de equilibrio** $(\dot v=0)$:")
    st.latex(r"""
    u^* F_{\max} \;=\; \tfrac{1}{2}\rho C_d A\,(v^*)^2 \;+\; C_r\,m\,g \;+\; m\,g\,\sin(\theta)
    """)
    st.markdown("**Linealización** alrededor de $(v^*,u^*)$ (con salida $y=v$):")
    st.latex(r"""
    \tilde{\dot v} = A\,\tilde v + B\,\tilde u,\quad
    y = \tilde v,\quad
    A = -\frac{\rho C_d A\,v^*}{m},\quad
    B = \frac{F_{\max}}{m}
    """)
    st.caption("Interpretación: a mayor velocidad v*, el término de arrastre aumenta → A más negativo (planta más lenta/‘amortiguada’).")

# ---------------------------------------------------------------------------------
# TAB 4: Comparación de controladores (A vs B) y optimización simple
# ---------------------------------------------------------------------------------
with tab4:
    st.markdown("Compara dos sintonías **A** y **B** (guárdalas en la pestaña de Simulación). "
                "Opcional: obtén un **B optimizado** con búsqueda aleatoria que minimiza una función de costo.")

    # Panel de optimización rápida (random search sin librerías extra)
    with st.expander("🔧 Optimización por búsqueda aleatoria (minimiza ITAE + λ · Energía de control)", expanded=False):
        c1, c2, c3 = st.columns(3)
        with c1:
            n_samples = st.number_input("Evaluaciones (N)", min_value=20, max_value=2000, value=200, step=20)
            lam = st.number_input("λ (peso energía)", min_value=0.0, max_value=2.0, value=0.1, step=0.05)
        with c2:
            kp_min = st.number_input("Kp min", value=0.0, step=0.01)
            kp_max = st.number_input("Kp max", value=1.0, step=0.01)
            ki_min = st.number_input("Ki min", value=0.0, step=0.001, format="%.3f")
            ki_max = st.number_input("Ki max", value=1.0, step=0.001, format="%.3f")
        with c3:
            kd_min = st.number_input("Kd min", value=0.0, step=0.001, format="%.3f")
            kd_max = st.number_input("Kd max", value=0.1, step=0.001, format="%.3f")
            sim_T = st.slider("Duración optimización [s]", 5.0, 120.0, min(t_end, 40.0), 1.0)

        def cost_fun(m, lam_):
            return m["ITAE"] + lam_ * m["control_energy"]

        if st.button("🏁 Optimizar B (reemplazar)", type="primary"):
            cfg_opt = SimConfig(
                v_star_kmh=v_star_kmh, theta_deg=theta_deg,
                dt=dt, t_end=sim_T, ref_step=1.0, tau_aw=tau_aw
            )
            params_opt = params
            best = None
            bestJ = float("inf")

            rng = np.random.default_rng()
            for _ in range(int(n_samples)):
                kp = rng.uniform(kp_min, kp_max)
                ki = rng.uniform(ki_min, ki_max)
                kd = rng.uniform(kd_min, kd_max)
                cand = PID(kp, ki, kd)
                tt, yy, uu = simulate_with_aw(cand, cfg_opt, params_opt)
                mm = compute_metrics(tt, yy, uu, ref=1.0)
                J = cost_fun(mm, lam)
                if J < bestJ and not np.isnan(J):
                    best = (cand, tt, yy, uu, mm, J)
                    bestJ = J

            if best is None:
                st.error("No se encontró candidato válido.")
            else:
                cand, tt, yy, uu, mm, J = best
                st.session_state.ctrlB = make_result_dict("B (optimizado)", cand, tt, yy, uu, mm)
                st.success(f"Listo: B=({cand.Kp:.3f}, {cand.Ki:.3f}, {cand.Kd:.3f}) | J={J:.3f}")

    st.divider()

    # Mostrar selección A/B
    a_ok = st.session_state.ctrlA is not None
    b_ok = st.session_state.ctrlB is not None

    info_cols = st.columns(2)
    with info_cols[0]:
        st.subheader("Controlador A")
        if a_ok:
            pa = st.session_state.ctrlA["pid"]
            st.write(f"**A**: Kp={pa.Kp:.3f}, Ki={pa.Ki:.3f}, Kd={pa.Kd:.3f}")
        else:
            st.info("Guarda A en la pestaña de **Simulación**.")
    with info_cols[1]:
        st.subheader("Controlador B")
        if b_ok:
            pb = st.session_state.ctrlB["pid"]
            st.write(f"**B**: Kp={pb.Kp:.3f}, Ki={pb.Ki:.3f}, Kd={pb.Kd:.3f}")
        else:
            st.info("Guarda B (o usa la optimización) en esta pestaña.")

    if not (a_ok and b_ok):
        st.stop()

    # Re-simular A y B con condiciones actuales del sidebar (comparación justa)
    cfg_cmp = SimConfig(v_star_kmh=v_star_kmh, theta_deg=theta_deg, dt=dt, t_end=t_end, ref_step=1.0, tau_aw=tau_aw)
    resA_pid = st.session_state.ctrlA["pid"]
    resB_pid = st.session_state.ctrlB["pid"]
    tA, yA, uA = simulate_with_aw(resA_pid, cfg_cmp, params)
    tB, yB, uB = simulate_with_aw(resB_pid, cfg_cmp, params)
    mA = compute_metrics(tA, yA, uA)
    mB = compute_metrics(tB, yB, uB)

    # Gráficas superpuestas
    g1, g2 = st.columns(2)
    with g1:
        fig_resp_cmp = go.Figure()
        fig_resp_cmp.add_trace(go.Scatter(x=tA, y=yA, mode="lines", name="A (salida)"))
        fig_resp_cmp.add_trace(go.Scatter(x=tB, y=yB, mode="lines", name="B (salida)"))
        fig_resp_cmp.add_hline(y=1.0, line=dict(dash="dash", width=1), annotation_text="Ref=1 m/s")
        fig_resp_cmp.update_layout(title="Respuesta (A vs B)", xaxis_title="Tiempo [s]", yaxis_title="m/s (desvío)", height=360)
        st.plotly_chart(fig_resp_cmp, use_container_width=True)
    with g2:
        fig_u_cmp = go.Figure()
        fig_u_cmp.add_trace(go.Scatter(x=tA, y=uA, mode="lines", name="A: u(t)"))
        fig_u_cmp.add_trace(go.Scatter(x=tB, y=uB, mode="lines", name="B: u(t)"))
        fig_u_cmp.add_hline(y=0.0, line=dict(dash="dot", width=1))
        fig_u_cmp.add_hline(y=1.0, line=dict(dash="dot", width=1), annotation_text="Saturación [0,1]")
        fig_u_cmp.update_layout(title="Señal de control (A vs B)", xaxis_title="Tiempo [s]", yaxis_title="u (real)", height=360)
        st.plotly_chart(fig_u_cmp, use_container_width=True)

    # Métricas lado a lado
    st.subheader("Métricas comparadas")
    tA1, tA2, tA3 = st.columns(3)
    tB1, tB2, tB3 = st.columns(3)

    tA1.metric("A · Overshoot", f"{mA['overshoot_pct']:.1f} %")
    tA2.metric("A · t_est", f"{mA['settling_time_s']:.2f} s")
    tA3.metric("A · Energía", f"{mA['control_energy']:.3f}")

    tB1.metric("B · Overshoot", f"{mB['overshoot_pct']:.1f} %")
    tB2.metric("B · t_est", f"{mB['settling_time_s']:.2f} s")
    tB3.metric("B · Energía", f"{mB['control_energy']:.3f}")

    st.caption("Sugerencia: ajusta λ y el rango de Kp/Ki/Kd en la optimización para explorar distintos compromisos entre rapidez y esfuerzo.")
# ---------------------------------------------------------------------------------
# TAB 5: Robustez / Sensibilidad (A vs B bajo escenarios)
# ---------------------------------------------------------------------------------
with tab5:
    st.markdown("Analiza la **robustez** de los controladores A y B al variar parámetros del vehículo y la pendiente.")

    # ------------- Configuración de escenarios -------------
    st.subheader("Escenarios")
    c1, c2, c3 = st.columns(3)
    with c1:
        mass_delta = st.slider("Variación de masa ± [%]", 0, 50, 20, 5)
        rho_delta  = st.slider("Variación de arrastre (ρ·Cd·A) ± [%]", 0, 50, 20, 5)
    with c2:
        slope_fix = st.slider("Escenarios con pendiente ± [°]", 0.0, 10.0, 5.0, 0.5)
        sim_T_rb  = st.slider("Duración de simulación [s]", 5.0, 120.0, min(60.0, t_end), 1.0)
    with c3:
        add_nominal = st.checkbox("Incluir escenario nominal", value=True)
        show_u_plot = st.checkbox("Mostrar u(t) además de la respuesta", value=False)

    # Requiere A y B
    if (st.session_state.ctrlA is None) or (st.session_state.ctrlB is None):
        st.info("Guarda primero **A** en la pestaña *Simulación* y genera/guarda **B** (manual u optimizado) para correr robustez.")
        st.stop()

    # Construye lista de escenarios
    scenarios = []
    if add_nominal:
        scenarios.append({"name": "Nominal", "m_scale": 1.0, "drag_scale": 1.0, "theta_deg": theta_deg})

    scenarios.extend([
        {"name": f"m +{mass_delta}%", "m_scale": 1.0 + mass_delta/100.0, "drag_scale": 1.0, "theta_deg": theta_deg},
        {"name": f"m -{mass_delta}%", "m_scale": 1.0 - mass_delta/100.0, "drag_scale": 1.0, "theta_deg": theta_deg},
        {"name": f"drag +{rho_delta}%", "m_scale": 1.0, "drag_scale": 1.0 + rho_delta/100.0, "theta_deg": theta_deg},
        {"name": f"drag -{rho_delta}%", "m_scale": 1.0, "drag_scale": 1.0 - rho_delta/100.0, "theta_deg": theta_deg},
        {"name": f"pend +{slope_fix}°", "m_scale": 1.0, "drag_scale": 1.0, "theta_deg": theta_deg + slope_fix},
        {"name": f"pend -{slope_fix}°", "m_scale": 1.0, "drag_scale": 1.0, "theta_deg": theta_deg - slope_fix},
    ])

    # Utilidad para escalar parámetros
    def scale_params(base: CruiseParams, m_scale=1.0, drag_scale=1.0):
        return CruiseParams(
            m=base.m * m_scale,
            Fmax=base.Fmax,  # dejamos Fmax fijo
            rho=base.rho * drag_scale,
            Cd=base.Cd,      # puedes repartir el factor en rho/Cd/A: aquí lo aplicamos a rho para simplificar
            A=base.A,
            Cr=base.Cr,
            g=base.g
        )

    # PIDs a evaluar
    pidA = st.session_state.ctrlA["pid"]
    pidB = st.session_state.ctrlB["pid"]

    # Simular todos los escenarios para A y B
    cfg_rb = SimConfig(v_star_kmh=v_star_kmh, theta_deg=0.0, dt=dt, t_end=sim_T_rb, ref_step=1.0, tau_aw=tau_aw)

    results_rows = []
    # Gráficas: respuesta
    gA, gB = st.columns(2)
    fig_A = go.Figure(); fig_B = go.Figure()
    figu_A = go.Figure(); figu_B = go.Figure()

    for sc in scenarios:
        p_sc = scale_params(params, sc["m_scale"], sc["drag_scale"])
        cfg_rb.theta_deg = sc["theta_deg"]

        # A
        tA, yA, uA = simulate_with_aw(pidA, cfg_rb, p_sc)
        mA = compute_metrics(tA, yA, uA)
        fig_A.add_trace(go.Scatter(x=tA, y=yA, mode="lines", name=sc["name"]))
        figu_A.add_trace(go.Scatter(x=tA, y=uA, mode="lines", name=sc["name"]))

        # B
        tB, yB, uB = simulate_with_aw(pidB, cfg_rb, p_sc)
        mB = compute_metrics(tB, yB, uB)
        fig_B.add_trace(go.Scatter(x=tB, y=yB, mode="lines", name=sc["name"]))
        figu_B.add_trace(go.Scatter(x=tB, y=uB, mode="lines", name=sc["name"]))

        # Guardar métricas en tabla
        results_rows.append({
            "Escenario": sc["name"],
            "A_overshoot_%": round(mA["overshoot_pct"], 1),
            "A_t_est_s": round(mA["settling_time_s"], 2),
            "A_IAE": round(mA["IAE"], 3),
            "A_Energía": round(mA["control_energy"], 3),
            "B_overshoot_%": round(mB["overshoot_pct"], 1),
            "B_t_est_s": round(mB["settling_time_s"], 2),
            "B_IAE": round(mB["IAE"], 3),
            "B_Energía": round(mB["control_energy"], 3),
        })

    with gA:
        fig_A.add_hline(y=1.0, line=dict(dash="dash", width=1), annotation_text="Ref=1 m/s")
        fig_A.update_layout(title="Respuesta A (varios escenarios)", xaxis_title="Tiempo [s]", yaxis_title="m/s (desvío)", height=360)
        st.plotly_chart(fig_A, use_container_width=True)
        if show_u_plot:
            figu_A.add_hline(y=0.0, line=dict(dash="dot", width=1))
            figu_A.add_hline(y=1.0, line=dict(dash="dot", width=1), annotation_text="Saturación [0,1]")
            figu_A.update_layout(title="u(t) A (varios escenarios)", xaxis_title="Tiempo [s]", yaxis_title="u", height=300)
            st.plotly_chart(figu_A, use_container_width=True)

    with gB:
        fig_B.add_hline(y=1.0, line=dict(dash="dash", width=1), annotation_text="Ref=1 m/s")
        fig_B.update_layout(title="Respuesta B (varios escenarios)", xaxis_title="Tiempo [s]", yaxis_title="m/s (desvío)", height=360)
        st.plotly_chart(fig_B, use_container_width=True)
        if show_u_plot:
            figu_B.add_hline(y=0.0, line=dict(dash="dot", width=1))
            figu_B.add_hline(y=1.0, line=dict(dash="dot", width=1), annotation_text="Saturación [0,1]")
            figu_B.update_layout(title="u(t) B (varios escenarios)", xaxis_title="Tiempo [s]", yaxis_title="u", height=300)
            st.plotly_chart(figu_B, use_container_width=True)

    # ---------- Tabla de métricas comparadas ----------
    st.subheader("Métricas por escenario (A vs B)")
    df_rb = pd.DataFrame(results_rows)
    st.dataframe(df_rb, use_container_width=True)

    st.caption(
        "Interpretación sugerida: un controlador **más robusto** mantiene overshoot y tiempo de establecimiento "
        "en rangos razonables cuando cambian masa, pendiente o arrastre. Compara cómo escalan A y B."
    )
