from dataclasses import dataclass
import math

@dataclass
class CruiseParams:
    m: float = 1500.0
    Fmax: float = 4000.0
    rho: float = 1.2
    Cd: float = 0.30
    A: float = 2.4
    Cr: float = 0.012
    g: float = 9.81

def f_dot_v(v: float, u: float, theta: float, p: CruiseParams) -> float:
    F_trac = u * p.Fmax
    F_drag = 0.5 * p.rho * p.Cd * p.A * v**2
    F_roll = p.Cr * p.m * p.g
    F_slope = p.m * p.g * math.sin(theta)
    return (F_trac - F_drag - F_roll - F_slope) / p.m

def equilibrium_for(v_star: float, theta: float, p: CruiseParams):
    F_drag = 0.5 * p.rho * p.Cd * p.A * v_star**2
    F_roll = p.Cr * p.m * p.g
    F_slope = p.m * p.g * math.sin(theta)
    required = F_drag + F_roll + F_slope
    u_star = required / p.Fmax
    feasible = 0.0 <= u_star <= 1.0
    return {"v_star": v_star, "theta": theta, "u_star": u_star, "feasible": feasible, "forces_N": {
        "drag": F_drag, "roll": F_roll, "slope": F_slope, "required": required
    }}

def linearize_at(v_star: float, p: CruiseParams):
    A = -(p.rho * p.Cd * p.A * v_star) / p.m
    B = p.Fmax / p.m
    C = 1.0
    D = 0.0
    return {"A": A, "B": B, "C": C, "D": D}

if __name__ == "__main__":
    # Pequeña demo: 90 km/h en llano
    p = CruiseParams()
    v_star = 90/3.6
    eq = equilibrium_for(v_star, 0.0, p)
    lin = linearize_at(v_star, p)
    print("Equilibrio:", eq)
    print("Linearización (A,B,C,D):", lin)
