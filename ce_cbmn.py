"""
================================================================================
CE-CBMN: Co-Evolutionary Common Biological Market Networks
================================================================================
Author      : Adan Facundo
ORCID       : 0009-0003-9110-4744
Version     : 2.1.0
Date        : 2026-05-11
License     : CC BY 4.0 (https://creativecommons.org/licenses/by/4.0/)
Repository  : https://github.com/facundoyamil96/CE-CBMN-Algorithm
Preprint    : 
================================================================================

DESCRIPTION
-----------
Reference Python implementation of the CE-CBMN algorithm (v2.0 spec),
benchmarked on the 2D Rastrigin function under stochastic spatial perturbations.

This implementation is faithful to the formal specification (v2.0):
  - Strict phi/B decoupling (Genetic Potential vs. Biomass)
  - Independent fungal agents with profitability-based desinvestment
  - Spatially-localized Gaussian perturbations with dual-metric propagation
  - Phenotypic defense D_i activated by stress signal cascades
  - Normalized gradient transfer (biomass, NOT phi)
  - Global Resilience R(t) as convergence criterion

AUDIT NOTES (v2.0 → v2.1 corrections applied)
-----------------------------------------------
[BUG-01 CRITICAL] Rastrigin output not normalized → phi outside [0,1]
[BUG-02 CRITICAL] F (genetic potential) never used in harvest → phi/B decoupling broken
[BUG-03 HIGH]     Transfer formula ignores T_j, P_i, R_i, normalized gradient
[BUG-04 HIGH]     EFFICIENCY_E = 0.1 makes 90% of transfers absorbed by hongo
[BUG-05 HIGH]     No reproduction/mutation (Phase 7 completely absent)
[BUG-06 HIGH]     No Resilience metric R(t) computed
[BUG-07 MEDIUM]   Shock deterministic (always node 0, gen 40) — not stochastic
[BUG-08 MEDIUM]   Wave 3 only 1 hop, no BFS with topological decay exp(-lambda*d_T)
[BUG-09 MEDIUM]   No spatial positions (cx, cy) separate from solution (sx, sy)
[BUG-10 LOW]      BETA_BONUS makes B saturate at 1.0 in 6 gens regardless of phi

================================================================================
"""

import numpy as np
import random
import math
from dataclasses import dataclass, field
from typing import List, Optional, Tuple
from collections import deque

# ─── REPRODUCIBILITY ─────────────────────────────────────────────────────────
SEED = 42
np.random.seed(SEED)
random.seed(SEED)

# ─── HYPERPARAMETERS (v2.0 spec) ─────────────────────────────────────────────
@dataclass
class Config:
    # Population
    num_nodes: int   = 40
    generations: int = 150
    K_max: int       = 3          # Initial kin modules

    # Solution space (Rastrigin domain)
    sol_range: float = 5.12

    # Spatial plane (separate from solution space)
    # Nodes have fixed positions in [0,1]^2 for perturbation geometry
    plane_size: float = 1.0

    # Biomass
    harvest_rate: float  = 0.08   # η: biomass harvested per unit of phi
    B_max: float         = 2.0
    B_init_min: float    = 0.3
    B_init_max: float    = 0.7
    C_node: float        = 0.004  # Maintenance cost per generation
    B_critical: float    = 0.04   # Death threshold

    # Fungal network
    conn_prob: float     = 0.12   # Initial connection probability
    conn_radius: float   = 0.40   # Max spatial distance for initial connections
    C_fungus: float      = 0.003  # Fungal maintenance cost per active connection
    delta_scale: float   = 0.10   # τ_cut scale: τ = ceil((1-A)/delta)
    beta_kin: float      = 0.72   # Kin solidarity: retention relaxation factor
    max_deg_factor: float= 8.0    # deg_max(n) = ceil(P_i * max_deg_factor)

    # Stress signaling
    sigma_stress: float  = 0.12   # Stress detection threshold (biomass drop)
    lambda_decay: float  = 0.65   # Topological signal decay exp(-lambda * hops)
    theta_perc: float    = 0.08   # Minimum signal to activate defense
    alpha_defense: float = 0.30   # Defense boost per signal unit
    gamma_decay: float   = 0.92   # Defense decay per generation (no signal)
    max_hops: int        = 5      # Max BFS depth for signal propagation

    # Survival
    tau_isolated: int    = 5      # Gens isolated before death
    tau_parasite: int    = 7      # Gens as chronic parasite before death

    # Reproduction
    N_children: int      = 4      # New nodes per generation
    mu_base: float       = 0.025  # Base mutation rate (solution space)
    sigma_pos: float     = 0.04   # Offspring spatial dispersion
    Omega: float         = 8.0    # Rescue mutation amplifier
    epsilon_rescue: float= 0.08   # Biomass proximity to critical triggering rescue
    p_wind: float        = 0.12   # Lévy wind dispersal probability
    mu_K: float          = 0.012  # Kin speciation probability

    # Fungal expansion
    Pi_expand: float     = 0.015  # Profitability threshold to spawn child fungus
    p_expand: float      = 0.25   # Probability of expansion when profitable

    # Perturbations
    sigma_global: float  = 0.015  # Global noise σ (solution space)
    p_epicenter: float   = 0.10   # Probability of random epicenter per generation

    # Resilience weights (w1+w2+w3+w4 = 1)
    w1: float = 0.30  # Mean performance Φ
    w2: float = 0.30  # Topological diversity D
    w3: float = 0.20  # Structural connectivity C
    w4: float = 0.20  # Recovery capacity A
    W_recovery: int = 10  # Window for recovery metric


CFG = Config()


# ─── BENCHMARK FUNCTION ───────────────────────────────────────────────────────

def rastrigin_2d(sx: float, sy: float) -> float:
    """
    Standard 2D Rastrigin function.
    Global minimum = 0 at (0, 0).
    Approximate maximum ≈ 80 within [-5.12, 5.12]^2.
    """
    A = 10.0
    return (2 * A
            + sx**2 - A * math.cos(2 * math.pi * sx)
            + sy**2 - A * math.cos(2 * math.pi * sy))


# FIX BUG-01: Normalized phi in [0,1], 1.0 = global optimum (origin)
RASTRIGIN_MAX = 80.0  # Approximate upper bound within domain

def phi_from_solution(sx: float, sy: float) -> float:
    """
    Genetic Potential phi_i in [0, 1].
    phi = 1 means the solution is at the global optimum (0,0).
    phi = 0 means the solution is at the worst possible location.
    """
    return max(0.0, 1.0 - rastrigin_2d(sx, sy) / RASTRIGIN_MAX)


# ─── AGENTS ──────────────────────────────────────────────────────────────────

class Node:
    """
    Node agent: candidate solution n_i = <phi_i, B_i, P_i, R_i, K_i, D_i, x_i>

    KEY DISTINCTION (v2.0 spec):
      sx, sy  — solution coordinates in Rastrigin space (mutate, crossover)
      cx, cy  — fixed spatial position in [0,1]^2 (governs epicenter distance)
      phi     — F(sx,sy): pure mathematical potential, NEVER transferred
      B       — Biomass: operational currency, IS transferred
    """
    _id_counter = 0

    def __init__(self,
                 cx: float, cy: float,
                 sx: float, sy: float,
                 P: float,  R: float,  K: int):
        Node._id_counter += 1
        self.id    = Node._id_counter
        self.cx    = cx          # Fixed spatial x in [0,1]
        self.cy    = cy          # Fixed spatial y in [0,1]
        self.sx    = sx          # Solution x in [-5.12, 5.12]
        self.sy    = sy          # Solution y in [-5.12, 5.12]
        self.phi   = phi_from_solution(sx, sy)  # Genetic potential [0,1]
        self.B     = CFG.B_init_min + random.uniform(0, CFG.B_init_max - CFG.B_init_min)
        self.P     = P           # Processing capacity (0,1]
        self.R     = R           # Retention threshold [0,1)
        self.K     = K           # Kin module
        self.D     = 0.0         # Phenotypic defense [0,1]
        self.role  = 'NEUTRAL'
        self.alive = True
        self.tau_isolated = 0
        self.tau_parasite = 0
        self.B_prev  = self.B    # For stress detection
        self.stress  = 0.0       # Emitted stress signal this generation

    def evaluate(self):
        """Recompute phi from current solution coordinates."""
        self.phi = phi_from_solution(self.sx, self.sy)

    def harvest(self):
        """
        FIX BUG-02: Biomass harvested proportional to phi (genetic potential).
        Good solutions accumulate more biomass; bad solutions may not cover costs.
        """
        self.B = min(CFG.B_max, self.B + CFG.harvest_rate * self.phi - CFG.C_node)

    def apply_damage(self, damage: float):
        """Apply mitigated damage. Defense D_i absorbs fraction of impact."""
        effective = damage * max(0.0, 1.0 - self.D)
        self.B = max(0.0, self.B - effective)

    def decay_defense(self):
        """Defense decays if no stress signal received this generation."""
        if self.stress == 0.0:
            self.D *= CFG.gamma_decay

    @property
    def deg_max(self) -> int:
        """FIX BUG from audit #4: Degree cap prevents mycelial inflation."""
        return max(1, math.ceil(self.P * CFG.max_deg_factor))


class FungalEdge:
    """
    Fungal agent: f_j = <T_j, E_j, A_j, Pi_j>  connecting (n_a, n_b)

    FIX BUG-03: Full transfer formula from v2.0 spec:
      ΔB = min(T_j·P_source, B_source - β·R_source) · gradient_normalized · E_eff
    """
    _id_counter = 0

    def __init__(self, na: Node, nb: Node):
        FungalEdge._id_counter += 1
        self.id     = FungalEdge._id_counter
        self.na     = na
        self.nb     = nb
        self.T      = random.uniform(0.25, 1.0)   # Transfer bandwidth
        self.E      = random.uniform(0.55, 0.95)  # Efficiency (FIX BUG-04: not 0.1)
        self.A      = random.uniform(0.10, 0.85)  # Greed / cut sensitivity
        self.Pi     = 0.0
        self.tau_neg = 0
        self.active = True

    @property
    def tau_cut(self) -> int:
        return max(1, math.ceil((1.0 - self.A) / CFG.delta_scale))

    def process_trade(self, B_max: float, B_min: float):
        """
        Execute one generation of biological market trade.
        Transfers BIOMASS (not phi). Computes profitability.
        """
        if not self.active or not self.na.alive or not self.nb.alive:
            self.active = False
            return

        # Determine source (higher biomass) and sink (lower biomass)
        source = self.na if self.na.B >= self.nb.B else self.nb
        sink   = self.nb if self.na.B >= self.nb.B else self.na

        if source.role == 'SURVIVAL':
            # Source is in survival mode: no export, hongo earns nothing
            self.Pi -= CFG.C_fungus
            self.tau_neg += 1
            if self.tau_neg >= self.tau_cut:
                self.active = False
            return

        # Kin recognition: same module → no efficiency loss, relaxed retention
        kin = (source.K == sink.K)
        E_eff   = 1.0         if kin else self.E
        beta    = CFG.beta_kin if kin else 1.0

        # Normalized gradient
        span = B_max - B_min
        gradient = (source.B - sink.B) / span if span > 1e-6 else 0.0

        # Transfer cap: cannot exceed what source can spare
        excess  = max(0.0, source.B - beta * source.R)
        cap     = min(self.T * source.P, excess)
        delta_B = cap * gradient * E_eff

        # Apply transfer (biomass only, phi untouched)
        source.B = min(CFG.B_max, max(0.0, source.B - delta_B))
        sink.B   = min(CFG.B_max, sink.B + delta_B)

        # Fungal profitability: hongo keeps (1-E)*delta_B minus maintenance
        income = (1.0 - self.E) * delta_B
        self.Pi = income - CFG.C_fungus

        # Desinvestment counter
        if self.Pi < 0:
            self.tau_neg += 1
        else:
            self.tau_neg = 0

        if self.tau_neg >= self.tau_cut:
            self.active = False


# ─── ECOSYSTEM ───────────────────────────────────────────────────────────────

class Ecosystem:
    """Full CE-CBMN ecosystem: G(t) = (N, F, E)"""

    def __init__(self, cfg: Config = CFG):
        self.cfg        = cfg
        self.generation = 0
        self.nodes: List[Node]       = []
        self.fungi: List[FungalEdge] = []
        self.epicenters              = []
        self.hist_R: List[float]     = []
        self._init_population()

    def _init_population(self):
        """Initialize nodes and fungal topology."""
        Node._id_counter      = 0
        FungalEdge._id_counter = 0

        for _ in range(self.cfg.num_nodes):
            cx = random.uniform(0.05, 0.95)
            cy = random.uniform(0.05, 0.95)
            sx = random.uniform(-self.cfg.sol_range, self.cfg.sol_range)
            sy = random.uniform(-self.cfg.sol_range, self.cfg.sol_range)
            P  = random.uniform(0.30, 1.00)
            R  = random.uniform(0.05, 0.35)
            K  = random.randint(1, self.cfg.K_max)
            self.nodes.append(Node(cx, cy, sx, sy, P, R, K))

        # Connect spatially nearby nodes
        n = self.nodes
        for i in range(len(n)):
            for j in range(i + 1, len(n)):
                d = math.dist((n[i].cx, n[i].cy), (n[j].cx, n[j].cy))
                if d < self.cfg.conn_radius and random.random() < self.cfg.conn_prob:
                    deg_i = self._degree(n[i])
                    deg_j = self._degree(n[j])
                    if deg_i < n[i].deg_max and deg_j < n[j].deg_max:
                        self.fungi.append(FungalEdge(n[i], n[j]))

    # ── HELPERS ───────────────────────────────────────────────────────────────

    def _alive(self) -> List[Node]:
        return [n for n in self.nodes if n.alive]

    def _active_fungi(self) -> List[FungalEdge]:
        return [f for f in self.fungi if f.active]

    def _degree(self, node: Node) -> int:
        return sum(1 for f in self.fungi if f.active and (f.na is node or f.nb is node))

    def _neighbors(self, node: Node) -> List[Node]:
        return [
            (f.nb if f.na is node else f.na)
            for f in self.fungi
            if f.active and (f.na is node or f.nb is node)
            and (f.nb if f.na is node else f.na).alive
        ]

    # ── 7-PHASE GENERATION CYCLE ──────────────────────────────────────────────

    def step(self) -> float:
        alive = self._alive()
        if not alive:
            return 0.0

        # Save previous biomass for stress detection
        for n in alive:
            n.B_prev = n.B
            n.stress = 0.0

        # ── Phase 1: Perturbation ──────────────────────────────────────────
        self._phase1_perturbation(alive)

        # ── Phase 2: Harvest ──────────────────────────────────────────────
        for n in alive:
            n.evaluate()   # Update phi from current (possibly perturbed) solution
            n.harvest()    # B += η·phi - C_node  (FIX BUG-02)

        # ── Phase 3: Classify ─────────────────────────────────────────────
        B_vals = [n.B for n in alive]
        B_mean = sum(B_vals) / len(B_vals)
        B_max  = max(B_vals)
        B_min  = min(B_vals)

        for n in alive:
            if n.B <= n.R:
                n.role = 'SURVIVAL'
            elif n.B > B_mean:
                n.role = 'SOURCE'
            elif abs(n.B - B_mean) < 0.03:
                n.role = 'NEUTRAL'
            else:
                n.role = 'SINK'

        # ── Phase 4: Biological market ────────────────────────────────────
        for f in self._active_fungi():
            f.process_trade(B_max, B_min)   # Transfers BIOMASS, not phi (FIX BUG-03)

        # ── Phase 5: Stress signaling ─────────────────────────────────────
        self._phase5_signaling(alive)

        # ── Phase 6: Natural selection ────────────────────────────────────
        self._phase6_selection(alive)

        # ── Phase 7: Reproduction & mutation ─────────────────────────────
        self._phase7_reproduce()

        self.generation += 1
        R = self._resilience()
        self.hist_R.append(R)
        if len(self.hist_R) > 200:
            self.hist_R.pop(0)
        return R

    def _phase1_perturbation(self, alive: List[Node]):
        """
        FIX BUG-07/09: Spatially-localized Gaussian perturbations.
        Damage hits node biomass proportional to epicenter proximity.
        Global noise perturbs solution coordinates.
        """
        sr = self.cfg.sol_range

        # Global noise on solution coordinates
        for n in alive:
            n.sx = np.clip(n.sx + np.random.normal(0, self.cfg.sigma_global * sr), -sr, sr)
            n.sy = np.clip(n.sy + np.random.normal(0, self.cfg.sigma_global * sr), -sr, sr)

        # Active epicenters damage biomass via Gaussian spatial decay
        for ep in self.epicenters:
            for n in alive:
                d = math.dist((n.cx, n.cy), (ep['x'], ep['y']))
                impact = ep['I'] * math.exp(-(d**2) / (2 * ep['rho']**2))
                n.apply_damage(impact * 0.35)
            ep['dur'] -= 1
        self.epicenters = [ep for ep in self.epicenters if ep['dur'] > 0]

        # Stochastic new epicenter
        if random.random() < self.cfg.p_epicenter:
            self.add_epicenter(random.uniform(0.1, 0.9),
                               random.uniform(0.1, 0.9),
                               random.uniform(0.25, 0.60),
                               random.uniform(0.10, 0.25),
                               random.randint(1, 3))

    def add_epicenter(self, x: float, y: float, I: float, rho: float, dur: int):
        """Manually inject a spatial perturbation epicenter."""
        self.epicenters.append({'x': x, 'y': y, 'I': I, 'rho': rho, 'dur': dur})

    def _phase5_signaling(self, alive: List[Node]):
        """
        FIX BUG-08: Full BFS stress propagation with topological decay.
        Activates PHENOTYPIC DEFENSE D_i (not solution mutation) in neighbors.
        """
        for n in alive:
            delta_B = n.B - n.B_prev
            if delta_B < -self.cfg.sigma_stress:
                n.stress = max(0.0, (-delta_B - self.cfg.sigma_stress) / (1.0 - self.cfg.sigma_stress))

        # BFS propagation from all stressed nodes
        for origin in alive:
            if origin.stress <= 0:
                continue
            queue   = deque([(origin, 0, origin.stress)])
            visited = {origin.id}

            while queue:
                cur, hops, sig = queue.popleft()
                if hops >= self.cfg.max_hops:
                    continue
                for nb in self._neighbors(cur):
                    if nb.id in visited:
                        continue
                    s_arrived = sig * math.exp(-self.cfg.lambda_decay * 1)
                    if s_arrived > self.cfg.theta_perc:
                        # Activate phenotypic defense (NOT solution mutation)
                        nb.D = min(1.0, nb.D + self.cfg.alpha_defense * s_arrived)
                        visited.add(nb.id)
                        queue.append((nb, hops + 1, s_arrived))

        # Decay defense for nodes that received no signal
        for n in alive:
            n.decay_defense()

    def _phase6_selection(self, alive: List[Node]):
        """Natural selection: starvation, isolation, chronic parasitism."""
        for n in alive:
            if n.B < self.cfg.B_critical:
                n.alive = False
                continue
            has_conn = any(f.active and (f.na is n or f.nb is n) for f in self.fungi)
            if not has_conn:
                n.tau_isolated += 1
                if n.tau_isolated >= self.cfg.tau_isolated:
                    n.alive = False
                    continue
            else:
                n.tau_isolated = 0

            if n.B < n.R:
                n.tau_parasite += 1
                if n.tau_parasite >= self.cfg.tau_parasite:
                    n.alive = False
            else:
                n.tau_parasite = 0

        # Prune orphaned fungi
        for f in self.fungi:
            if not f.na.alive or not f.nb.alive:
                f.active = False

    def _phase7_reproduce(self):
        """
        FIX BUG-05: Full reproduction + mutation.
        - Base mutation (all nodes)
        - Rescue mutation (near-death nodes)
        - Topological crossover (adjacent nodes, weighted by biomass)
        - Lévy wind dispersal (exploration)
        - Kin speciation (μ_K)
        - Fungal expansion (profitable fungi)
        """
        alive = self._alive()
        if len(alive) < 2:
            return

        sr = self.cfg.sol_range
        new_nodes: List[Node] = []
        new_fungi: List[FungalEdge] = []

        # ── Base mutation ──────────────────────────────────────────────────
        for n in alive:
            n.sx = np.clip(n.sx + np.random.normal(0, self.cfg.mu_base * sr), -sr, sr)
            n.sy = np.clip(n.sy + np.random.normal(0, self.cfg.mu_base * sr), -sr, sr)
            # Rescue mutation if near death
            if n.B < self.cfg.B_critical + self.cfg.epsilon_rescue:
                n.sx = np.clip(n.sx + np.random.uniform(-1, 1) * self.cfg.Omega * 0.08 * sr, -sr, sr)
                n.sy = np.clip(n.sy + np.random.uniform(-1, 1) * self.cfg.Omega * 0.08 * sr, -sr, sr)

        # ── Topological crossover ─────────────────────────────────────────
        total_B = sum(n.B for n in alive) or 1e-9

        for _ in range(self.cfg.N_children):
            # Select parent A by biomass
            pa = self._select_by_biomass(alive, total_B)
            if pa is None:
                continue
            neighbors = self._neighbors(pa)
            if not neighbors:
                continue
            nb_B = sum(n.B for n in neighbors) or 1e-9
            pb = self._select_by_biomass(neighbors, nb_B)
            if pb is None or pa is pb:
                continue

            wa = pa.B / (pa.B + pb.B)
            wb = 1.0 - wa

            if random.random() < self.cfg.p_wind:
                # Lévy wind: global exploration
                new_cx = random.uniform(0.05, 0.95)
                new_cy = random.uniform(0.05, 0.95)
                new_sx = random.uniform(-sr, sr)
                new_sy = random.uniform(-sr, sr)
            else:
                # Local exploitation: weighted centroid + noise
                new_cx = np.clip(wa * pa.cx + wb * pb.cx + np.random.normal(0, self.cfg.sigma_pos), 0.02, 0.98)
                new_cy = np.clip(wa * pa.cy + wb * pb.cy + np.random.normal(0, self.cfg.sigma_pos), 0.02, 0.98)
                new_sx = np.clip(wa * pa.sx + wb * pb.sx + np.random.normal(0, self.cfg.mu_base * sr), -sr, sr)
                new_sy = np.clip(wa * pa.sy + wb * pb.sy + np.random.normal(0, self.cfg.mu_base * sr), -sr, sr)

            # Kin speciation
            if random.random() < self.cfg.mu_K:
                self.cfg.K_max += 1
                new_K = self.cfg.K_max
            else:
                new_K = pa.K if random.random() < wa else pb.K

            new_P = np.clip(wa * pa.P + wb * pb.P + np.random.normal(0, 0.05), 0.15, 1.0)
            new_R = np.clip(wa * pa.R + wb * pb.R + np.random.normal(0, 0.04), 0.01, 0.85)

            child = Node(new_cx, new_cy, new_sx, new_sy, new_P, new_R, new_K)
            new_nodes.append(child)

        # ── Fungal expansion ───────────────────────────────────────────────
        alive_set = set(id(n) for n in alive)
        for f in self._active_fungi():
            if f.Pi <= self.cfg.Pi_expand or random.random() > self.cfg.p_expand:
                continue
            parent = f.na if f.na.phi >= f.nb.phi else f.nb
            if self._degree(parent) >= parent.deg_max:
                continue
            connected_ids = {id(nb) for nb in self._neighbors(parent)}
            candidates = sorted(
                [n for n in alive if id(n) not in connected_ids and n is not parent],
                key=lambda n: self._utility(parent, n),
                reverse=True
            )
            if not candidates:
                continue
            target = candidates[0]
            if self._degree(target) >= target.deg_max:
                continue
            child_f = FungalEdge(parent, target)
            child_f.T = np.clip(f.T + np.random.normal(0, 0.05), 0.10, 1.0)
            child_f.E = np.clip(f.E + np.random.normal(0, 0.05), 0.10, 1.0)
            child_f.A = np.clip(f.A + np.random.normal(0, 0.10), 0.00, 1.0)
            new_fungi.append(child_f)

        self.nodes.extend(new_nodes)
        self.fungi.extend(new_fungi)

    @staticmethod
    def _select_by_biomass(candidates: List[Node], total: float) -> Optional[Node]:
        if not candidates or total <= 0:
            return None
        r = random.uniform(0, total)
        for n in candidates:
            r -= n.B
            if r <= 0:
                return n
        return candidates[-1]

    @staticmethod
    def _utility(parent: Node, candidate: Node) -> float:
        """Expected utility for a fungus seeking a new node to connect."""
        grad = abs(parent.phi - candidate.phi)
        prox = 1.0 - math.dist((parent.cx, parent.cy), (candidate.cx, candidate.cy))
        return grad * prox

    # ── RESILIENCE METRIC R(t) ────────────────────────────────────────────────

    def _resilience(self) -> float:
        """
        FIX BUG-06: Full R(t) = w1·Φ + w2·D + w3·C + w4·A
        """
        alive = self._alive()
        if not alive:
            return 0.0
        cfg = self.cfg

        # Φ — mean genetic performance (phi, NOT biomass)
        phi_mean = sum(n.phi for n in alive) / len(alive)

        # D — topological diversity (1 - HHI over phi)
        sum_phi = sum(n.phi for n in alive) or 1e-9
        hhi = sum((n.phi / sum_phi)**2 for n in alive)
        diversity = max(0.0, 1.0 - hhi)

        # C — structural connectivity
        af   = len(self._active_fungi())
        min_e = max(1, len(alive) - 1)
        degs  = [self._degree(n) for n in alive]
        var_d = float(np.var(degs)) if degs else 0.0
        max_var = max(1.0, (len(alive) / 2)**2)
        connectivity = np.clip((af / min_e) * (1.0 - var_d / max_var), 0.0, 1.0)

        # A — recovery ratio over last W_recovery gens
        recovery = 0.5
        if len(self.hist_R) >= cfg.W_recovery:
            window = self.hist_R[-cfg.W_recovery:]
            ratios = [window[i] / window[i-1] for i in range(1, len(window)) if window[i-1] > 0]
            if ratios:
                recovery = np.clip(sum(ratios) / len(ratios), 0.0, 1.0)

        R = (cfg.w1 * phi_mean + cfg.w2 * diversity +
             cfg.w3 * connectivity + cfg.w4 * recovery)
        return float(np.clip(R, 0.0, 1.0))


# ─── SIMULATION RUNNER ───────────────────────────────────────────────────────

def run_simulation(cfg: Config = CFG, verbose: bool = True) -> dict:
    """
    Run a full CE-CBMN simulation.

    Parameters
    ----------
    cfg     : Config dataclass with all hyperparameters
    verbose : Print generation-level summary

    Returns
    -------
    dict with keys: history_R, history_phi, history_nodes, history_fungi,
                    final_ecosystem, best_phi_ever
    """
    eco = Ecosystem(cfg)

    hist_R      = []
    hist_phi    = []
    hist_nodes  = []
    hist_fungi  = []
    best_phi    = 0.0

    sep = "=" * 72
    if verbose:
        print(sep)
        print("  CE-CBMN v2.1.0  |  Author: Adan Facundo  |  ORCID: 0009-0003-9110-4744")
        print("  Benchmark: Rastrigin 2D  |  Optimum: phi=1.0 at (0,0)")
        print(sep)
        print(f"  Config: N={cfg.num_nodes} nodes | {cfg.generations} gens | "
              f"p_wind={cfg.p_wind} | mu_K={cfg.mu_K}")
        print(sep)
        header = f"{'Gen':>4}  {'Nodes':>5}  {'Fungi':>5}  {'phi_mean':>8}  "
        header += f"{'phi_max':>8}  {'B_mean':>7}  {'R(t)':>7}  {'Event'}"
        print(header)
        print("-" * 72)

    # Inject a guaranteed hard shock at generation 40 for benchmark purposes
    SHOCK_GEN = 40

    for t in range(1, cfg.generations + 1):
        # Inject shock at generation 40
        event = ""
        if t == SHOCK_GEN:
            # Epicenter at center of plane (worst-case)
            eco.add_epicenter(0.5, 0.5, I=0.70, rho=0.18, dur=2)
            event = "⚡ STOCHASTIC SHOCK"

        R = eco.step()

        alive   = eco._alive()
        af      = eco._active_fungi()
        phi_m   = sum(n.phi for n in alive) / len(alive) if alive else 0.0
        phi_mx  = max((n.phi for n in alive), default=0.0)
        B_m     = sum(n.B   for n in alive) / len(alive) if alive else 0.0
        best_phi = max(best_phi, phi_mx)

        hist_R.append(R)
        hist_phi.append(phi_m)
        hist_nodes.append(len(alive))
        hist_fungi.append(len(af))

        if verbose and (t % 10 == 0 or t == SHOCK_GEN or t == SHOCK_GEN + 1):
            print(f"  {t:>4}  {len(alive):>5}  {len(af):>5}  "
                  f"{phi_m:>8.4f}  {phi_mx:>8.4f}  {B_m:>7.4f}  "
                  f"{R:>7.4f}  {event}")

        if not alive:
            if verbose:
                print(f"\n  [!] Population extinct at generation {t}")
            break

    if verbose:
        print(sep)
        final_alive  = eco._alive()
        final_phi_m  = sum(n.phi for n in final_alive) / len(final_alive) if final_alive else 0.0
        final_R      = hist_R[-1] if hist_R else 0.0
        # Find best solution
        best_node = max(final_alive, key=lambda n: n.phi) if final_alive else None
        print(f"\n  FINAL RESULTS")
        print(f"  {'Surviving nodes':<30}: {len(final_alive)}")
        print(f"  {'Active fungi':<30}: {len(eco._active_fungi())}")
        print(f"  {'Mean phi (performance)':<30}: {final_phi_m:.4f}")
        print(f"  {'Best phi ever reached':<30}: {best_phi:.4f}")
        print(f"  {'Global Resilience R(t_final)':<30}: {final_R:.4f}")
        if best_node:
            print(f"  {'Best solution (sx,sy)':<30}: ({best_node.sx:.4f}, {best_node.sy:.4f})")
            print(f"  {'Rastrigin value at best':<30}: {rastrigin_2d(best_node.sx, best_node.sy):.4f}")
        print(f"  {'Kin modules active':<30}: {cfg.K_max}")
        print(sep)

    return {
        'history_R':     hist_R,
        'history_phi':   hist_phi,
        'history_nodes': hist_nodes,
        'history_fungi': hist_fungi,
        'final_ecosystem': eco,
        'best_phi_ever': best_phi,
    }


# ─── OPTIONAL: MATPLOTLIB VISUALIZATION ──────────────────────────────────────

def plot_results(results: dict, save_path: str = None):
    """
    Generate a 4-panel figure suitable for preprint submission.
    Requires matplotlib.
    """
    try:
        import matplotlib.pyplot as plt
        import matplotlib.gridspec as gridspec
    except ImportError:
        print("matplotlib not installed. Run: pip install matplotlib")
        return

    hist_R     = results['history_R']
    hist_phi   = results['history_phi']
    hist_nodes = results['history_nodes']
    hist_fungi = results['history_fungi']
    eco        = results['final_ecosystem']
    gens       = list(range(1, len(hist_R) + 1))

    fig = plt.figure(figsize=(14, 10), facecolor='#0a0f0a')
    fig.suptitle(
        'CE-CBMN v2.1 — Rastrigin 2D Benchmark\n'
        'Author: Adan Facundo | ORCID: 0009-0003-9110-4744',
        color='#00ff88', fontsize=13, fontfamily='monospace', y=0.98
    )
    gs = gridspec.GridSpec(2, 2, hspace=0.42, wspace=0.35,
                           left=0.08, right=0.96, top=0.90, bottom=0.07)

    panel_style = dict(facecolor='#04100a')
    tick_style  = dict(colors='#3a7a50', labelsize=9)
    label_style = dict(color='#7dcf9a', fontsize=10, fontfamily='monospace')

    # ── Panel 1: R(t) and phi(t) ──────────────────────────────────────────────
    ax1 = fig.add_subplot(gs[0, 0], **panel_style)
    ax1.plot(gens, hist_R,   color='#00ff88', lw=1.5, label='R(t) — Resilience')
    ax1.plot(gens, hist_phi, color='#00aaff', lw=1.2, alpha=0.8, label='φ̄(t) — Mean Performance')
    ax1.axvline(x=40, color='#ff4444', lw=1, ls='--', alpha=0.7, label='Shock @ gen 40')
    ax1.set_title('Global Resilience & Performance', color='#7dcf9a',
                  fontsize=10, fontfamily='monospace')
    ax1.set_xlabel('Generation', **label_style)
    ax1.set_ylabel('Value', **label_style)
    ax1.tick_params(**tick_style)
    ax1.legend(fontsize=8, facecolor='#04100a', labelcolor='#7dcf9a',
               edgecolor='#1a3a25')
    ax1.set_facecolor('#04100a')
    ax1.spines[:].set_color('#1a3a25')
    ax1.set_ylim(0, 1.05)

    # ── Panel 2: Population dynamics ─────────────────────────────────────────
    ax2 = fig.add_subplot(gs[0, 1], **panel_style)
    ax2.plot(gens, hist_nodes, color='#00ff88', lw=1.5, label='Alive nodes')
    ax2_r = ax2.twinx()
    ax2_r.plot(gens, hist_fungi, color='#ffaa00', lw=1.2, alpha=0.8, label='Active fungi')
    ax2.axvline(x=40, color='#ff4444', lw=1, ls='--', alpha=0.7)
    ax2.set_title('Population Dynamics', color='#7dcf9a',
                  fontsize=10, fontfamily='monospace')
    ax2.set_xlabel('Generation', **label_style)
    ax2.set_ylabel('Nodes', **label_style)
    ax2_r.set_ylabel('Fungi', color='#ffaa00', fontsize=10, fontfamily='monospace')
    ax2.tick_params(**tick_style)
    ax2_r.tick_params(colors='#ffaa00', labelsize=9)
    ax2.set_facecolor('#04100a')
    ax2.spines[:].set_color('#1a3a25')
    ax2_r.spines[:].set_color('#1a3a25')
    lines1, labels1 = ax2.get_legend_handles_labels()
    lines2, labels2 = ax2_r.get_legend_handles_labels()
    ax2.legend(lines1 + lines2, labels1 + labels2, fontsize=8,
               facecolor='#04100a', labelcolor='#7dcf9a', edgecolor='#1a3a25')

    # ── Panel 3: Final node positions in Rastrigin space ─────────────────────
    ax3 = fig.add_subplot(gs[1, 0], **panel_style)
    sr  = CFG.sol_range
    xs_bg = np.linspace(-sr, sr, 150)
    ys_bg = np.linspace(-sr, sr, 150)
    Z = np.array([[rastrigin_2d(x, y) for x in xs_bg] for y in ys_bg])
    ax3.contourf(xs_bg, ys_bg, Z, levels=25, cmap='Greens_r', alpha=0.6)
    alive = eco._alive()
    if alive:
        xs = [n.sx for n in alive]
        ys = [n.sy for n in alive]
        cs = [n.phi for n in alive]
        sc = ax3.scatter(xs, ys, c=cs, cmap='RdYlGn', s=30, alpha=0.85,
                         vmin=0, vmax=1, edgecolors='#ffffff44', linewidths=0.3)
        plt.colorbar(sc, ax=ax3, label='φᵢ', shrink=0.8)
    ax3.scatter([0], [0], c='white', s=80, marker='*', zorder=5, label='Global optimum')
    ax3.set_title('Node Distribution in Rastrigin Space\n(Final generation)',
                  color='#7dcf9a', fontsize=10, fontfamily='monospace')
    ax3.set_xlabel('sₓ', **label_style)
    ax3.set_ylabel('sᵧ', **label_style)
    ax3.tick_params(**tick_style)
    ax3.set_facecolor('#04100a')
    ax3.spines[:].set_color('#1a3a25')
    ax3.legend(fontsize=8, facecolor='#04100a', labelcolor='#7dcf9a',
               edgecolor='#1a3a25')

    # ── Panel 4: Network topology in spatial plane ────────────────────────────
    ax4 = fig.add_subplot(gs[1, 1], **panel_style)
    for f in eco._active_fungi():
        ax4.plot([f.na.cx, f.nb.cx], [f.na.cy, f.nb.cy],
                 color='#00cc5544', lw=0.6 + f.T, alpha=0.5)
    if alive:
        degs = {n.id: eco._degree(n) for n in alive}
        max_deg = max(degs.values()) if degs else 1
        for n in alive:
            size = 20 + 120 * (degs[n.id] / max_deg)
            color = '#ffd700' if degs[n.id] > 5 else (
                    '#00ff88' if n.role == 'SOURCE' else
                    '#ff4466' if n.role == 'SINK' else
                    '#ffaa00' if n.role == 'SURVIVAL' else '#3a9a60')
            ax4.scatter(n.cx, n.cy, s=size, c=color, alpha=0.85,
                        edgecolors='#ffffff22', linewidths=0.3)
    ax4.set_title('Final Network Topology\n(Spatial plane, node size ∝ degree)',
                  color='#7dcf9a', fontsize=10, fontfamily='monospace')
    ax4.set_xlabel('xₛₚₐₜᵢₐₗ', **label_style)
    ax4.set_ylabel('yₛₚₐₜᵢₐₗ', **label_style)
    ax4.tick_params(**tick_style)
    ax4.set_facecolor('#04100a')
    ax4.spines[:].set_color('#1a3a25')
    ax4.set_xlim(0, 1); ax4.set_ylim(0, 1)

    # Legend for roles
    from matplotlib.patches import Patch
    legend_elements = [
        Patch(facecolor='#ffd700', label='Hub (deg > 5)'),
        Patch(facecolor='#00ff88', label='Source'),
        Patch(facecolor='#ff4466', label='Sink'),
        Patch(facecolor='#ffaa00', label='Survival'),
    ]
    ax4.legend(handles=legend_elements, fontsize=7, facecolor='#04100a',
               labelcolor='#7dcf9a', edgecolor='#1a3a25', loc='upper right')

    if save_path:
        plt.savefig(save_path, dpi=150, bbox_inches='tight', facecolor='#0a0f0a')
        print(f"\n  Figure saved → {save_path}")
    plt.tight_layout()
    plt.show()


# ─── ENTRY POINT ─────────────────────────────────────────────────────────────

if __name__ == "__main__":
    results = run_simulation(CFG, verbose=True)

    # Generate figure (requires matplotlib)
    try:
        plot_results(results, save_path="ce_cbmn_results.png")
    except Exception as e:
        print(f"\n  [visualization skipped: {e}]")

    print("\n  To cite this work:")
    print("  Facundo, A. (2026). CE-CBMN: Co-Evolutionary Common Biological Market Networks.")
    print("  Preprint. https://github.com/facundoyamil96/CE-CBMN-Algorithm ")
    print("  ORCID: 0009-0003-9110-4744")
