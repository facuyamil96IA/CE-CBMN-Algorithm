"""
Co-Evolutionary Common Biological Market Networks (CE-CBMN)
Author: Adan Facundo
ORCID: 0009-0003-9110-4744
Description: Formal implementation of the CE-CBMN algorithm benchmarking 
against stochastic shocks on a 2D Inverted Rastrigin topology.
"""

import numpy as np
import math
import random

# --- Algorithm Hyperparameters ---
NUM_NODES = 50
GENERATIONS = 100
SHOCK_GENERATION = 40
C_BASAL = 0.05       # Basal maintenance cost
TAU_CORTE = 3        # Patience threshold for fungal greed
BETA_BONUS = 1.05    # Symbiotic bonus
EFFICIENCY_E = 0.1   # Transfer efficiency
DEFENSE_BOOST = 0.5  # Shield capacity increase during Wave 3

def inverted_rastrigin(x, y):
    """2D Inverted Rastrigin Function. Global optimum at (0,0)"""
    return 20 - (x**2 - 10 * np.cos(2 * math.pi * x) + y**2 - 10 * np.cos(2 * math.pi * y))

class Node:
    def __init__(self, id, x, y):
        self.id = id
        self.x = x
        self.y = y
        self.F = inverted_rastrigin(x, y) # Mathematical Potential
        self.B = random.uniform(0.6, 1.0) # Initial Living Biomass
        self.D = 0.0                      # Phenotypic Defense Shield
        self.is_alive = True

    def apply_shock(self):
        """Applies stochastic damage. Mitigated by defense D."""
        damage = 0.8 * (1.0 - self.D)
        self.B -= damage
        if self.B <= 0:
            self.is_alive = False
            self.B = 0

class FungalEdge:
    def __init__(self, n1, n2):
        self.n1 = n1
        self.n2 = n2
        self.profitability = 0.0
        self.negative_streak = 0
        self.is_active = True

    def process_market_trade(self):
        if not self.is_active or not self.n1.is_alive or not self.n2.is_alive:
            return

        # 1. Calculate Resource Gradient (Delta R)
        delta_R = abs(self.n1.B - self.n2.B)

        # 2. Fungal Profitability: Pi_j(t) = (1 - E_j) * Delta R - C_basal
        self.profitability = ((1 - EFFICIENCY_E) * delta_R) - C_BASAL

        # 3. Market Disconnection (Greed)
        if self.profitability < 0:
            self.negative_streak += 1
        else:
            self.negative_streak = 0

        if self.negative_streak > TAU_CORTE:
            self.is_active = False # Pruning
            return

        # 4. Resource Transfer
        transfer = delta_R * EFFICIENCY_E
        if self.n1.B > self.n2.B:
            self.n1.B -= transfer
            self.n2.B += transfer
        else:
            self.n2.B -= transfer
            self.n1.B += transfer

def initialize_ecosystem():
    """Deploys nodes randomly and connects them into a network."""
    nodes = [Node(i, random.uniform(-5.12, 5.12), random.uniform(-5.12, 5.12)) for i in range(NUM_NODES)]
    # Force Node 0 to be exactly at the global optimum for benchmark testing
    nodes[0].x, nodes[0].y = 0.0, 0.0
    nodes[0].F = inverted_rastrigin(0, 0)
    
    edges = []
    # Create random initial fungal topology
    for i in range(NUM_NODES):
        for j in range(i + 1, NUM_NODES):
            if random.random() < 0.15: # 15% connection probability
                edges.append(FungalEdge(nodes[i], nodes[j]))
    return nodes, edges

def run_simulation():
    print("--- Starting CE-CBMN Simulation ---")
    nodes, edges = initialize_ecosystem()
    
    for t in range(1, GENERATIONS + 1):
        # Phase 1: Node Evaluation & Harvest
        for node in nodes:
            if node.is_alive:
                # Symbiotic Bonus & Basal Cost application abstracted for simplicity
                node.B = min(1.0, node.B * BETA_BONUS - (C_BASAL * 0.1))

        # Phase 2 & 3: Stochastic Disturbance & Signaling (Gen 40)
        if t == SHOCK_GENERATION:
            print(f"\n[!] GENERATION {t}: STOCHASTIC SHOCK DETECTED AT OPTIMUM (0,0)")
            # Attack the optimum node
            nodes[0].apply_shock()
            
            # Topological Wave 3: Defense Activation for neighbors
            print("[+] WAVE 3: Propagating Topological Stress Signal...")
            for edge in edges:
                if edge.is_active:
                    if edge.n1.id == 0 and edge.n2.is_alive:
                        edge.n2.D = min(1.0, edge.n2.D + DEFENSE_BOOST)
                    elif edge.n2.id == 0 and edge.n1.is_alive:
                        edge.n1.D = min(1.0, edge.n1.D + DEFENSE_BOOST)

        # Phase 4: Dynamic Fungal Market Trade
        for edge in edges:
            edge.process_market_trade()

        # Phase 5: State Update & Pruning
        active_nodes = sum(1 for n in nodes if n.is_alive)
        active_edges = sum(1 for e in edges if e.is_active)
        
        if t % 10 == 0 or t == SHOCK_GENERATION:
            print(f"Gen {t:03d} | Active Nodes: {active_nodes:02d} | Active Fungi: {active_edges:02d} | Node 0 (Optimum) Biomass: {nodes[0].B:.2f}")

    print("\n--- Simulation Complete ---")

if __name__ == "__main__":
    run_simulation()
