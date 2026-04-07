import sys
import os
import time
import json
import shutil
import pygad
import numpy as np
import matplotlib.pyplot as plt
from pathlib import Path
from sklearn.linear_model import Ridge
from sklearn.metrics import r2_score
from sklearn.preprocessing import StandardScaler

# --- Import Paths ---
current_dir = Path(__file__).parent
src_dir = current_dir.parent
sys.path.insert(0, str(src_dir))

try:
    # Your specific training module
    from openprc.analysis.utils.training_utils import compute_ipc_components
    from openprc.optimization.search_spaces.fourier_series_2D import FourierSeries2D
    from openprc.examples.spring_mass_2D import run_pipeline
    from openprc.demlat.utils.animator import ShowSimulation
except ImportError as e:
    print(f"Import failed: {e}")
    sys.exit(1)

# --- Configuration ---
np.set_printoptions(linewidth=240)

EPS = -1E9
N_TARGET = 2
TAU_D_TARGET = 5
ALPHA = 0.04
BETA = 0.04
ROWS, COLS = 4, 4
N, M = 5, 5
MEAN_FITNESS_HISTORY = []

POP_SIZE = 64
GENERATIONS = 100
RESERVOIR_DOF = 0 # X-axis

# --- Directory Setup ---
# experiments/spring_mass_{R}x{C}_test
EXPERIMENT_ROOT = src_dir / "experiments" / f"spring_mass_{ROWS}x{COLS}_test"
os.makedirs(EXPERIMENT_ROOT, exist_ok=True)

fourier = FourierSeries2D(ROWS, COLS)

# --- Helpers ---
def _jsonify(x):
    if isinstance(x, (np.integer,)): return int(x)
    if isinstance(x, (np.floating,)): return float(x)
    if isinstance(x, (np.bool_,)): return bool(x)
    if isinstance(x, (np.ndarray,)): return x.tolist()
    return x

def save_initial_population_json(filepath: str, pop: np.ndarray, meta: dict | None = None):
    pop = np.asarray(pop, dtype=float)
    payload = {"created_at": time.strftime("%Y-%m-%d"), "population": pop.tolist(), "meta": meta or {}}
    with open(filepath, "w") as f: json.dump(payload, f, indent=2)
    print(f"[init-pop] Saved: {filepath}")

def load_initial_population_json(filepath: str) -> np.ndarray | None:
    if not os.path.exists(filepath): return None
    with open(filepath, "r") as f: return np.asarray(json.load(f)["population"], dtype=float)

def get_or_generate_initial_population(num_pop: int, cache_path: str) -> np.ndarray:
    pop = load_initial_population_json(cache_path)
    if pop is not None and pop.shape[0] == num_pop: return pop
    
    # Generate new
    pop_list = []
    gene_len = fourier.N * fourier.M * 4
    seed = 0
    while len(pop_list) < num_pop:
        np.random.seed(seed)
        p = np.random.uniform(-1, 1, gene_len)
        F_cont = fourier.generate_fourier_series(p)
        F_norm = fourier.discretize_fourier_series(F_cont)
        conn, _ = fourier.build_adjacency_matrix(F_norm, ROWS, COLS, rigid_outer_frame=False)
        if fourier.check_global_connectivity(conn)[0]: pop_list.append(p)
        seed += 1
    pop = np.array(pop_list)
    save_initial_population_json(cache_path, pop)
    return pop

def heuristic_crossover(parents, offspring_size, ga_instance):
    offspring = []
    fitness = ga_instance.last_generation_fitness
    for k in range(offspring_size[0]):
        p1_idx = k % parents.shape[0]
        p2_idx = (k + 1) % parents.shape[0]
        if fitness[p2_idx] > fitness[p1_idx]: p1_idx, p2_idx = p2_idx, p1_idx
        offspring.append(parents[p2_idx] + np.random.uniform(0, 1.2) * (parents[p1_idx] - parents[p2_idx]))
    return np.array(offspring)

def save_ga_result_json(ga, filepath, best_sol, best_fit, fourier=None, extra=None):
    data = {
        "best_solution": {"fitness": _jsonify(best_fit), "genes": _jsonify(best_sol)},
        "history": {"best": _jsonify(ga.best_solutions_fitness), "mean": _jsonify(extra["mean"])}
    }
    print(f"Attempting to save JSON to: {filepath}")
    try:
        with open(filepath, "w") as f: json.dump(data, f, indent=4)
        print(f"Successfully saved GA JSON.")
    except Exception as e:
        print(f"Failed to save JSON: {e}")

# --- FITNESS FUNCTION ---
def fitness_func(ga_instance, solution, solution_idx):
    # 1. Decode
    F_cont = fourier.generate_fourier_series(solution)
    F_norm = fourier.discretize_fourier_series(F_cont)
    conn, k_mat = fourier.build_adjacency_matrix(F_norm, ROWS, COLS, rigid_outer_frame=False)
    
    if not fourier.check_global_connectivity(conn)[0]: return EPS

    # 2. Run Pipeline
    gen = ga_instance.generations_completed
    # We pass rows/cols to pipeline to ensure grid matches GA config
    pos_data, _ = run_pipeline(
        rows=ROWS, cols=COLS,
        k_mat=k_mat, c_mat=conn * 0.4, 
        ga_generation=gen
    )

    if pos_data is None or np.any(np.isnan(pos_data)): 
        return EPS

    # 3. Post-Process
    displacements = pos_data - pos_data[0, :, :]
    X_raw = displacements[:, :, RESERVOIR_DOF]
    
    scaler = StandardScaler()
    X = scaler.fit_transform(X_raw)
    
    rank_X = np.linalg.matrix_rank(X)
    cond_X = np.linalg.cond(X)
    if not np.isfinite(cond_X) or cond_X <= 0: 
        return EPS

    # 4. Training
    dt = 0.01 
    washout, train_len, test_len = int(5/dt), int(10/dt), int(10/dt)
    train_stop = washout + train_len
    
    try:
        act_idx = 0
        u_target = pos_data[:, act_idx, RESERVOIR_DOF]

        _, R2_basis_s, _ = compute_ipc_components(
            X, u_target, TAU_D_TARGET, N_TARGET, 
            washout, train_stop, test_len, 100
        )
        meanR2 = np.nanmean(R2_basis_s)
        if not np.isfinite(meanR2): return EPS
    except: 
        return EPS
    
    return meanR2 + ALPHA*(rank_X/min(X.shape)) - BETA*np.log(cond_X)

# --- Generation Callback ---
def on_generation(ga_instance):
    global MEAN_FITNESS_HISTORY

    best_sol, best_fit, best_idx = ga_instance.best_solution()

    # Fitness values for the just-finished generation (size = sol_per_pop)
    gen_fit = np.asarray(ga_instance.last_generation_fitness, dtype=float)

    # Option A (recommended): exclude invalid guardrail fitness (EPS) from mean
    valid = gen_fit[np.isfinite(gen_fit) & (gen_fit > EPS/2)]  # EPS is -1e9, so EPS/2 is still huge negative
    mean_fit = float(np.mean(valid)) if valid.size > 0 else float(np.mean(gen_fit))

    MEAN_FITNESS_HISTORY.append(mean_fit)

    print(f"\n=== Generation {ga_instance.generations_completed} done ===")
    print(f"Best gene[0]: {best_sol[0]:.6f} | best fitness: {best_fit:.6f} | mean fitness: {mean_fit:.6f}\n")

# --- MAIN ---
def main():
    print("="*60)
    print(f"Generating initial population... Saving to {EXPERIMENT_ROOT}")
    init_pop_path = EXPERIMENT_ROOT / "initial_population.json"
    initial_pop = get_or_generate_initial_population(POP_SIZE, init_pop_path)
    
    ga = pygad.GA(
        num_generations=GENERATIONS,
        num_parents_mating=2,
        fitness_func=fitness_func,
        initial_population=initial_pop,
        parent_selection_type="tournament",
        crossover_type=heuristic_crossover,
        mutation_type="random",
        mutation_probability=0.3,
        sol_per_pop=POP_SIZE,
        num_genes=initial_pop.shape[1],
        on_generation=on_generation,
        keep_elitism=2
    )

    t0 = time.time()
    ga.run()
    print(f"GA Finished. Time: {(time.time()-t0)/3600:.2f}h")

    best_sol, best_fit, _ = ga.best_solution()
    
    # Save & Plot
    json_path = EXPERIMENT_ROOT / "ga_results.json"
    save_ga_result_json(ga, json_path, best_sol, best_fit, fourier, {"mean": MEAN_FITNESS_HISTORY})
    
    plt.figure()
    plt.plot(ga.best_solutions_fitness, label="Best")
    plt.plot(MEAN_FITNESS_HISTORY, label="Mean")
    plt.xlabel("Generation")
    plt.ylabel("Fitness")
    plt.legend()
    plt.title("Convergence")
    plt.savefig(EXPERIMENT_ROOT / "convergence.png")
    
    # Visualize Best
    F_cont = fourier.generate_fourier_series(best_sol)
    F_norm = fourier.discretize_fourier_series(F_cont)
    A_opt, K_opt = fourier.build_adjacency_matrix(F_norm, ROWS, COLS, rigid_outer_frame=False)
    A_orig, K_orig = fourier.build_full_neighbor_topology(ROWS, COLS, rigid_outer_frame=False)
    
    fourier.plot_fourier_viz(
        F=F_cont, F_discr_norm=F_norm,
        A_new=A_opt, K_new=K_opt,
        A_orig=A_orig, K_orig=K_orig,
        rows=ROWS, cols=COLS
    )
    
    # Launch Player
    last_gen = ga.generations_completed
    best_dir = EXPERIMENT_ROOT / f"generation_{last_gen}"
    
    print(f"\n[Opening Visualizer for {best_dir}]")
    if best_dir.exists():
        ShowSimulation(str(best_dir))
    else:
        # Re-run if missing for some reason
        run_pipeline(k_mat=K_opt, c_mat=A_opt*0.4, ga_generation=last_gen, rows=ROWS, cols=COLS)
        ShowSimulation(str(best_dir))

if __name__ == "__main__":
    main()