import json
import os
from datetime import datetime
import numpy as np


def _jsonify(x):
    """Convert numpy types/arrays to JSON-serializable Python types."""
    if isinstance(x, (np.integer,)):
        return int(x)
    if isinstance(x, (np.floating,)):
        return float(x)
    if isinstance(x, (np.bool_,)):
        return bool(x)
    if isinstance(x, (np.ndarray,)):
        return x.tolist()
    if callable(x):
        return getattr(x, "__name__", str(x))
    return x


def save_initial_population_json(filepath: str, pop: np.ndarray, meta: dict | None = None):
    pop = np.asarray(pop, dtype=float)
    payload = {
        "created_at": datetime.now().isoformat(timespec="seconds"),
        "shape": list(pop.shape),
        "dtype": "float64",
        "population": pop.tolist(),
        "meta": meta or {},
    }
    os.makedirs(os.path.dirname(filepath), exist_ok=True)
    with open(filepath, "w", encoding="utf-8") as f:
        json.dump(payload, f, indent=2)
    print(f"[init-pop] Saved: {filepath}")


def load_initial_population_json(filepath: str) -> np.ndarray | None:
    if not os.path.exists(filepath):
        return None
    try:
        with open(filepath, "r", encoding="utf-8") as f:
            payload = json.load(f)
        pop = np.asarray(payload["population"], dtype=float)

        # Basic sanity checks
        if pop.ndim != 2:
            print(f"[init-pop] Bad file (ndim!=2): {filepath}")
            return None
        if not np.all(np.isfinite(pop)):
            print(f"[init-pop] Bad file (non-finite values): {filepath}")
            return None

        print(f"[init-pop] Loaded: {filepath}  shape={pop.shape}")
        return pop
    except Exception as e:
        print(f"[init-pop] Failed to load {filepath}: {e}")
        return None


def _generate_initial_population(num_pop, fourier, N, M, ROWS, COLS):
    pop_list = []
    seed = 0
    while len(pop_list) < num_pop:
        if num_pop <= 0:
            raise ValueError("num_pop must be positive.")

        np.random.seed(seed)
        pop = np.random.uniform(-1, 1, N * M * 4)

        F_cont = fourier.generate_fourier_series(np.array(pop))
        F_discr_norm = fourier.discretize_fourier_series(F_cont)
        conn, k_mat = fourier.build_adjacency_matrix(F_discr_norm, ROWS, COLS, rigid_outer_frame=False)
        connected, _ = fourier.check_global_connectivity(conn)

        seed += 1
        if connected:
            pop_list.append(pop)
        else:
            continue

    return np.array(pop_list)


def get_or_generate_initial_population(num_pop: int, cache_path: str, fourier, N: int, M: int, ROWS: int, COLS: int) -> np.ndarray:
    """
    1) Try load cache
    2) Validate it matches current expected gene length
    3) Else generate + save
    """
    expected_genes = N * M * 4

    pop = load_initial_population_json(cache_path)
    if pop is not None:
        # Validate compatibility with current run
        if pop.shape[0] == num_pop and pop.shape[1] == expected_genes:
            return pop
        else:
            print(
                f"[init-pop] Cache shape mismatch. "
                f"Found {pop.shape}, expected ({num_pop}, {expected_genes}). Regenerating..."
            )

    pop = _generate_initial_population(num_pop=num_pop, fourier=fourier, N=N, M=M, ROWS=ROWS, COLS=COLS)

    # Save with useful metadata
    meta = {
        "num_pop": num_pop,
        "rows": ROWS,
        "cols": COLS,
        "N": N,
        "M": M,
        "gene_length": expected_genes,
        "note": "Population includes only globally connected topologies (per FourierSeries2D check).",
    }
    save_initial_population_json(cache_path, pop, meta=meta)
    return pop


def save_ga_result_json(
    ga_instance,
    filepath,
    best_sol,
    best_fit,
    *,
    fourier=None,
    rows=None,
    cols=None,
    include_best_topology_snapshot=True,
    extra=None,
):
    """
    Save a readable GA summary to JSON.

    Parameters
    ----------
    ga_instance : pygad.GA
    filepath : str
    best_sol : np.ndarray
    best_fit : float
    fourier : FourierSeries2D (optional)
        If provided and include_best_topology_snapshot=True, we will compute
        F_cont, F_discr_norm, conn, k_mat for the best solution and store a compact snapshot.
    rows, cols : int (optional)
        Needed for building adjacency if your fourier methods require them.
    include_best_topology_snapshot : bool
    extra : dict (optional)
    """

    payload = {
        "created_at": datetime.now().isoformat(timespec="seconds"),
        "ga_config": {
            "num_generations": _jsonify(ga_instance.num_generations),
            "sol_per_pop": _jsonify(ga_instance.sol_per_pop),
            "num_parents_mating": _jsonify(ga_instance.num_parents_mating),
            "keep_elitism": _jsonify(ga_instance.keep_elitism),
            "parent_selection_type": _jsonify(ga_instance.parent_selection_type),
            "crossover_type": _jsonify(ga_instance.crossover_type),
            "mutation_type": _jsonify(ga_instance.mutation_type),
            "mutation_probability": _jsonify(ga_instance.mutation_probability),
        },
        "best_solution": {
            "fitness": _jsonify(best_fit),
            "genes": _jsonify(np.asarray(best_sol, dtype=float)),
            "num_genes": _jsonify(len(best_sol)),
        },
        "history": {
            "best_fitness_by_generation": _jsonify(np.asarray(ga_instance.best_solutions_fitness, dtype=float)),
            "mean_fitness_by_generation": _jsonify(np.asarray(extra.get("mean_fitness_by_generation", []), dtype=float)),
        },
    }

    # Optional: also store per-generation best solutions (can be large if genes are big)
    # Uncomment if you really need it.
    # if ga_instance.best_solutions is not None:
    #     payload["history"]["best_solution_by_generation"] = _jsonify(np.asarray(ga_instance.best_solutions, dtype=float))

    # Optional: topology snapshot for best solution (compact)
    if include_best_topology_snapshot and (fourier is not None):
        try:
            best_sol_np = np.asarray(best_sol, dtype=float)

            F_cont = fourier.generate_fourier_series(best_sol_np)
            F_discr_norm = fourier.discretize_fourier_series(F_cont)

            # Your class method signature may differ:
            # - If it's fourier.build_adjacency_matrix(...)
            # - Or fourier.build_adjacency_matrix(F_discr_norm, rows, cols, ...)
            if rows is not None and cols is not None:
                conn, k_mat = fourier.build_adjacency_matrix(
                    F_discr_norm, rows, cols, rigid_outer_frame=False
                )
                connected, eigvals = fourier.check_global_connectivity(conn)
            else:
                conn, k_mat = fourier.build_adjacency_matrix(F_discr_norm, rigid_outer_frame=False)
                connected, eigvals = fourier.check_global_connectivity(conn)

            # Store a *compact* snapshot (avoid huge continuous arrays if you want)
            payload["best_solution"]["topology_snapshot"] = {
                "connected": _jsonify(connected),
                "laplacian_eigvals_first10": _jsonify(np.asarray(eigvals[:10], dtype=float)),
                "F_discr_norm": _jsonify(np.asarray(F_discr_norm, dtype=float)),
                "conn": _jsonify(np.asarray(conn, dtype=int)),
                "k_mat": _jsonify(np.asarray(k_mat, dtype=float)),
                "stats": {
                    "num_edges": _jsonify(int(np.sum(conn) // 2)),
                    "num_rigid_edges": _jsonify(int(np.sum(k_mat < 0) // 2)),
                    "num_soft_edges": _jsonify(int(np.sum(k_mat > 0) // 2)),
                }
            }

            # If you *really* want the continuous field too, uncomment:
            # payload["best_solution"]["topology_snapshot"]["F_cont"] = _jsonify(np.asarray(F_cont, dtype=float))

        except Exception as e:
            payload["best_solution"]["topology_snapshot_error"] = str(e)

    if extra is not None:
        payload["extra"] = {k: _jsonify(v) for k, v in extra.items()}

    with open(filepath, "w", encoding="utf-8") as f:
        json.dump(payload, f, indent=4, sort_keys=False)

    print(f"Saved GA JSON to: {filepath}")
