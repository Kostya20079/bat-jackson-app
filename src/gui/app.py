"""
Main Tkinter application window for Bat Algorithm Optimization.
Cross-platform (Linux & Windows), clean layout, with real-time matplotlib convergence curve.
"""
import threading
import tkinter as tk
from tkinter import ttk, messagebox
import numpy as np
import matplotlib
matplotlib.use("TkAgg")
from matplotlib.backends.backend_tkagg import FigureCanvasTkAgg
from matplotlib.figure import Figure

from src.core.parameters import BatParameters
from src.core.bat_algorithm import BatAlgorithm
from src.evaluation.jackson_adapter import (
    JacksonNetworkConfig,
    JacksonNetworkEvaluator,
)
from src.gui.components import ParametersFrame


class BatOptimizationApp(tk.Tk):
    """Main application GUI."""

    def __init__(self):
        super().__init__()
        self.title("Optymalizacja Parametrów Sieci - Algorytm Nietoperza")
        self.geometry("1100x750")
        self.minsize(950, 650)

        self._init_network_defaults()
        self._build_ui()

    def _init_network_defaults(self):
        """Set default academic example network (3 nodes)."""
        self.num_nodes = 3
        self.gamma = np.array([4.0, 2.0, 1.0])
        self.mu = np.array([5.0, 4.0, 3.0])
        self.P = np.array([
            [0.0, 0.3, 0.2],
            [0.2, 0.0, 0.3],
            [0.1, 0.2, 0.0],
        ])
        self.c_min = np.array([1, 1, 1])
        self.c_max = np.array([8, 8, 8])
        self.initial_c = np.array([2, 2, 2])

    def _build_ui(self):
        # Top Header
        header = ttk.Frame(self, padding=10)
        header.pack(fill=tk.X)
        title_lbl = ttk.Label(
            header,
            text="Optymalizacja Liczby Kanałów Obsługi (M/M/c) w Otwartej Sieci Jacksona",
            font=("Helvetica", 14, "bold"),
        )
        title_lbl.pack(anchor=tk.W)

        subtitle_lbl = ttk.Label(
            header,
            text="Metoda metaheurystyczna: Algorytm Nietoperza (Bat Algorithm) z optymalizacją numeryczną",
            font=("Helvetica", 10),
        )
        subtitle_lbl.pack(anchor=tk.W)

        # Main horizontal split
        main_paned = ttk.PanedWindow(self, orient=tk.HORIZONTAL)
        main_paned.pack(fill=tk.BOTH, expand=True, padx=10, pady=5)

        # Left panel: Parameters and Control
        left_frame = ttk.Frame(main_paned, padding=5)
        main_paned.add(left_frame, weight=1)

        self.param_panel = ParametersFrame(left_frame)
        self.param_panel.pack(fill=tk.X, pady=5)

        # Network bounds frame
        net_frame = ttk.LabelFrame(left_frame, text="Zakresy Kanałów Węzłów [c_min, c_max]", padding=10)
        net_frame.pack(fill=tk.X, pady=5)

        self.c_entries = []
        for i in range(self.num_nodes):
            row_frame = ttk.Frame(net_frame)
            row_frame.pack(fill=tk.X, pady=2)
            ttk.Label(row_frame, text=f"Węzeł {i+1}: min=").pack(side=tk.LEFT)
            e_min = ttk.Entry(row_frame, width=5)
            e_min.insert(0, str(self.c_min[i]))
            e_min.pack(side=tk.LEFT, padx=3)
            ttk.Label(row_frame, text="max=").pack(side=tk.LEFT)
            e_max = ttk.Entry(row_frame, width=5)
            e_max.insert(0, str(self.c_max[i]))
            e_max.pack(side=tk.LEFT, padx=3)
            self.c_entries.append((e_min, e_max))

        # Action Buttons
        btn_frame = ttk.Frame(left_frame, padding=5)
        btn_frame.pack(fill=tk.X, pady=10)

        self.run_btn = ttk.Button(
            btn_frame, text="▶ Uruchom Optymalizację", command=self._start_optimization_thread
        )
        self.run_btn.pack(side=tk.LEFT, fill=tk.X, expand=True, padx=2)

        self.reset_btn = ttk.Button(btn_frame, text="↺ Resetuj", command=self._reset_defaults)
        self.reset_btn.pack(side=tk.RIGHT, padx=2)

        # Progress bar & Status
        self.progress_var = tk.DoubleVar(value=0.0)
        self.progress_bar = ttk.Progressbar(left_frame, variable=self.progress_var, maximum=100)
        self.progress_bar.pack(fill=tk.X, pady=5)

        self.status_lbl = ttk.Label(left_frame, text="Gotowy do uruchomienia", font=("Helvetica", 9, "italic"))
        self.status_lbl.pack(anchor=tk.W)

        # Right panel: Convergence Plot & Result Metrics
        right_frame = ttk.Frame(main_paned, padding=5)
        main_paned.add(right_frame, weight=2)

        # Plot frame
        plot_frame = ttk.LabelFrame(right_frame, text="Zbieżność Funkcji Celu f(c) w Iteracjach", padding=5)
        plot_frame.pack(fill=tk.BOTH, expand=True)

        self.fig = Figure(figsize=(5, 3.5), dpi=100)
        self.ax = self.fig.add_subplot(111)
        self.ax.set_title("Wykres zbieżności")
        self.ax.set_xlabel("Iteracja")
        self.ax.set_ylabel("Wartość funkcji celu f(c)")
        self.ax.grid(True, linestyle="--", alpha=0.6)
        self.canvas = FigureCanvasTkAgg(self.fig, master=plot_frame)
        self.canvas.get_tk_widget().pack(fill=tk.BOTH, expand=True)

        # Bottom results summary frame
        summary_frame = ttk.LabelFrame(right_frame, text="Podsumowanie Osiągniętych Wyników", padding=10)
        summary_frame.pack(fill=tk.X, pady=5)

        self.summary_text = tk.Text(summary_frame, height=5, wrap=tk.WORD, font=("Courier", 9))
        self.summary_text.pack(fill=tk.BOTH, expand=True)
        self.summary_text.insert(
            tk.END,
            "Konfiguracja początkowa: Węzły c = [2, 2, 2]\nKliknij 'Uruchom Optymalizację', aby rozpocząć proces.\n",
        )
        self.summary_text.configure(state=tk.DISABLED)

    def _reset_defaults(self):
        """Reset inputs to defaults."""
        self._init_network_defaults()
        for i, (emin, emax) in enumerate(self.c_entries):
            emin.delete(0, tk.END)
            emin.insert(0, str(self.c_min[i]))
            emax.delete(0, tk.END)
            emax.insert(0, str(self.c_max[i]))
        self.status_lbl.config(text="Zresetowano parametry domyślne.")

    def _start_optimization_thread(self):
        """Run optimization asynchronously so GUI remains responsive."""
        self.run_btn.config(state=tk.DISABLED)
        self.progress_var.set(0)
        self.status_lbl.config(text="Obliczenia w toku...")
        thread = threading.Thread(target=self._run_optimization, daemon=True)
        thread.start()

    def _run_optimization(self):
        try:
            params_dict = self.param_panel.get_values()
            c_min_arr = np.array([int(e[0].get()) for e in self.c_entries], dtype=np.int64)
            c_max_arr = np.array([int(e[1].get()) for e in self.c_entries], dtype=np.int64)

            config = JacksonNetworkConfig(
                num_nodes=self.num_nodes,
                gamma=self.gamma,
                mu=self.mu,
                P=self.P,
                c_min=c_min_arr,
                c_max=c_max_arr,
                alpha=params_dict["weight_alpha"],
                beta=params_dict["weight_beta"],
            )

            evaluator = JacksonNetworkEvaluator(config)

            bat_params = BatParameters(
                population_size=params_dict["population_size"],
                max_iterations=params_dict["max_iterations"],
                f_min=params_dict["f_min"],
                f_max=params_dict["f_max"],
                a_init=params_dict["a_init"],
                r_init=params_dict["r_init"],
                alpha=params_dict["alpha"],
                gamma=params_dict["gamma"],
            )

            optimizer = BatAlgorithm(evaluator, c_min_arr, c_max_arr, bat_params)

            total_iters = bat_params.max_iterations
            iter_history = []
            fitness_history = []

            def on_iteration(t, best_f, best_sol):
                iter_history.append(t)
                fitness_history.append(best_f)
                # Update progress bar safely in UI thread
                prog = (t / total_iters) * 100
                self.after(0, lambda: self.progress_var.set(prog))

            result = optimizer.optimize(iteration_callback=on_iteration)

            # Calculate network stats before and after
            chars_after = evaluator.calculate_characteristics(result.best_solution)

            # Update UI components
            self.after(0, lambda: self._update_results_ui(result, chars_after, evaluator))

        except Exception as e:
            self.after(0, lambda: messagebox.showerror("Błąd", str(e)))
            self.after(0, lambda: self.status_lbl.config(text=f"Błąd: {e}"))
            self.after(0, lambda: self.run_btn.config(state=tk.NORMAL))

    def _update_results_ui(self, result, chars, evaluator):
        """Render updated plot and table summary."""
        self.ax.clear()
        self.ax.plot(
            range(len(result.convergence_history)),
            result.convergence_history,
            color="#0066cc",
            linewidth=2,
            label="f(c) best",
        )
        self.ax.set_title("Zbieżność Funkcji Celu")
        self.ax.set_xlabel("Iteracja")
        self.ax.set_ylabel("f(c) = α·W_sieci + β·C(c)")
        self.ax.grid(True, linestyle="--", alpha=0.6)
        self.ax.legend(loc="upper right")
        self.canvas.draw()

        # Update text summary
        self.summary_text.configure(state=tk.NORMAL)
        self.summary_text.delete("1.0", tk.END)

        report = (
            f"✓ Zoptymalizowana liczba kanałów c*: {list(result.best_solution)}\n"
            f"✓ Najlepsza wartość funkcji celu f(c*): {result.best_fitness:.4f}\n"
            f"✓ Średni czas w sieci W_sieci: {chars.w_network:.4f} s | Koszt kanałów C(c): {chars.channel_cost:.1f}\n"
            f"✓ Łączna liczba klientów L_sieci: {chars.l_network:.4f} | Stabilność: {'TAK (rho < 1)' if chars.is_stable else 'NIE'}\n"
            f"✓ Czas wykonania optymalizacji: {result.execution_time_sec * 1000:.2f} ms ({result.iterations_run} iteracji)\n"
        )
        self.summary_text.insert(tk.END, report)
        self.summary_text.configure(state=tk.DISABLED)

        self.status_lbl.config(text=f"Optymalizacja zakończona sukcesem w {result.execution_time_sec:.3f} s.")
        self.run_btn.config(state=tk.NORMAL)
