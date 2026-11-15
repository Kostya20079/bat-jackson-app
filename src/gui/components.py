"""GUI components and dialogs for parameter settings and network visualization."""
import tkinter as tk
from tkinter import ttk
from typing import Dict, Any
import numpy as np


class ParametersFrame(ttk.LabelFrame):
    """Configuration panel for Bat Algorithm and Jackson Network weights."""

    def __init__(self, parent, **kwargs):
        super().__init__(parent, text="Parametry Algorytmu Nietoperza (BA)", padding=10, **kwargs)
        self._build_widgets()

    def _build_widgets(self):
        # Grid layout
        labels = [
            ("Liczba osobników (N):", "population_size", "30"),
            ("Maksymalna liczba iteracji (T):", "max_iterations", "100"),
            ("Min. częstotliwość (f_min):", "f_min", "0.0"),
            ("Maks. częstotliwość (f_max):", "f_max", "2.0"),
            ("Początkowa głośność (A₀):", "a_init", "0.9"),
            ("Początkowa emisja (r₀):", "r_init", "0.1"),
            ("Współczynnik alfa (α):", "alpha", "0.95"),
            ("Współczynnik gamma (γ):", "gamma", "0.05"),
            ("Waga czasu W_sieci (α_celu):", "weight_alpha", "1.0"),
            ("Waga kosztu kanałów (β_celu):", "weight_beta", "0.2"),
        ]

        self.entries: Dict[str, ttk.Entry] = {}
        for row, (lbl_text, key, default_val) in enumerate(labels):
            lbl = ttk.Label(self, text=lbl_text)
            lbl.grid(row=row, column=0, sticky=tk.W, pady=2, padx=4)
            entry = ttk.Entry(self, width=12)
            entry.insert(0, default_val)
            entry.grid(row=row, column=1, sticky=tk.E, pady=2, padx=4)
            self.entries[key] = entry

    def get_values(self) -> Dict[str, Any]:
        """Parse entries into structured dictionary."""
        return {
            "population_size": int(self.entries["population_size"].get()),
            "max_iterations": int(self.entries["max_iterations"].get()),
            "f_min": float(self.entries["f_min"].get()),
            "f_max": float(self.entries["f_max"].get()),
            "a_init": float(self.entries["a_init"].get()),
            "r_init": float(self.entries["r_init"].get()),
            "alpha": float(self.entries["alpha"].get()),
            "gamma": float(self.entries["gamma"].get()),
            "weight_alpha": float(self.entries["weight_alpha"].get()),
            "weight_beta": float(self.entries["weight_beta"].get()),
        }


class ResultsTable(ttk.LabelFrame):
    """Table showing node-by-node characteristics and optimization comparison."""

    def __init__(self, parent, **kwargs):
        super().__init__(parent, text="Wyniki Optymalizacji Węzłów Sieci", padding=10, **kwargs)
        self._build_table()

    def _build_table(self):
        columns = ("node", "channels", "lambda", "mu", "rho", "lq", "wq", "w", "l")
        self.tree = ttk.Treeview(self, columns=columns, show="headings", height=6)

        headers = [
            ("node", "Węzeł"),
            ("channels", "Kanały (cᵢ)"),
            ("lambda", "Napływ (λᵢ)"),
            ("mu", "Obsługa (μᵢ)"),
            ("rho", "Obciążenie (ρᵢ)"),
            ("lq", "Kolejka (Lqᵢ)"),
            ("wq", "Czas w kol. (Wqᵢ)"),
            ("w", "Czas w węźle (Wᵢ)"),
            ("l", "Liczba w węźle (Lᵢ)"),
        ]

        for col_id, col_name in headers:
            self.tree.heading(col_id, text=col_name)
            self.tree.column(col_id, width=95, anchor=tk.CENTER)

        self.tree.pack(fill=tk.BOTH, expand=True)

    def populate(self, characteristics):
        """Populate table with JacksonNetworkCharacteristics data."""
        for item in self.tree.get_children():
            self.tree.delete(item)

        n = len(characteristics.lambdas)
        for i in range(n):
            self.tree.insert(
                "",
                tk.END,
                values=(
                    f"Węzeł {i + 1}",
                    f"{round(characteristics.lambdas[i] / (characteristics.rhos[i] * 5.0)) if characteristics.rhos[i] > 0 else '-'}",
                    f"{characteristics.lambdas[i]:.3f}",
                    f"-",
                    f"{characteristics.rhos[i]:.3f}",
                    f"{characteristics.lq[i]:.3f}",
                    f"{characteristics.wq[i]:.3f}",
                    f"{characteristics.w[i]:.3f}",
                    f"{characteristics.l[i]:.3f}",
                ),
            )
