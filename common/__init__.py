# Спільний модуль для експериментів: стиль, IO, повторюваність.
from common.style import set_thesis_style, set_dstu_style, legend_outside, create_figure, PALETTE
from common.io import (
    ensure_dir,
    save_figure,
    save_table_latex,
    save_table_png,
    save_table_markdown,
    format_adaptive_decimal,
    clean_output_dir,
    clean_pycache,
)
from common.experiment import (
    suppress_stdout,
    run_etalon,
    measure_etalon,
    setup_experiment,
)
from common.rcu import (
    ANCHOR_LOOPS,
    anchor_ns,
    profile_rcu,
    pin_to_p_cores,
    setup_rcu_worker,
)
from common.data import (
    ExperimentData,
    load_experiment_data,
    save_experiment_data,
)
from common.dependencies import (
    check_and_install_package,
    check_dependencies,
    ensure_basic_dependencies,
    ensure_experiment_dependencies,
    safe_import,
)

__all__ = [
    "set_thesis_style",
    "set_dstu_style",
    "legend_outside",
    "create_figure",
    "PALETTE",
    "ensure_dir",
    "save_figure",
    "save_table_latex",
    "save_table_png",
    "save_table_markdown",
    "format_adaptive_decimal",
    "clean_output_dir",
    "clean_pycache",
    "suppress_stdout",
    "run_etalon",
    "measure_etalon",
    "setup_experiment",
    "ExperimentData",
    "save_experiment_data",
    "load_experiment_data",
    "check_and_install_package",
    "check_dependencies",
    "ensure_basic_dependencies",
    "ensure_experiment_dependencies",
    "safe_import",
    "ANCHOR_LOOPS",
    "anchor_ns",
    "profile_rcu",
    "pin_to_p_cores",
    "setup_rcu_worker",
]
