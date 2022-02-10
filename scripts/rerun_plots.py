import pathlib
import subprocess


DP_NOTEBOOKS = pathlib.Path(__file__).absolute().parent.parent / "data_and_analysis"


notebooks = [
    # "2.1 Bayes' rule.ipynb",
    # "2.1 Normal vs. StudentT.ipynb",
    # "2.4 Residuals And Lack of Fit.ipynb",
    # "3.2.1 Asymmetric Logistic Interactive.ipynb",
    # "4.1.1 Plot calibration models.ipynb",
    # "4.1.3 Infer UQ Multimodality.ipynb",
    # "4.2.2 MLE Plot.ipynb",
    # "4.2.3 Pairplot and Kinetic Density.ipynb",
    # "4.2.4 Hierarchical X0 analysis.ipynb",
    # "4.2.4 Log-CDW and Monod Residuals.ipynb",
    # "4.2.4 PM with Linear vs. Logistic CM.ipynb",
]

for nb in notebooks:
    fp = str(DP_NOTEBOOKS / nb)
    print(f"Running {nb}")
    subprocess.check_call([
        "jupyter",
        "nbconvert",
        "--ExecutePreprocessor.kernel_name=\"python3\"",
        "--ExecutePreprocessor.timeout=14000",
        "--execute",
        "--inplace",
        fp,
    ])
    print(f"Notebook {nb} completed 🥳")
