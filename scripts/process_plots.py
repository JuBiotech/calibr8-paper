import pathlib
import shutil
import subprocess
from PIL import Image


DP_FIGURES = pathlib.Path(__file__).absolute().parent.parent / "data_and_analysis" / "figures"
DP_OUTPUT = DP_FIGURES / "converted"
DP_OUTPUT.mkdir(exist_ok=True)
assert DP_FIGURES.exists(), DP_FIGURES.absolute()


strategy = {
    "2.1 Bayes' rule.pdf": "Fig1.eps",                           # ✔
    "2.4 dependent 3D.png": "Fig2.tif",                          # ✔
    "2.4 independent 3D.png": "Fig3.tif",                        # ✔
    "2.4 theory_residualplots.pdf": "Fig4.eps",                  # ✔ (würde mit nicht-alpha t_bands noch besser)
    "3.2.1 Asymmetric Logistic.pdf": "Fig5.eps",                 # ✔
    "3.2.2 calibr8 Class Diagram.pdf": "Fig6.eps",               # ✔
    "3.2.4 murefi Model Graph.pdf": "Fig7.eps",                  # ✔
    "4.1.1 CM_glucose_linear+logistic.pdf": "Fig8.eps",          # ✔ (würde mit nicht-alpha t_bands noch besser)
    "4.1.1 CM_CDW.pdf": "Fig9.eps",                              # ✔ (würde mit nicht-alpha t_bands noch besser)
    "4.1.3 UQ_infer_independent.pdf": "Fig10.eps",               # ✔
    "4.2.2 Process Model MLE with data.pdf": "Fig11.eps",        # ✔ (weißes bitmap im linken subplot muss manuell gelöscht werden)
    "4.2.3 Pair plot and kinetics.pdf": "Fig12.eps",             # ✔
    "4.2.4 PM with Linear vs. Logistic CM.png": "Fig13.tif",     # ✔
    "4.2.4 Log-CDW and Monod Residuals.png": "Fig14.tif",        # ✔
    "4.2.4 Hierarchical X0 analysis.pdf": "Fig15.eps",           # ✔ (wird von CorelDRAW automatisch gedreht)
    "2.1 Normal vs. StudentT.pdf": "S1_Fig.eps",                 # ✔
    "Appendix_pairplot.pdf": "S2_Fig.eps",                       # ✔
}


for src, dst in strategy.items():
    fp_src = DP_FIGURES / src
    fp_dst = DP_OUTPUT / dst
    if fp_dst.exists():
        continue
    if src.endswith(".pdf"):
        # Convert with inkscape
        subprocess.check_call([f"C:\Program Files\Inkscape\Inkscape.exe", str(fp_src), f"--export-eps={fp_dst}"])
    elif src.endswith(".png"):
        # Convert with Pillow
        img = Image.open(fp_src)
        img.save(fp_dst)
    elif src.split(".")[-1] == dst.split(".")[-1]:
        # Just copy it
        shutil.copy(fp_src, fp_dst)
    else:
        raise NotImplementedError(f"Don't know how to process {src} 👉 {dst}.")
    print(f"Processed {src} 👉 {dst}")
