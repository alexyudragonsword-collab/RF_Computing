# PyInstaller spec: one-file, windowed build of the WISE demo GUI.
#   cd wise_repro/gui && pyinstaller --noconfirm --clean wise_demo.spec
# The exe lands in wise_repro/gui/dist/.
import os

GUI = SPECPATH
ROOT = os.path.dirname(GUI)

a = Analysis(
    [os.path.join(GUI, "wise_demo.py")],
    pathex=[ROOT],                       # so wise_sim.py is found and bundled
    datas=[
        (os.path.join(GUI, "demo_mnist.npz"), "gui"),
        (os.path.join(GUI, "model_fc3.npz"), "gui"),
        (os.path.join(ROOT, "results", "dataset_mnist_fc3.json"), "results"),
    ],
    excludes=["scipy", "torch", "tkinter", "PyQt5", "PyQt6", "PySide2", "IPython", "pandas"],
)
pyz = PYZ(a.pure)
exe = EXE(
    pyz,
    a.scripts,
    a.binaries,
    a.datas,
    [],
    name="WISE-Demo",
    console=False,
    upx=False,
)
