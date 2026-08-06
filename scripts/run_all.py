"""Generate all main and supplementary figures."""
import os, subprocess, sys
HERE=os.path.dirname(__file__)
for name in [
 "fig1_reduction_topology.py",
 "fig2_dimensionless_certificate.py",
 "fig3_subextensive_skin.py",
 "figS1_certificate_map.py",
 "figS2_boundary_geometry.py",
]:
 print("\n===",name,"===")
 subprocess.run([sys.executable,os.path.join(HERE,name)],check=True)
