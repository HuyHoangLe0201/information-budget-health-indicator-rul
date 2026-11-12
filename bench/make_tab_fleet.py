"""Table of the fleet-size curve (results/fleet_size_*.json).  Run from bench/."""
import json
S = ['FD001', 'FD002', 'FD003', 'FD004', 'HUST', 'MATR', 'Tongji']; K = ['5', '10', '20', '40', '80', '120']
R = {s: json.load(open(f'../results/fleet_size_{s}.json')) for s in S}
with open('../results/tab_fleet.tex', 'w') as f:
    f.write("\\begin{tabular}{l " + "c "*len(K) + "}\n\\hline\nData set & " + " & ".join(f"$k={k}$" for k in K) + " \\\\\n\\hline\n")
    for s in S:
        f.write(s + " & " + " & ".join((f"{R[s][k]['valid']:.1f} ({R[s][k]['gain_mean']:.2f})" if k in R[s] else "--") for k in K) + " \\\\\n")
    f.write("\\hline\n\\end{tabular}\n")
print(open('../results/tab_fleet.tex').read())
for s in S: print(s, 'first k with share>=0.9:', next((k for k in K if k in R[s] and R[s][k]['valid'] >= 0.9), None), '| control mean at k=40:', R[s].get('40', {}).get('age_mean'))
