"""Render publication figures from the published pair data with Matplotlib."""
import json
from pathlib import Path

import matplotlib
matplotlib.use('Agg')
import matplotlib.pyplot as plt
import numpy as np

ROOT = Path(__file__).resolve().parents[1]
OUTPUT = ROOT / 'assets'
OUTPUT.mkdir(exist_ok=True)
plt.rcParams.update({'font.family': 'DejaVu Sans', 'font.size': 10,
                     'svg.fonttype': 'none', 'axes.spines.top': False,
                     'axes.spines.right': False})

cross = json.loads((ROOT / 'data/r7_cross24.json').read_text(encoding='utf-8'))
rows = sorted(cross['rows'], key=lambda r: (r['opponent'], r['seed'], r['seat']))
labels, gains, colors, flags = [], [], [], []
palette = {'Carrot_ref': '#4C78A8', 'Build14_ref': '#2A9D8F', 'Guru_ref': '#8D6CAB'}
for row in rows:
    a, b = row['candidate'], row['control']
    gains.append((a['own_cash'] - a['rival_cash']) - (b['own_cash'] - b['rival_cash']))
    labels.append(f"{row['opponent'].removesuffix('_ref')} / {row['seed']} / seat {row['seat']}")
    colors.append(palette[row['opponent']])
    flags.append(not b['strict_healthy'])

fig, ax = plt.subplots(figsize=(10.8, 6.4), layout='constrained')
bars = ax.barh(labels, gains, color=colors, height=0.68)
for bar, value, flagged in zip(bars, gains, flags):
    if flagged:
        bar.set_hatch('///')
        bar.set_edgecolor('#253045')
    ax.text(value + 28, bar.get_y() + bar.get_height() / 2, f'+{value:.0f}', va='center')
ax.axvline(np.mean(gains), color='#253045', linestyle='--', linewidth=1.2,
           label='Full-panel mean: +961 coins')
ax.invert_yaxis()
ax.set_xlim(0, max(gains) + 300)
ax.set_xlabel('Candidate-minus-control change in final cash margin (coins)')
ax.set_title('R7 vs R4: 12 paired comparisons across 2 conditional worlds', loc='left', pad=18)
ax.legend(loc='lower right', frameon=False)
fig.text(0.01, -0.035, '24 complete games; 3 opponents; swapped seats. Hatched bars retain flagged R4 controls.\n'
         'The worlds were previously exposed; this panel does not estimate population win rate.', fontsize=9)
fig.savefig(OUTPUT / 'r7_pair_gains.svg', bbox_inches='tight')
plt.close(fig)

perf = json.loads((ROOT / 'data/r7_performance.json').read_text(encoding='utf-8'))
groups = sorted({r['opponent'] for r in perf['rows']})
ratios = [np.mean([r['candidate']['callback_total_seconds'] / r['control']['callback_total_seconds']
                  for r in perf['rows'] if r['opponent'] == group]) for group in groups]
fig, ax = plt.subplots(figsize=(8.4, 4), layout='constrained')
bars = ax.bar([g.removesuffix('_ref') for g in groups], [100 * (1-r) for r in ratios], color='#2A9D8F')
for bar, ratio in zip(bars, ratios):
    ax.text(bar.get_x() + bar.get_width()/2, bar.get_height()+0.3,
            f'{100 * (1-ratio):.1f}%', ha='center')
ax.set_ylim(0, max(100*(1-r) for r in ratios) + 4)
ax.set_ylabel('Decrease in mean paired callback-time ratio (%)')
ax.set_title('Runtime-only ablation: actions and cash unchanged in 16/16 pairs', loc='left', pad=18)
fig.text(0.01, -0.035, '32 games; 2 fixed worlds; local host with 6 concurrent games. Not Kaggle hardware timing.', fontsize=9)
fig.savefig(OUTPUT / 'runtime_savings.svg', bbox_inches='tight')
plt.close(fig)
for path in OUTPUT.glob('*.svg'):
    # Canonical text formatting without changing SVG geometry.
    path.write_text('\n'.join(line.rstrip() for line in path.read_text(encoding='utf-8').splitlines()) + '\n',
                    encoding='utf-8', newline='\n')
print('Rendered assets/r7_pair_gains.svg and assets/runtime_savings.svg')
