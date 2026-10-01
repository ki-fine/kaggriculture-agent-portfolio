# Kaggriculture: agent integration and reproducible evaluation

This project documents my work integrating a three-person team's strategy ideas into a final agent for Kaggle's two-player farming competition. My responsibilities covered strategy synthesis, module integration, experiment selection, and the final build. The work is relevant to algorithm, machine learning, and AI engineering internships.

## What the project demonstrates

- Fixed production and care schedules combined with strategic choice and budgeted market search
- Integration of six opponent behavior models, with shared action-prefix reuse
- Exact state copying for the Kaggle SDK's dictionary and attribute views
- Behavior-cloning experiments and analysis of offline-to-rollout performance gaps
- Matched local evaluation, cash reconciliation, and source-to-package SHA-256 checks

The development process used AI coding assistance. Team contributions and ownership are documented in [CREDITS.md](docs/CREDITS.md) and [TEAMWORK.md](docs/TEAMWORK.md).

Our shared team [retrospective](docs/RETROSPECTIVE.md) examines unapproved persistent restrictions introduced by AI tools, overgeneralized negative results, outdated comparison baselines, and incomplete mechanism validation. It distinguishes completed work from proposed changes for the next competition.

## Measured local results

| Study | Observation | Scope |
|---|---|---|
| Final R7 vs. R4 | Mean cash-margin improvement of +961; all 12 comparisons positive | 2 previously exposed conditional worlds, 3 opponents, swapped seats, 24 games |
| Copy optimization | Identical actions, cash, and decision semantics in 16 pairs; mean paired callback-time ratio reduced by 15.6% | 32 games, 2 fixed worlds, 4 opponents; local six-process timing |
| Behavior cloning | 88.25% complete-action agreement on 23,008 development-validation turns | 416 training trajectories; autonomous rollout lost all 6 games vs Master and was not deployed |
| Cash audit | 17,256 official transitions with zero reconciliation residuals for both players | Final 24-game panel |
| Kaggle delivery | Accepted submission 56713467 | Final competition score and ranking pending as of 2026-10-01 |

The strict full-panel health gate was 22/24 and failed because two R4 control games retained historical input-error flags. All 12 candidate games passed. The complete panel is retained; the ten clean comparisons are a diagnostic subset. These results do not establish population win rate or leaderboard improvement.

## Reproduce the evidence summary

Python 3.11 or later is sufficient; the commands require only the standard library.

```bash
python src/analyze_results.py --check
python -m unittest discover -s tests -v
```

This reproduces metrics from published compact results and exercises state-copy boundaries. It does not rerun the complete competition agent or regenerate the full replay audit.

## Read further

- [Project report in Chinese](docs/PROJECT_REPORT.md)
- [Post-competition retrospective in Chinese](docs/RETROSPECTIVE.md): scope of negative evidence, route lock-in, and oversight of AI-assisted development
- [Machine learning experiments](docs/ML_EXPERIMENTS.md)
- [Evidence definitions and limitations](docs/EVIDENCE.md)
- [Resume and interview materials](docs/RESUME.md)

The complete agent contains team-supplied routes and models and remains in the team's research repository. This public repository contains selected evaluation code, state-copy improvements, compact evidence, and attribution. Newly authored code is MIT-licensed; referenced Kaggle SDK code is Apache-2.0.
