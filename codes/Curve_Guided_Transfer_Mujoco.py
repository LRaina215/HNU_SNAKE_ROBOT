"""Run the bounded, simulation-only C0/B candidate; see its motion definition."""
from pathlib import Path
import sys
sys.path.insert(0, str(Path(__file__).resolve().parents[1] / 'validation'))
from curve_guided_experiment import main

if __name__ == '__main__':
    main()
