#!/usr/bin/env python3
"""
Generate agent-aware orthogonal dataset for Van Benthem product space validation.

Key innovations:
1. Natural language names (Alice, Bob, Carol...) for complexity → encourages grokking
2. Systematic agent observation patterns for Van Benthem structure
3. Tests BOTH:
   - Task-level: X ≈ X_n_agents × X_depth
   - Agent-level: X ≈ X_agent0 × X_agent1 × ... × X_depth

Design:
- Train: n_agents∈{2,3}, depth∈{0,1}
  - Different name pool: {Alice, Bob, Carol, David, Emma}
  - Systematic observation patterns per agent

- Val: n_agents∈{4,5}, depth∈{2,3}
  - Different name pool: {Xavier, Yara, Zeke, Quinn, Riley}
  - Same observation pattern structure

Observation Pattern Strategy:
For each n_agents level, create patterns that systematically vary:
- Agent 0 observations: cycles through {0}, {1}, {0,1}, {0,2}
- Agent 1 observations: cycles through {0}, {1}, {1,2}, {0,1}
- etc.

This creates true agent-level variation while maintaining complexity gradient.
"""

import os
import sys
# Problem.py / util.py come from https://github.com/sileod/llm-theory-of-mind (src/).
# Point MINDGAMES_SRC at that checkout's src/ directory.
sys.path.append(os.environ.get('MINDGAMES_SRC', '../llm-theory-of-mind/src'))

from Problem import Problem, Var, Expression, Announcement, Law
import pandas as pd
import numpy as np
import random
import re, string
from util import solve, class_to_dict
from string import Template
from tqdm.auto import tqdm
import copy
import jinja2
from dataclasses import dataclass
import json
from pathlib import Path
from collections import Counter
from itertools import product, cycle

tqdm.pandas()
randint = np.random.randint


# Name pools for train vs val
TRAIN_NAME_POOL = ['Alice', 'Bob', 'Carol', 'David', 'Emma', 'Frank']
VAL_NAME_POOL = ['Xavier', 'Yara', 'Zeke', 'Quinn', 'Riley', 'Sofia']


def postprocess(s):
    s = s.replace('.', '. ')
    s = ".".join(s.split('.')).replace('. .', '.')
    s = " ".join(s.split())
    if not s.endswith('.'):
        s = s + '.'
    return s


def postprocess_hyp(s, seed=None):
    random.seed(seed)
    s = s.replace('knows', jinja2.Template(
        "{{ ['can know'] | random }}"
    ).render())
    s = s.replace('can know', jinja2.Template(
        "{{ ['can now know'] | random }}"
    ).render(), 1)
    return s


@dataclass
class Setup:
    n_announcements: int = 3
    n_agents: int = 3
    hypothesis_depth: int = 0
    announcement_depth: int = 1
    variables = None
    agents = None
    announcements = None
    hypothesis = None
    type = None
    base_observation = None
    law = Expression(Law('Top'))
    matrix = None  # Observation matrix

    def __post_init__(self):
        self.agents = [f'Agent{string.ascii_lowercase[i]}' for i in range(self.n_agents)]


clique = "Everyone is visible to others."


class forehead(Setup):
    base_observation = clique
    variables_template = "$agent's forehead is muddy"
    type = 'visual'

    def __post_init__(self):
        super().__post_init__()
        n_agents = self.n_agents
        self.matrix = np.ones((n_agents, n_agents)) - np.eye(n_agents)


class forehead_mirror(forehead):
    base_observation = f'{clique} There is a mirror in the room. '

    def __post_init__(self):
        super().__post_init__()
        n_agents = self.n_agents
        self.matrix = np.ones((n_agents, n_agents))


class arm(forehead_mirror):
    base_observation = clique
    variables_template = "$agent's arm is muddy"


class internal(Setup):
    base_observation = clique
    variables_template = '$agent is thirsty'

    def __post_init__(self):
        super().__post_init__()
        self.matrix = np.eye(self.n_agents)

    type = 'mental'


class explicit(Setup):
    variables_template = '$agent is muddy'
    matrix = None


def create_observation_patterns(n_agents, n_vars_range=(2, 4)):
    """
    Create systematic observation patterns for agents.

    Returns list of observation patterns, where each pattern is a list of
    sets indicating which variables each agent observes.

    Example for n_agents=2:
    [
        [{0}, {1}],       # Agent0 sees var0, Agent1 sees var1
        [{0,1}, {0,1}],   # Both see both
        [{0}, {0,1}],     # Agent0 sees var0, Agent1 sees both
        [{0,1}, {1}],     # Agent0 sees both, Agent1 sees var1
    ]
    """
    patterns = []

    # Define base observation sets for cycling
    if n_agents == 2:
        # For 2 agents, create varied patterns
        agent_obs_options = [
            [{0}, {1}],
            [{0, 1}, {0, 1}],
            [{0}, {0, 1}],
            [{0, 1}, {1}],
            [{0}, {0}],
            [{1}, {1}],
        ]
    elif n_agents == 3:
        # For 3 agents, more complex patterns
        agent_obs_options = [
            [{0}, {1}, {2}],           # Each sees one unique var
            [{0, 1}, {1, 2}, {0, 2}],  # Pairwise overlaps
            [{0}, {1}, {0, 1, 2}],     # One sees all
            [{0, 1}, {0, 1}, {2}],     # Two share, one separate
            [{0}, {0, 1}, {1, 2}],     # Chain pattern
            [{0, 1, 2}, {0, 1, 2}, {0, 1, 2}],  # All see all
        ]
    elif n_agents == 4:
        agent_obs_options = [
            [{0}, {1}, {2}, {3}],      # Each sees one
            [{0, 1}, {1, 2}, {2, 3}, {0, 3}],  # Ring structure
            [{0}, {1}, {2}, {0, 1, 2, 3}],  # One sees all
            [{0, 1}, {0, 1}, {2, 3}, {2, 3}],  # Paired
            [{0}, {0, 1}, {1, 2}, {2, 3}],  # Chain
            [{0, 1, 2, 3}, {0, 1, 2, 3}, {0, 1, 2, 3}, {0, 1, 2, 3}],  # All see all
        ]
    elif n_agents == 5:
        agent_obs_options = [
            [{0}, {1}, {2}, {3}, {4}],  # Each sees one
            [{0, 1}, {1, 2}, {2, 3}, {3, 4}, {0, 4}],  # Ring
            [{0}, {1}, {2}, {3}, {0, 1, 2, 3, 4}],  # One sees all
            [{0, 1}, {0, 1}, {2, 3}, {2, 3}, {4}],  # Paired + singleton
            [{0}, {0, 1}, {1, 2}, {2, 3}, {3, 4}],  # Chain
            [{0, 1, 2, 3, 4}, {0, 1, 2, 3, 4}, {0, 1, 2, 3, 4}, {0, 1, 2, 3, 4}, {0, 1, 2, 3, 4}],  # All
        ]
    else:
        # Fallback: random patterns
        agent_obs_options = []
        for _ in range(6):
            pattern = []
            for agent_idx in range(n_agents):
                n_obs = random.randint(1, min(3, n_agents))
                obs_set = set(random.sample(range(n_agents), n_obs))
                pattern.append(obs_set)
            agent_obs_options.append(pattern)

    return agent_obs_options


def create_observation_matrix(observation_pattern, n_agents):
    """
    Convert observation pattern to matrix form.

    observation_pattern: list of sets, e.g., [{0}, {1}, {0,1}]
    Returns: n_agents × n_agents matrix where entry (i,j) = 1 if agent i observes var j
    """
    matrix = np.zeros((n_agents, n_agents), dtype=int)
    for agent_idx, obs_vars in enumerate(observation_pattern):
        for var_idx in obs_vars:
            if var_idx < n_agents:  # Ensure valid index
                matrix[agent_idx, var_idx] = 1
    return matrix


def generate_orthogonal_split_agent_aware(
    samples_per_combination=120,
    n_agents_range=(2, 3),
    hypothesis_depth_range=(0, 1),
    n_announcements_max=5,
    seed=42,
    split_name="train",
    name_pool=None
):
    """
    Generate split with natural language names and systematic agent observations.

    Args:
        samples_per_combination: Number of samples per (n_agents, depth) combination
        n_agents_range: Tuple of n_agents values
        hypothesis_depth_range: Tuple of hypothesis_depth values
        n_announcements_max: Maximum announcements
        seed: Random seed
        split_name: Name for logging
        name_pool: List of names to use (e.g., TRAIN_NAME_POOL or VAL_NAME_POOL)
    """
    random.seed(seed)
    np.random.seed(seed)

    if name_pool is None:
        name_pool = TRAIN_NAME_POOL

    print(f"\n{'='*70}")
    print(f"Generating {split_name} split (agent-aware):")
    print(f"{'='*70}")
    print(f"  n_agents: {n_agents_range}")
    print(f"  hypothesis_depth: {hypothesis_depth_range}")
    print(f"  samples_per_combination: {samples_per_combination}")
    print(f"  name_pool: {name_pool[:6]}")
    print(f"  Total combinations: {len(n_agents_range) * len(hypothesis_depth_range)}")

    problems = []

    # Pre-generate observation patterns for each n_agents level
    observation_patterns_by_n = {}
    for n_agents in n_agents_range:
        observation_patterns_by_n[n_agents] = create_observation_patterns(n_agents)
        print(f"  Created {len(observation_patterns_by_n[n_agents])} observation patterns for n_agents={n_agents}")

    # Generate problems for each combination
    for n_agents in n_agents_range:
        for hyp_depth in hypothesis_depth_range:
            print(f"\n  Generating n_agents={n_agents}, hypothesis_depth={hyp_depth}...")

            # Get observation patterns for this n_agents level
            obs_patterns = observation_patterns_by_n[n_agents]
            pattern_cycle = cycle(obs_patterns)

            generated = 0
            attempts = 0
            max_attempts = samples_per_combination * 50

            while generated < samples_per_combination and attempts < max_attempts:
                attempts += 1

                # Select observation pattern (cycle through systematically)
                obs_pattern = next(pattern_cycle)

                # Select setup type (prefer explicit for custom matrices)
                setup_class = explicit

                # Create setup with custom observation matrix
                setup = class_to_dict(setup_class(
                    n_agents=n_agents,
                    n_announcements=n_announcements_max,
                    hypothesis_depth=hyp_depth,
                ))

                setup = copy.deepcopy(setup)
                setup['n_announcements'] = n_announcements = randint(0, setup['n_announcements'] + 1)

                # Set custom observation matrix
                setup['matrix'] = create_observation_matrix(obs_pattern, n_agents)

                # Create variables
                setup['variables'] = [
                    Var(Template(setup['variables_template']).substitute(agent=setup['agents'][i]), i + 1)
                    for i in range(len(setup['agents']))
                ]

                # Generate base problem
                base_pb = Problem(**setup)
                setup['announcements'] = [Expression(Announcement(base_pb.someone()))] + [
                    Expression(Announcement(random.choice([
                        base_pb.random_expression(1), base_pb.random_knowledge(0)
                    ])))
                    for _ in range(setup['n_announcements'])
                ]

                # Check validity
                pbcheck = Problem(**setup)
                pbcheck.hypothesis = Expression(Var('0', 0))
                pbcheck_str = str(pbcheck).replace('VARS ', 'VARS 0,')

                try:
                    valid = solve(pbcheck_str)
                    if valid:
                        continue
                except:
                    continue

                # Generate hypothesis and solve
                try:
                    pb = Problem(**setup)
                    smcdel_problem = str(pb)

                    label = solve(smcdel_problem)
                    if label == -1:
                        continue

                    pb.change_format('natural')
                    premise = pb.observations_to_str() + str(pb.law) + pb.announcements_to_str()
                    hypothesis = str(pb.hypothesis)

                    # Extract agent observations from the generated problem
                    agent_observations = []
                    for agent_idx in range(n_agents):
                        agent_observations.append(list(obs_pattern[agent_idx]))

                    problems.append({
                        'smcdel_problem': smcdel_problem,
                        'premise': premise,
                        'hypothesis': hypothesis,
                        'label': 'entailment' if label == 1 else 'non_entailment',
                        'constraint': 'entailment' if label == 1 else 'non_entailment',
                        'setup': setup['name'],
                        'hypothesis_depth': hyp_depth,
                        'n_agents': n_agents,
                        'n_announcements': n_announcements,
                        'agent_observations': agent_observations,  # NEW: per-agent observations
                    })

                    generated += 1

                except Exception as e:
                    continue

            print(f"    Generated {generated}/{samples_per_combination} samples (attempts: {attempts})")
            if generated < samples_per_combination:
                print(f"    ⚠️  Warning: Could not generate enough samples")

    print(f"\n  Total problems generated: {len(problems)}")

    # Create DataFrame
    df = pd.DataFrame(problems)

    # Apply natural language names (from name pool)
    df['names'] = df.apply(lambda x: random.sample(name_pool, x.n_agents), axis=1)

    # Name substitution
    for k in ['premise', 'hypothesis']:
        def apply_names(row):
            text = row[k]
            # Replace Agenta, Agentb, etc. with actual names
            agents = [f'Agent{string.ascii_lowercase[i]}' for i in range(row['n_agents'])]
            for old_agent, new_name in zip(agents, row['names']):
                text = re.sub(rf'\b{old_agent}\b', new_name, text)
            return text

        df[k] = df.apply(apply_names, axis=1)
        df[k] = df[k].map(postprocess)
        if 'hyp' in k:
            df[k] = df.apply(lambda x: postprocess_hyp(x[k], seed=x['premise']), axis=1)

    # Calculate interaction score
    df['interaction_score'] = df['n_agents'] * (df['hypothesis_depth'] + 1)

    return df


def print_split_statistics(df, split_name):
    """Print detailed statistics for a split."""
    print(f"\n{'='*70}")
    print(f"{split_name.upper()} SPLIT STATISTICS")
    print(f"{'='*70}")

    print(f"Total samples: {len(df)}")

    print(f"\nDistribution by n_agents:")
    agent_counts = df['n_agents'].value_counts().sort_index()
    for n_agents, count in agent_counts.items():
        print(f"  {n_agents} agents: {count} ({100*count/len(df):.1f}%)")

    print(f"\nDistribution by hypothesis_depth:")
    depth_counts = df['hypothesis_depth'].value_counts().sort_index()
    for depth, count in depth_counts.items():
        print(f"  depth={depth}: {count} ({100*count/len(df):.1f}%)")

    print(f"\nCross-tabulation (n_agents × hypothesis_depth):")
    crosstab = df.groupby(['n_agents', 'hypothesis_depth']).size().unstack(fill_value=0)
    print(crosstab)

    print(f"\nLabel balance:")
    label_counts = df['constraint'].value_counts()
    for label, count in label_counts.items():
        print(f"  {label}: {count} ({100*count/len(df):.1f}%)")

    # Average complexity
    avg_score = df['interaction_score'].mean()
    print(f"\nAverage interaction_score: {avg_score:.2f}")

    # Agent observation diversity
    print(f"\nAgent observation pattern diversity:")
    if 'agent_observations' in df.columns:
        unique_patterns = df['agent_observations'].apply(lambda x: str(x)).nunique()
        print(f"  Unique observation patterns: {unique_patterns}")

    return avg_score


def analyze_disjointness(train_df, val_df):
    """Verify train and val have no overlapping combinations."""
    print(f"\n{'='*70}")
    print("DISJOINTNESS ANALYSIS")
    print(f"{'='*70}")

    train_combos = set(zip(train_df['n_agents'], train_df['hypothesis_depth']))
    val_combos = set(zip(val_df['n_agents'], val_df['hypothesis_depth']))

    print(f"\nTrain combinations: {sorted(train_combos)}")
    print(f"Val combinations: {sorted(val_combos)}")

    overlap = train_combos & val_combos

    if overlap:
        print(f"\n❌ ERROR: Overlapping combinations: {overlap}")
        return False
    else:
        print(f"\n✓ Perfect disjointness: ZERO overlapping combinations")
        return True


def analyze_name_separation(train_df, val_df):
    """Verify train and val use different name pools."""
    print(f"\n{'='*70}")
    print("NAME POOL SEPARATION ANALYSIS")
    print(f"{'='*70}")

    train_names = set()
    val_names = set()

    for names in train_df['names']:
        train_names.update(names)

    for names in val_df['names']:
        val_names.update(names)

    print(f"\nTrain names: {sorted(train_names)}")
    print(f"Val names: {sorted(val_names)}")

    overlap = train_names & val_names

    if overlap:
        print(f"\n⚠️ Warning: Overlapping names: {overlap}")
        print("  (This is okay if intentional for testing)")
        return False
    else:
        print(f"\n✓ Perfect name separation: Different name pools")
        return True


def main():
    import argparse

    parser = argparse.ArgumentParser(description='Generate agent-aware orthogonal dataset')
    parser.add_argument('--output_dir', type=str, default='mindgames_orthogonal_v4_agent_aware',
                       help='Output directory')
    parser.add_argument('--samples_per_combo', type=int, default=120,
                       help='Samples per (n_agents, depth) combination')
    parser.add_argument('--seed', type=int, default=42,
                       help='Random seed')

    args = parser.parse_args()

    output_dir = Path(__file__).parent / args.output_dir
    output_dir.mkdir(exist_ok=True)

    print("="*70)
    print("AGENT-AWARE ORTHOGONAL DATASET GENERATION")
    print("="*70)
    print("\nStrategy: Natural language names + systematic agent observations")
    print("\nTrain: n_agents∈{2,3}, depth∈{0,1}, names={Alice,Bob,Carol,...}")
    print("Val:   n_agents∈{4,5}, depth∈{2,3}, names={Xavier,Yara,Zeke,...}")
    print("\nEnables testing:")
    print("  1. Task-level: X ≈ X_n_agents × X_depth")
    print("  2. Agent-level: X ≈ X_agent0 × X_agent1 × ... × X_depth")

    # Generate training split
    print("\n" + "="*70)
    print("PHASE 1: GENERATING TRAINING SPLIT")
    print("="*70)

    train_df = generate_orthogonal_split_agent_aware(
        samples_per_combination=args.samples_per_combo,
        n_agents_range=(2, 3),
        hypothesis_depth_range=(0, 1),
        n_announcements_max=5,
        seed=args.seed,
        split_name="TRAIN",
        name_pool=TRAIN_NAME_POOL
    )

    train_avg_score = print_split_statistics(train_df, "TRAIN")

    # Generate validation split
    print("\n" + "="*70)
    print("PHASE 2: GENERATING VALIDATION SPLIT")
    print("="*70)

    val_df = generate_orthogonal_split_agent_aware(
        samples_per_combination=args.samples_per_combo,
        n_agents_range=(4, 5),
        hypothesis_depth_range=(2, 3),
        n_announcements_max=5,
        seed=args.seed + 1000,
        split_name="VAL",
        name_pool=VAL_NAME_POOL
    )

    val_avg_score = print_split_statistics(val_df, "VAL")

    # Analyze relationships
    print("\n" + "="*70)
    print("PHASE 3: ANALYZING TRAIN-VAL RELATIONSHIP")
    print("="*70)

    is_disjoint = analyze_disjointness(train_df, val_df)
    names_separated = analyze_name_separation(train_df, val_df)

    print(f"\nComplexity gradient:")
    print(f"  Train avg score: {train_avg_score:.2f}")
    print(f"  Val avg score: {val_avg_score:.2f}")
    print(f"  Ratio: {val_avg_score/train_avg_score:.2f}x")

    if val_avg_score > train_avg_score * 3.0:
        print(f"\n✓✓✓ EXCELLENT gradient for grokking!")
    elif val_avg_score > train_avg_score * 2.0:
        print(f"\n✓✓ Strong gradient for grokking")
    else:
        print(f"\n✓ Good gradient for grokking")

    # Save datasets
    print(f"\n{'='*70}")
    print("PHASE 4: SAVING DATASETS")
    print(f"{'='*70}")

    def prepare_generation_format(row):
        return {
            'smcdel_problem': row['smcdel_problem'],
            'premise': row['premise'],
            'hypothesis': row['hypothesis'],
            'constraint': row['constraint'],
            'setup': row['setup'],
            'n_agents': int(row['n_agents']),
            'hypothesis_depth': int(row['hypothesis_depth']),
            'interaction_score': int(row['interaction_score']),
            'n_announcements': int(row['n_announcements']),
            'agent_observations': row['agent_observations'],  # NEW
            'agent_names': row['names'],  # NEW
        }

    train_data = [prepare_generation_format(row) for _, row in train_df.iterrows()]
    val_data = [prepare_generation_format(row) for _, row in val_df.iterrows()]

    with open(output_dir / 'train_generation.json', 'w') as f:
        json.dump(train_data, f, indent=2)

    with open(output_dir / 'val_generation.json', 'w') as f:
        json.dump(val_data, f, indent=2)

    print(f"\nSaved:")
    print(f"  {output_dir / 'train_generation.json'}: {len(train_data)} samples")
    print(f"  {output_dir / 'val_generation.json'}: {len(val_data)} samples")

    # Save metadata
    metadata = {
        'version': 'v4_agent_aware',
        'description': 'Agent-aware dataset with natural names + systematic observations',
        'strategy': 'Train: simple (n∈{2,3}, d∈{0,1}), Val: complex (n∈{4,5}, d∈{2,3})',
        'train_n_agents': [2, 3],
        'train_depth': [0, 1],
        'val_n_agents': [4, 5],
        'val_depth': [2, 3],
        'train_samples': len(train_data),
        'val_samples': len(val_data),
        'samples_per_combination': args.samples_per_combo,
        'train_avg_score': float(train_avg_score),
        'val_avg_score': float(val_avg_score),
        'complexity_ratio': float(val_avg_score / train_avg_score),
        'disjoint_combinations': is_disjoint,
        'disjoint_names': names_separated,
        'train_name_pool': TRAIN_NAME_POOL,
        'val_name_pool': VAL_NAME_POOL,
        'seed': args.seed,
        'innovations': [
            'Natural language names for complexity (prevents trivial learning)',
            'Systematic agent observation patterns (enables Van Benthem test)',
            'Different name pools train/val (stronger generalization test)',
            'Agent-level metadata tracking (per-agent observations)',
            'Supports both task-level AND agent-level factorization tests'
        ]
    }

    with open(output_dir / 'metadata.json', 'w') as f:
        json.dump(metadata, f, indent=2)

    print(f"  {output_dir / 'metadata.json'}")

    print(f"\n{'='*70}")
    print("GENERATION COMPLETE!")
    print(f"{'='*70}")
    print(f"\n✓ Train: {len(train_data)} samples, natural names")
    print(f"✓ Val: {len(val_data)} samples, different natural names")
    print(f"✓ Disjoint combinations: {is_disjoint}")
    print(f"✓ Disjoint names: {names_separated}")
    print(f"✓ Complexity ratio: {val_avg_score/train_avg_score:.2f}x")
    print(f"\nThis dataset enables full Van Benthem validation!")


if __name__ == '__main__':
    main()
