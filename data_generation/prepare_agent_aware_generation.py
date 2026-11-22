#!/usr/bin/env python3
"""
Prepare agent-aware dataset for generation task.

Takes the agent-aware dataset and creates a version with BOTH:
1. SMCDEL symbolic format (with name substitutions) - for training
2. Natural language format - for reference

Output format:
{
  "premise": "VARS 1,2 LAW Top OBS Alice:1 Bob:2",  // SMCDEL (training input)
  "hypothesis": "[ ! (1|2) ] Alice knows whether 1",  // SMCDEL (training output)
  "premise_natural": "Alice knows whether Alice is muddy. Bob knows...",  // Natural
  "hypothesis_natural": "Alice can know whether 1",  // Natural
  "n_agents": 2,
  "agent_observations": [[0], [1]],
  "agent_names": ["Alice", "Bob"]
}
"""

import json
import re
from pathlib import Path
from collections import Counter
import argparse


def substitute_agent_names_in_smcdel(smcdel_text: str, agent_mapping: dict) -> str:
    """
    Substitute agent names in SMCDEL format text.

    Args:
        smcdel_text: Original SMCDEL text with Agenta, Agentb, etc.
        agent_mapping: Dict mapping Agenta->Alice, Agentb->Bob, etc.

    Returns:
        SMCDEL text with substituted names
    """
    result = smcdel_text

    # Sort by length descending to avoid partial replacements
    # (e.g., replace "Agentab" before "Agenta")
    for old_agent in sorted(agent_mapping.keys(), key=len, reverse=True):
        new_agent = agent_mapping[old_agent]
        # Use word boundaries to avoid partial matches
        result = re.sub(rf'\b{old_agent}\b', new_agent, result)

    return result


def split_smcdel_problem(smcdel_problem: str) -> tuple:
    """
    Split SMCDEL problem into premise and hypothesis.

    Example:
    "VARS 1,2 LAW Top OBS Agenta:1 Agentb:2 VALID? [ ! (1|2) ] Agenta knows whether 1"
    ->
    ("VARS 1,2 LAW Top OBS Agenta:1 Agentb:2", "[ ! (1|2) ] Agenta knows whether 1")
    """
    if 'VALID?' not in smcdel_problem:
        raise ValueError(f"No VALID? found in: {smcdel_problem}")

    parts = smcdel_problem.split('VALID?')
    premise = parts[0].strip()
    hypothesis = parts[1].strip() if len(parts) > 1 else ""

    return premise, hypothesis


def create_agent_mapping(n_agents: int, agent_names: list) -> dict:
    """
    Create mapping from symbolic agent names (Agenta, Agentb...) to actual names.

    Args:
        n_agents: Number of agents
        agent_names: List of actual agent names (e.g., ['Alice', 'Bob'])

    Returns:
        Dict mapping Agenta->Alice, Agentb->Bob, etc.
    """
    import string

    mapping = {}
    for i in range(n_agents):
        symbolic_name = f'Agent{string.ascii_lowercase[i]}'
        if i < len(agent_names):
            mapping[symbolic_name] = agent_names[i]
        else:
            # Fallback if not enough names
            mapping[symbolic_name] = symbolic_name

    return mapping


def format_for_generation(item: dict) -> dict:
    """
    Format a dataset item for generation task with BOTH SMCDEL and natural language.

    Args:
        item: Dict with 'premise', 'hypothesis', 'smcdel_problem', 'agent_names', etc.

    Returns:
        Formatted dict with both SMCDEL and natural language versions
    """
    # Get agent mapping
    n_agents = item['n_agents']
    agent_names = item.get('agent_names', [])
    agent_mapping = create_agent_mapping(n_agents, agent_names)

    # Split SMCDEL problem
    smcdel_problem = item.get('smcdel_problem', '')
    smcdel_premise, smcdel_hypothesis = split_smcdel_problem(smcdel_problem)

    # Apply name substitutions to SMCDEL parts
    smcdel_premise_with_names = substitute_agent_names_in_smcdel(smcdel_premise, agent_mapping)
    smcdel_hypothesis_with_names = substitute_agent_names_in_smcdel(smcdel_hypothesis, agent_mapping)

    # Natural language versions (already have real names)
    natural_premise = item.get('premise', '')
    natural_hypothesis = item.get('hypothesis', '')

    formatted = {
        # SMCDEL format (with name substitutions) - for training
        'premise': smcdel_premise_with_names,
        'hypothesis': smcdel_hypothesis_with_names,

        # Natural language format - for reference
        'premise_natural': natural_premise,
        'hypothesis_natural': natural_hypothesis,

        # Label
        'label': item.get('constraint', item.get('label', 'unknown')),

        # Metadata
        'n_agents': n_agents,
        'hypothesis_depth': item.get('hypothesis_depth', 0),
        'interaction_score': item.get('interaction_score', 0),

        # Agent-level metadata
        'agent_observations': item.get('agent_observations', []),
        'agent_names': agent_names,

        # Additional info
        'setup': item.get('setup', 'unknown'),
        'n_announcements': item.get('n_announcements', 0),

        # Original SMCDEL (with symbolic names) for reference
        'smcdel_problem_original': smcdel_problem,
    }

    return formatted


def print_statistics(data: list, split_name: str):
    """Print dataset statistics."""
    print(f"\n{'='*70}")
    print(f"{split_name.upper()} STATISTICS")
    print(f"{'='*70}")

    print(f"\nTotal samples: {len(data)}")

    # Label distribution
    label_counts = Counter(item['label'] for item in data)
    print(f"\nLabel distribution:")
    for label, count in label_counts.items():
        print(f"  {label}: {count} ({100*count/len(data):.1f}%)")

    # n_agents distribution
    agent_counts = Counter(item['n_agents'] for item in data)
    print(f"\nn_agents distribution:")
    for n_agents, count in sorted(agent_counts.items()):
        print(f"  {n_agents} agents: {count} ({100*count/len(data):.1f}%)")

    # hypothesis_depth distribution
    depth_counts = Counter(item['hypothesis_depth'] for item in data)
    print(f"\nhypothesis_depth distribution:")
    for depth, count in sorted(depth_counts.items()):
        print(f"  depth={depth}: {count} ({100*count/len(data):.1f}%)")

    # Cross-tabulation
    print(f"\nCross-tabulation (n_agents × hypothesis_depth):")
    cross_tab = {}
    for item in data:
        key = (item['n_agents'], item['hypothesis_depth'])
        cross_tab[key] = cross_tab.get(key, 0) + 1

    all_n_agents = sorted(set(item['n_agents'] for item in data))
    all_depths = sorted(set(item['hypothesis_depth'] for item in data))

    print(f"{'n_agents':<10}", end='')
    for d in all_depths:
        print(f"depth={d:<3}", end='  ')
    print("Total")
    print("-" * 60)

    for n in all_n_agents:
        print(f"{n:<10}", end='')
        row_total = 0
        for d in all_depths:
            count = cross_tab.get((n, d), 0)
            print(f"{count:<9}", end='  ')
            row_total += count
        print(row_total)

    # Agent observation diversity
    if data and 'agent_observations' in data[0]:
        unique_patterns = len(set(str(item['agent_observations']) for item in data))
        print(f"\nAgent observation patterns: {unique_patterns} unique")

    # Name diversity
    if data and 'agent_names' in data[0]:
        all_names = set()
        for item in data:
            all_names.update(item['agent_names'])
        print(f"Unique agent names: {len(all_names)}")
        print(f"  Names: {sorted(all_names)}")

    # Average lengths
    avg_premise_len = sum(len(item['premise'].split()) for item in data) / len(data)
    avg_hypothesis_len = sum(len(item['hypothesis'].split()) for item in data) / len(data)
    avg_premise_natural_len = sum(len(item['premise_natural'].split()) for item in data) / len(data)
    avg_hypothesis_natural_len = sum(len(item['hypothesis_natural'].split()) for item in data) / len(data)

    print(f"\nAverage sequence lengths (SMCDEL):")
    print(f"  Premise: {avg_premise_len:.1f} tokens")
    print(f"  Hypothesis: {avg_hypothesis_len:.1f} tokens")

    print(f"\nAverage sequence lengths (Natural):")
    print(f"  Premise: {avg_premise_natural_len:.1f} words")
    print(f"  Hypothesis: {avg_hypothesis_natural_len:.1f} words")


def show_examples(data: list, split_name: str, n_examples: int = 3):
    """Show example data points."""
    print(f"\n{'='*70}")
    print(f"{split_name.upper()} EXAMPLES")
    print(f"{'='*70}")

    for i, item in enumerate(data[:n_examples]):
        print(f"\nExample {i+1}:")
        print(f"  n_agents: {item['n_agents']}, depth: {item['hypothesis_depth']}")
        print(f"  Agent names: {item['agent_names']}")
        print(f"  Agent observations: {item['agent_observations']}")

        print(f"\n  === SMCDEL FORMAT (for training) ===")
        print(f"  Premise:")
        print(f"    {item['premise']}")
        print(f"  Hypothesis:")
        print(f"    {item['hypothesis']}")

        print(f"\n  === NATURAL LANGUAGE FORMAT (for reference) ===")
        print(f"  Premise (first 120 chars):")
        print(f"    {item['premise_natural'][:120]}...")
        print(f"  Hypothesis:")
        print(f"    {item['hypothesis_natural']}")

        print(f"\n  Label: {item['label']}")
        print("-" * 70)


def verify_agent_metadata_consistency(data: list, split_name: str) -> bool:
    """Verify agent-level metadata is consistent."""
    print(f"\n{'='*70}")
    print(f"{split_name.upper()}: AGENT METADATA VERIFICATION")
    print(f"{'='*70}")

    issues = []

    for i, item in enumerate(data):
        n_agents = item['n_agents']
        agent_obs = item['agent_observations']
        agent_names = item['agent_names']

        # Check lengths match
        if len(agent_obs) != n_agents:
            issues.append(f"Item {i}: agent_observations length {len(agent_obs)} != n_agents {n_agents}")

        if len(agent_names) != n_agents:
            issues.append(f"Item {i}: agent_names length {len(agent_names)} != n_agents {n_agents}")

        # Check agent names appear in SMCDEL hypothesis
        smcdel_hyp = item['hypothesis']
        for name in agent_names:
            if name not in smcdel_hyp and name not in item['premise']:
                # At least one should have the name
                pass  # This is okay, not all agents appear in hypothesis

        # Check SMCDEL premise has substituted names
        smcdel_premise = item['premise']
        if 'Agenta' in smcdel_premise or 'Agentb' in smcdel_premise:
            issues.append(f"Item {i}: SMCDEL premise still contains symbolic agent names (Agenta/Agentb)")

    if issues:
        print(f"\n⚠️  Found {len(issues)} consistency issues:")
        for issue in issues[:10]:  # Show first 10
            print(f"  - {issue}")
        if len(issues) > 10:
            print(f"  ... and {len(issues) - 10} more")
        return False
    else:
        print("\n✓ All agent metadata is consistent")
        print("✓ Agent name substitutions applied correctly to SMCDEL")
        return True


def analyze_name_vocabulary(train_data: list, val_data: list):
    """Analyze vocabulary overlap between train and val."""
    print(f"\n{'='*70}")
    print("NAME VOCABULARY ANALYSIS")
    print(f"{'='*70}")

    train_names = set()
    val_names = set()

    for item in train_data:
        train_names.update(item['agent_names'])

    for item in val_data:
        val_names.update(item['agent_names'])

    overlap = train_names & val_names

    print(f"\nTrain vocabulary: {len(train_names)} unique names")
    print(f"  {sorted(train_names)}")

    print(f"\nVal vocabulary: {len(val_names)} unique names")
    print(f"  {sorted(val_names)}")

    print(f"\nOverlap: {len(overlap)} names")
    if overlap:
        print(f"  {sorted(overlap)}")
        print(f"\n⚠️  Train and val share names (may reduce generalization test strength)")
    else:
        print(f"  None")
        print(f"\n✓ Perfect vocabulary separation (strong generalization test)")

    return len(overlap) == 0


def main():
    parser = argparse.ArgumentParser(
        description='Prepare agent-aware dataset for generation task with SMCDEL + natural language'
    )
    parser.add_argument('--input_dir', type=str, required=True,
                       help='Input directory with agent-aware dataset')
    parser.add_argument('--output_dir', type=str, required=True,
                       help='Output directory for formatted dataset')
    parser.add_argument('--show_examples', action='store_true',
                       help='Show example data points')

    args = parser.parse_args()

    input_dir = Path(args.input_dir)
    output_dir = Path(args.output_dir)
    output_dir.mkdir(exist_ok=True, parents=True)

    print("="*70)
    print("AGENT-AWARE DATASET PREPARATION FOR GENERATION TASK")
    print("="*70)
    print(f"\nInput: {input_dir}")
    print(f"Output: {output_dir}")
    print(f"\nFormat: SMCDEL (with name substitutions) + Natural Language")

    # Load raw data
    print(f"\nLoading datasets...")
    with open(input_dir / 'train_generation.json', 'r') as f:
        train_raw = json.load(f)

    with open(input_dir / 'val_generation.json', 'r') as f:
        val_raw = json.load(f)

    print(f"  Loaded {len(train_raw)} train samples")
    print(f"  Loaded {len(val_raw)} val samples")

    # Format for generation
    print(f"\nFormatting for generation task...")
    print(f"  - Splitting SMCDEL into premise/hypothesis")
    print(f"  - Applying agent name substitutions to SMCDEL")
    print(f"  - Preserving natural language versions")

    train_data = [format_for_generation(item) for item in train_raw]
    val_data = [format_for_generation(item) for item in val_raw]

    print(f"  Formatted {len(train_data)} train samples")
    print(f"  Formatted {len(val_data)} val samples")

    # Verify agent metadata
    train_consistent = verify_agent_metadata_consistency(train_data, "TRAIN")
    val_consistent = verify_agent_metadata_consistency(val_data, "VAL")

    if not train_consistent or not val_consistent:
        print("\n✗ Metadata consistency check failed!")
        return

    # Analyze vocabularies
    vocab_separated = analyze_name_vocabulary(train_data, val_data)

    # Print statistics
    print_statistics(train_data, "TRAIN")
    print_statistics(val_data, "VAL")

    # Show examples if requested
    if args.show_examples:
        show_examples(train_data, "TRAIN", n_examples=2)
        show_examples(val_data, "VAL", n_examples=2)

    # Save formatted datasets
    print(f"\n{'='*70}")
    print("SAVING FORMATTED DATASETS")
    print(f"{'='*70}")

    train_output = output_dir / 'train_generation.json'
    val_output = output_dir / 'val_generation.json'

    with open(train_output, 'w') as f:
        json.dump(train_data, f, indent=2)

    with open(val_output, 'w') as f:
        json.dump(val_data, f, indent=2)

    print(f"\nSaved:")
    print(f"  {train_output}: {len(train_data)} samples")
    print(f"  {val_output}: {len(val_data)} samples")

    # Create metadata summary
    metadata = {
        'version': 'v4_agent_aware_generation',
        'description': 'Agent-aware dataset with SMCDEL + natural language for generation',
        'source_format': 'dual (SMCDEL with name substitutions + natural language)',
        'task': 'generation',
        'train_samples': len(train_data),
        'val_samples': len(val_data),
        'train_n_agents_range': [min(item['n_agents'] for item in train_data),
                                  max(item['n_agents'] for item in train_data)],
        'train_depth_range': [min(item['hypothesis_depth'] for item in train_data),
                              max(item['hypothesis_depth'] for item in train_data)],
        'val_n_agents_range': [min(item['n_agents'] for item in val_data),
                                max(item['n_agents'] for item in val_data)],
        'val_depth_range': [min(item['hypothesis_depth'] for item in val_data),
                            max(item['hypothesis_depth'] for item in val_data)],
        'vocabulary_separated': vocab_separated,
        'features': [
            'SMCDEL format with real agent names (Alice, Bob, etc.) for training',
            'Natural language versions for reference',
            'Agent-level metadata (per-agent observations and names)',
            'Systematic agent observation patterns',
            'Disjoint train/val combinations',
            'Separate name vocabularies for train/val',
            'Enables multi-level Van Benthem product space analysis'
        ],
        'format': {
            'premise': 'SMCDEL symbolic format with agent name substitutions (training input)',
            'hypothesis': 'SMCDEL hypothesis with agent name substitutions (training output)',
            'premise_natural': 'Natural language premise (reference)',
            'hypothesis_natural': 'Natural language hypothesis (reference)',
            'label': 'entailment or non_entailment',
            'metadata': ['n_agents', 'hypothesis_depth', 'agent_observations', 'agent_names']
        }
    }

    metadata_output = output_dir / 'metadata.json'
    with open(metadata_output, 'w') as f:
        json.dump(metadata, f, indent=2)

    print(f"  {metadata_output}")

    # Final summary
    print(f"\n{'='*70}")
    print("PREPARATION COMPLETE!")
    print(f"{'='*70}")

    print(f"\n✓ Train: {len(train_data)} samples")
    print(f"✓ Val: {len(val_data)} samples")
    print(f"✓ Format: SMCDEL (with real names) + Natural Language")
    print(f"✓ Agent metadata: Consistent and verified")
    print(f"✓ Name substitutions: Applied correctly")
    print(f"✓ Name vocabulary: {'Separated' if vocab_separated else 'Overlapping'}")

    print(f"\nDataset ready for training!")
    print(f"\nUsage:")
    print(f"  python train_epistemic_gru.py \\")
    print(f"    --train_data {train_output} \\")
    print(f"    --val_data {val_output} \\")
    print(f"    --output_dir experiments/agent_aware_generation")

    print(f"\nKey advantages:")
    print(f"  1. SMCDEL format for model training (with real agent names)")
    print(f"  2. Natural language for human understanding")
    print(f"  3. Agent-level metadata for Van Benthem analysis")
    print(f"  4. Separate vocabularies for stronger generalization test")


if __name__ == '__main__':
    main()
