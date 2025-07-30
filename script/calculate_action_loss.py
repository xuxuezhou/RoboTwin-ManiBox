#!/usr/bin/env python3
"""
Calculate Action Loss Script

This script calculates the loss between recorded actions from evaluation
and reference actions from HDF5 demonstration data.
"""

import os
import sys
import json
import numpy as np
import argparse
from typing import List, Dict, Any, Tuple
import matplotlib.pyplot as plt
from pathlib import Path


def load_action_log(log_path: str) -> List[Dict[str, Any]]:
    """Load action log from JSONL file"""
    actions = []
    with open(log_path, 'r') as f:
        for line in f:
            if line.strip():
                actions.append(json.loads(line))
    return actions


def load_reference_data(hdf5_path: str, max_steps: int = 40) -> np.ndarray:
    """Load reference joint actions from HDF5 file"""
    import h5py
    
    with h5py.File(hdf5_path, 'r') as f:
        # Try to load joint_action/vector
        if 'joint_action' in f and 'vector' in f['joint_action']:
            joint_actions = f['joint_action/vector'][:max_steps]
            print(f"Loaded {len(joint_actions)} reference actions from joint_action/vector")
            return joint_actions
        else:
            raise ValueError("No joint_action/vector found in HDF5 file")


def calculate_losses(recorded_actions: List[Dict], reference_actions: np.ndarray) -> Dict[str, Any]:
    """Calculate various loss metrics between recorded and reference actions"""
    
    # Extract action arrays
    recorded_arrays = []
    for action_entry in recorded_actions:
        if 'action' in action_entry:
            recorded_arrays.append(np.array(action_entry['action']))
    
    if not recorded_arrays:
        raise ValueError("No valid actions found in log file")
    
    recorded_actions_array = np.array(recorded_arrays)
    
    # Ensure we have the same number of steps
    min_steps = min(len(recorded_actions_array), len(reference_actions))
    recorded_actions_array = recorded_actions_array[:min_steps]
    reference_actions_array = reference_actions[:min_steps]
    
    print(f"Comparing {min_steps} steps")
    print(f"Recorded actions shape: {recorded_actions_array.shape}")
    print(f"Reference actions shape: {reference_actions_array.shape}")
    
    # Calculate various loss metrics
    losses = {}
    
    # Mean Squared Error (MSE)
    mse = np.mean((recorded_actions_array - reference_actions_array) ** 2)
    losses['mse'] = float(mse)
    
    # Root Mean Squared Error (RMSE)
    losses['rmse'] = float(np.sqrt(mse))
    
    # Mean Absolute Error (MAE)
    mae = np.mean(np.abs(recorded_actions_array - reference_actions_array))
    losses['mae'] = float(mae)
    
    # Per-joint losses
    joint_mse = np.mean((recorded_actions_array - reference_actions_array) ** 2, axis=0)
    losses['joint_mse'] = joint_mse.tolist()
    
    joint_mae = np.mean(np.abs(recorded_actions_array - reference_actions_array), axis=0)
    losses['joint_mae'] = joint_mae.tolist()
    
    # Per-timestep losses
    timestep_mse = np.mean((recorded_actions_array - reference_actions_array) ** 2, axis=1)
    losses['timestep_mse'] = timestep_mse.tolist()
    
    timestep_mae = np.mean(np.abs(recorded_actions_array - reference_actions_array), axis=1)
    losses['timestep_mae'] = timestep_mae.tolist()
    
    # Correlation analysis
    correlations = []
    for i in range(recorded_actions_array.shape[1]):
        corr = np.corrcoef(recorded_actions_array[:, i], reference_actions_array[:, i])[0, 1]
        correlations.append(float(corr) if not np.isnan(corr) else 0.0)
    losses['correlations'] = correlations
    
    return losses


def plot_loss_analysis(recorded_actions: List[Dict], reference_actions: np.ndarray, 
                      losses: Dict[str, Any], output_dir: str = None):
    """Create plots for loss analysis"""
    
    # Extract action arrays
    recorded_arrays = []
    for action_entry in recorded_actions:
        if 'action' in action_entry:
            recorded_arrays.append(np.array(action_entry['action']))
    
    recorded_actions_array = np.array(recorded_arrays)
    min_steps = min(len(recorded_actions_array), len(reference_actions))
    recorded_actions_array = recorded_actions_array[:min_steps]
    reference_actions_array = reference_actions[:min_steps]
    
    # Create plots
    fig, axes = plt.subplots(2, 2, figsize=(15, 10))
    fig.suptitle('Action Loss Analysis', fontsize=16)
    
    # Plot 1: Action trajectories comparison
    timesteps = range(min_steps)
    axes[0, 0].plot(timesteps, recorded_actions_array, 'b-', alpha=0.7, label='Recorded')
    axes[0, 0].plot(timesteps, reference_actions_array, 'r-', alpha=0.7, label='Reference')
    axes[0, 0].set_title('Action Trajectories Comparison')
    axes[0, 0].set_xlabel('Timestep')
    axes[0, 0].set_ylabel('Action Value')
    axes[0, 0].legend()
    axes[0, 0].grid(True)
    
    # Plot 2: Per-timestep MSE
    axes[0, 1].plot(timesteps, losses['timestep_mse'], 'g-')
    axes[0, 1].set_title('Per-Timestep MSE')
    axes[0, 1].set_xlabel('Timestep')
    axes[0, 1].set_ylabel('MSE')
    axes[0, 1].grid(True)
    
    # Plot 3: Per-joint MSE
    joint_indices = range(len(losses['joint_mse']))
    axes[1, 0].bar(joint_indices, losses['joint_mse'])
    axes[1, 0].set_title('Per-Joint MSE')
    axes[1, 0].set_xlabel('Joint Index')
    axes[1, 0].set_ylabel('MSE')
    axes[1, 0].grid(True)
    
    # Plot 4: Correlation coefficients
    axes[1, 1].bar(joint_indices, losses['correlations'])
    axes[1, 1].set_title('Per-Joint Correlation')
    axes[1, 1].set_xlabel('Joint Index')
    axes[1, 1].set_ylabel('Correlation')
    axes[1, 1].grid(True)
    axes[1, 1].axhline(y=0, color='k', linestyle='-', alpha=0.3)
    
    plt.tight_layout()
    
    if output_dir:
        os.makedirs(output_dir, exist_ok=True)
        plot_path = os.path.join(output_dir, 'action_loss_analysis.png')
        plt.savefig(plot_path, dpi=300, bbox_inches='tight')
        print(f"📊 Loss analysis plot saved to: {plot_path}")
    else:
        plt.show()


def save_loss_report(losses: Dict[str, Any], output_path: str):
    """Save detailed loss report to JSON file"""
    report = {
        'summary': {
            'mse': losses['mse'],
            'rmse': losses['rmse'],
            'mae': losses['mae'],
            'mean_correlation': np.mean(losses['correlations'])
        },
        'detailed_losses': losses
    }
    
    with open(output_path, 'w') as f:
        json.dump(report, f, indent=2)
    
    print(f"📄 Loss report saved to: {output_path}")


def main():
    parser = argparse.ArgumentParser(description='Calculate action loss between recorded and reference actions')
    parser.add_argument('--action-log', type=str, required=True, 
                       help='Path to action log file (JSONL format)')
    parser.add_argument('--reference-data', type=str, required=True,
                       help='Path to reference HDF5 file')
    parser.add_argument('--max-steps', type=int, default=40,
                       help='Maximum number of steps to compare (default: 40)')
    parser.add_argument('--output-dir', type=str, default='./loss_analysis',
                       help='Output directory for plots and reports (default: ./loss_analysis)')
    parser.add_argument('--plot', action='store_true',
                       help='Generate loss analysis plots')
    
    args = parser.parse_args()
    
    # Check input files
    if not os.path.exists(args.action_log):
        print(f"❌ Action log file not found: {args.action_log}")
        return
    
    if not os.path.exists(args.reference_data):
        print(f"❌ Reference data file not found: {args.reference_data}")
        return
    
    try:
        # Load data
        print(f"📖 Loading action log: {args.action_log}")
        recorded_actions = load_action_log(args.action_log)
        print(f"   Loaded {len(recorded_actions)} recorded actions")
        
        print(f"📖 Loading reference data: {args.reference_data}")
        reference_actions = load_reference_data(args.reference_data, args.max_steps)
        print(f"   Loaded {len(reference_actions)} reference actions")
        
        # Calculate losses
        print("🔢 Calculating losses...")
        losses = calculate_losses(recorded_actions, reference_actions)
        
        # Print summary
        print("\n📊 Loss Summary:")
        print(f"   MSE: {losses['mse']:.6f}")
        print(f"   RMSE: {losses['rmse']:.6f}")
        print(f"   MAE: {losses['mae']:.6f}")
        print(f"   Mean Correlation: {np.mean(losses['correlations']):.6f}")
        
        # Save report
        os.makedirs(args.output_dir, exist_ok=True)
        report_path = os.path.join(args.output_dir, 'loss_report.json')
        save_loss_report(losses, report_path)
        
        # Generate plots
        if args.plot:
            print("📊 Generating plots...")
            plot_loss_analysis(recorded_actions, reference_actions, losses, args.output_dir)
        
        print(f"✅ Loss analysis completed! Results saved to: {args.output_dir}")
        
    except Exception as e:
        print(f"❌ Error during loss calculation: {e}")
        import traceback
        traceback.print_exc()


if __name__ == "__main__":
    main() 