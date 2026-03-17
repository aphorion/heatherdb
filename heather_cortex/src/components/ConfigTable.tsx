import type { EAMConfig } from '../api/types';

const CONFIG_DESCRIPTIONS: Record<string, string> = {
  d: 'Dimensionality',
  l_0: 'Initial locations',
  l_max: 'Max locations',
  k: 'k-NN activation',
  eta_0: 'Initial learning rate',
  lambda: 'Learning rate decay',
  eta_min: 'Min learning rate',
  tau_split: 'Novelty split threshold',
  tau_merge: 'Merge threshold',
  gamma: 'Conscience factor',
  tau_damp: 'Damping time constant',
  tau_overload: 'Overload split threshold',
  beta: 'Softmax temperature',
  t_max: 'Max Hopfield iterations',
  epsilon: 'Convergence threshold',
};

interface ConfigTableProps {
  config: EAMConfig;
}

export function ConfigTable({ config }: ConfigTableProps) {
  return (
    <div className="overflow-hidden rounded-lg border border-border-subtle">
      <table className="w-full text-sm">
        <thead>
          <tr className="border-b border-border-subtle bg-bg-tertiary">
            <th className="text-left px-4 py-2 label">Parameter</th>
            <th className="text-left px-4 py-2 label">Description</th>
            <th className="text-right px-4 py-2 label">Value</th>
          </tr>
        </thead>
        <tbody>
          {Object.entries(config).map(([key, value]) => (
            <tr key={key} className="border-b border-border-subtle last:border-0 hover:bg-bg-tertiary/50 transition-colors">
              <td className="px-4 py-2 font-mono text-text-primary">{key}</td>
              <td className="px-4 py-2 text-text-secondary">{CONFIG_DESCRIPTIONS[key] || key}</td>
              <td className="px-4 py-2 text-right font-mono text-text-primary">{typeof value === 'number' ? value.toLocaleString(undefined, { maximumFractionDigits: 10 }) : String(value)}</td>
            </tr>
          ))}
        </tbody>
      </table>
    </div>
  );
}
