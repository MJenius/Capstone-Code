import json
import matplotlib.pyplot as plt

def main():
    try:
        with open('collusion_curve.json', 'r') as f:
            data = json.load(f)
    except FileNotFoundError:
        print("Error: collusion_curve.json not found.")
        return

    n_values = []
    ncs = []
    
    for entry in data:
        n_values.append(int(entry['n']))
        ncs.append(float(entry['nc']))
        
    plt.figure(figsize=(8, 5))
    plt.plot(n_values, ncs, marker='D', linestyle='-', color='dodgerblue', linewidth=2, markersize=8)
    
    plt.xscale('log')
    plt.xticks(n_values, [str(n) for n in n_values])
    plt.ylim(0, 1.0)
    plt.xlabel('Number of Colluders (N) [Log Scale]')
    plt.ylabel('Normalized Correlation (NC)')
    plt.title('Collusion Robustness Curve (N=2 to 100)')
    plt.grid(True, which="both", ls="--", alpha=0.5)
    
    # Value annotations
    for n, nc in zip(n_values, ncs):
        plt.annotate(f"{nc:.4f}", (n, nc), textcoords="offset points", xytext=(0,10), ha='center')
        
    plt.tight_layout()
    plt.savefig('collusion_robustness_curve.png', dpi=300)
    print("Successfully generated collusion_robustness_curve.png")

if __name__ == '__main__':
    main()
