import matplotlib.pyplot as plt
import matplotlib.patches as patches

def create_block_diagram():
    fig, ax = plt.subplots(figsize=(12, 6))
    ax.axis('off')
    
    # Define boxes
    boxes = {
        'Input': (0.05, 0.4, 0.1, 0.2),
        'White Balance': (0.2, 0.4, 0.15, 0.2),
        'Gamma': (0.4, 0.4, 0.1, 0.2),
        'Sharpening': (0.55, 0.4, 0.15, 0.2),
        'Fusion': (0.8, 0.4, 0.1, 0.2),
        'Output': (0.95, 0.4, 0.05, 0.2)
    }
    
    for name, (x, y, w, h) in boxes.items():
        rect = patches.Rectangle((x, y), w, h, linewidth=2, edgecolor='black', facecolor='skyblue')
        ax.add_patch(rect)
        ax.text(x + w/2, y + h/2, name, ha='center', va='center', wrap=True)
        
    # Arrows
    ax.arrow(0.15, 0.5, 0.05, 0, head_width=0.02, color='black') # Input -> WB
    ax.arrow(0.35, 0.5, 0.05, 0, head_width=0.02, color='black') # WB -> Gamma
    ax.arrow(0.5, 0.5, 0.05, 0, head_width=0.02, color='black') # Gamma -> Sharp
    ax.arrow(0.7, 0.5, 0.1, 0, head_width=0.02, color='black') # Sharp -> Fusion
    ax.arrow(0.9, 0.5, 0.05, 0, head_width=0.02, color='black') # Fusion -> Output
    
    plt.title("System Block Diagram")
    plt.savefig('report/block_diagram.png')
    plt.close()

def create_flowchart():
    fig, ax = plt.subplots(figsize=(10, 12))
    ax.axis('off')
    
    steps = [
        "Start",
        "Input Image",
        "White Balancing",
        "Gamma Correction",
        "Unsharp Masking",
        "Split Path",
        "CLAHE (Path A) | Hist. Lin (Path B)",
        "Weight Map Gen.",
        "Multiscale Fusion",
        "Output Image",
        "End"
    ]
    
    y_pos = 0.9
    for step in steps:
        rect = patches.Rectangle((0.3, y_pos), 0.4, 0.06, linewidth=2, edgecolor='black', facecolor='lightgreen')
        ax.add_patch(rect)
        ax.text(0.5, y_pos + 0.03, step, ha='center', va='center')
        
        if step != "End":
            ax.arrow(0.5, y_pos, 0, -0.04, head_width=0.02, color='black')
            
        y_pos -= 0.08

    plt.title("DSP Pipeline Flowchart")
    plt.savefig('report/flowchart.png')
    plt.close()

def create_use_case_diagram():
    fig, ax = plt.subplots(figsize=(8, 6))
    ax.axis('off')
    
    # User
    circle = patches.Circle((0.2, 0.5), 0.05, edgecolor='black', facecolor='white')
    ax.add_patch(circle)
    ax.text(0.2, 0.42, "User", ha='center')
    
    # System Boundary
    rect = patches.Rectangle((0.4, 0.2), 0.4, 0.6, linewidth=2, edgecolor='black', facecolor='none')
    ax.add_patch(rect)
    ax.text(0.6, 0.82, "Web Application", ha='center')
    
    # Use Cases
    cases = ["Upload Image", "View Comparison", "Download Result", "View Metrics"]
    y = 0.7
    for case in cases:
        ellipse = patches.Ellipse((0.6, y), 0.3, 0.1, edgecolor='black', facecolor='aliceblue')
        ax.add_patch(ellipse)
        ax.text(0.6, y, case, ha='center', va='center', fontsize=8)
        
        # Link User to Case
        ax.arrow(0.25, 0.5, 0.35-0.25, y-0.5, head_width=0.0, color='black')
        y -= 0.12
        
    plt.title("Use Case Diagram")
    plt.savefig('report/use_case_diagram.png')
    plt.close()

def create_system_architecture():
    fig, ax = plt.subplots(figsize=(10, 6))
    ax.axis('off')
    
    # Layers
    layers = [("Frontend", 0.7), ("Backend (Flask)", 0.4), ("DSP Engine", 0.1)]
    colors = ['lightyellow', 'lightcyan', 'lavender']
    
    for i, (name, y) in enumerate(layers):
        rect = patches.Rectangle((0.2, y), 0.6, 0.2, linewidth=2, edgecolor='black', facecolor=colors[i])
        ax.add_patch(rect)
        ax.text(0.5, y+0.1, name, ha='center', va='center', fontsize=12, fontweight='bold')
        
    # Arrows
    ax.arrow(0.5, 0.7, 0, -0.1, head_width=0.02, color='black') # Front -> Back
    ax.arrow(0.5, 0.4, 0, -0.1, head_width=0.02, color='black') # Back -> DSP
    
    plt.title("System Architecture")
    plt.savefig('report/system_architecture.png')
    plt.close()
    
def create_ui_wireframe():
    fig, ax = plt.subplots(figsize=(12, 8))
    ax.axis('off')
    
    # Browser Window
    rect = patches.Rectangle((0.1, 0.1), 0.8, 0.8, linewidth=2, edgecolor='black', facecolor='white')
    ax.add_patch(rect)
    
    # Navbar
    nav = patches.Rectangle((0.1, 0.8), 0.8, 0.1, facecolor='gray')
    ax.add_patch(nav)
    ax.text(0.15, 0.85, "AquaVision AI", color='white', va='center')
    
    # Comparison Area
    comp = patches.Rectangle((0.15, 0.4), 0.7, 0.35, edgecolor='black', facecolor='lightgray')
    ax.add_patch(comp)
    ax.text(0.5, 0.575, "Before / After Slider", ha='center')
    
    # Metrics
    for i in range(3):
        m = patches.Rectangle((0.15 + i*0.25, 0.2), 0.2, 0.1, edgecolor='black', facecolor='whitesmoke')
        ax.add_patch(m)
        ax.text(0.25 + i*0.25, 0.25, f"Metric {i+1}", ha='center')
        
    plt.title("UI Wireframe: Result Page")
    plt.savefig('report/ui_wireframe.png')
    plt.close()

if __name__ == "__main__":
    create_block_diagram()
    create_flowchart()
    create_use_case_diagram()
    create_system_architecture()
    create_ui_wireframe()
