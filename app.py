from flask import Flask, render_template, jsonify
import matplotlib
matplotlib.use('Agg')  # Use non-interactive backend
import matplotlib.pyplot as plt
from io import BytesIO
import base64

app = Flask(__name__)

@app.route('/')
def index():
    """Render the main dashboard page"""
    return render_template('index.html')

@app.route('/api/model_metrics')
def model_metrics():
    """Return model metrics as JSON for API consumption"""
    # Simulated metrics for a generic AI model
    report = {
        "model_name": "AI Documentation Model",
        "model_size": "Base (220M parameters)",
        "accuracy": 0.87,
        "precision": 0.85,
        "recall": 0.88,
        "f1": 0.86,
        "summary": {
            "strengths": [
                "High accuracy in capturing technical details",
                "Strong performance on code summarization tasks"
            ],
            "limitations": [
                "Struggles with business logic (41% accuracy)",
                "Limited ability to understand domain-specific terminology (45%)"
            ]
        },
        # Human documentation performance (simulated)
        "human_performance": {
            "technical_details": 0.99,
            "business_logic": 0.98,
            "safety_implications": 0.95,
            "domain_terminology": 0.92
        },
        # AI performance (simulated)
        "ai_performance": {
            "technical_details": 0.87,
            "business_logic": 0.41,
            "safety_implications": 0.39,
            "domain_terminology": 0.45
        }
    }
    return jsonify(report)

@app.route('/api/plot/performance')
def performance_plot():
    """Generate and return a performance plot"""
    # Create figure
    fig, ax = plt.subplots(figsize=(10, 6))
    
    # Sample data for the plot (relevant evaluation categories)
    categories = ['Technical Details', 'Business Logic', 'Domain Terminology']
    metrics = ['accuracy', 'precision', 'recall', 'f1']
    
    # Sample data for each metric
    data = {
        metric: [0.87, 0.41, 0.45] if metric == 'accuracy' else [0.85, 0.38, 0.42] for metric in metrics
    }
    
    # Set width of bars
    bar_width = 0.2
    index = range(len(categories))  # Use range instead of np.arange
    
    # Plot bars
    for i, metric in enumerate(metrics):
        ax.bar(
            [x + i * bar_width for x in index],  # Adjust bar position manually
            data[metric], 
            bar_width,
            label=metric.capitalize()
        )
    
    # Customize plot
    ax.set_xlabel('Documentation Aspect')
    ax.set_ylabel('Score')
    ax.set_title('AI Model Performance on Documentation Tasks')
    ax.set_xticks([x + bar_width * 1.5 for x in index])
    ax.set_xticklabels([cat for cat in categories], rotation=45, ha='right')
    ax.legend()
    ax.set_ylim(0, 1.0)
    plt.tight_layout()
    
    # Save plot to a BytesIO object
    img = BytesIO()
    plt.savefig(img, format='png')
    img.seek(0)
    
    # Convert to base64 for embedding in HTML
    plot_url = base64.b64encode(img.getvalue()).decode('utf8')
    plt.close(fig)
    
    return jsonify({'plot_url': plot_url})

if __name__ == '__main__':
    app.run(debug=True, host='0.0.0.0', port=5000)
