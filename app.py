from flask import Flask, render_template, jsonify
import matplotlib
matplotlib.use('Agg')  # Use non-interactive backend
import matplotlib.pyplot as plt
from io import BytesIO
import base64

# Import the CodeT5 evaluator module
from codet5_evaluation import CodeT5Evaluator

app = Flask(__name__)

# Initialize the CodeT5 evaluator
evaluator = CodeT5Evaluator()

@app.route('/')
def index():
    """Render the main dashboard page"""
    return render_template('index.html')

@app.route('/api/model_metrics')
def model_metrics():
    """Return the CodeT5 model metrics as JSON for API consumption"""
    report = evaluator.generate_report()
    return jsonify(report)

@app.route('/api/plot/performance')
def performance_plot():
    """Generate and return a performance plot"""
    # Create figure
    fig, ax = plt.subplots(figsize=(10, 6))
    
    # Get the brake detector results
    results = evaluator.brake_detector_results
    categories = list(results.keys())
    metrics = ['accuracy', 'precision', 'recall', 'f1']
    
    # Prepare data for plotting
    data = {
        metric: [results[cat][metric] for cat in categories]
        for metric in metrics
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
    ax.set_title('CodeT5 Performance on Documentation Aspects')
    ax.set_xticks([x + bar_width * 1.5 for x in index])
    ax.set_xticklabels([cat.replace('_', ' ').title() for cat in categories], rotation=45, ha='right')
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