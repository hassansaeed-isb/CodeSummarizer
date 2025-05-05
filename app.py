from flask import Flask, render_template

app = Flask(__name__)

@app.route('/')
def about():
    """Render the About the Research page"""
    return render_template('about.html')

if __name__ == '__main__':
    app.run(debug=True, host='0.0.0.0', port=5000)
